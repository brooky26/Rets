"""
Live paper-trading pipeline orchestrator — the live equivalent of what
`WalkForwardBacktester` already does against historical data.

Per-candle flow (see `on_candle`):
  1. Settle any pending trade for this symbol first. Two genuinely
     different paths, chosen by whether the pending trade has a real
     `contract_id` (see `paper_trading.types.PendingTrade`):
       - Paper trade (`contract_id is None`): settled fictionally, using
         this candle's close vs the entry candle's close (see
         paper_trading_schema module docstring for why settlement is
         defined this way).
       - Live trade (`contract_id` set): Deriv, not this code, decides
         win/loss/payout on its own real `duration_ticks` clock. This
         polls `ContractOutcomeTracker` for that contract's real
         settlement; if it hasn't settled yet, the trade simply stays
         pending and NO new trade is opened on that symbol this candle
         (one open position per symbol at a time) — it is not, and must
         not be, resolved by the next-candle fiction.
  2. If nothing is still pending after step 1: classify regime ->
     estimate probability -> compute EV -> assess risk -> score the
     opportunity -> CROSS-SYMBOL RANKING (see below) -> execute -> if
     bought, open a new pending trade (paper trades settle next candle;
     live trades await real settlement via step 1 on subsequent candles).

Cross-symbol ranking (the fix for the "isn't a true argmax" bug):
candles for different symbols do not complete simultaneously — each is
driven by its own independent tick stream, so there is no single moment
where all 5 symbols can be compared at once. What this does instead:
every evaluation (approved or not) updates `_latest_evaluations[symbol]`
with that symbol's quality_score and the epoch it was computed at. When a
symbol's opportunity is independently approved on its own merits, it is
only actually traded if its quality_score is the highest among all
OTHER symbols that currently have no pending trade (i.e. are actually
available to compete for capital right now) — using each of those
symbols' most recently known score, which may be up to one candle-period
stale for symbols whose own candle hasn't completed as recently. This is
an approximate, not exact, simultaneous argmax, for that inherent
timing reason — but it directly fixes the previous bug, where a symbol
could get traded while a genuinely stronger, currently-untraded symbol
sat idle with no comparison ever taking place.

A single, SHARED `TradeOpportunityScorer` (not one per symbol) backs
this — instantiating one per symbol was the direct cause of the
degenerate lock-in bug this replaces: each symbol's own adaptive
threshold (bucketed per regime) could independently random-walk down
toward `threshold_min`, and whichever symbol's threshold happened to
collapse first would then clear its own near-zero bar on almost every
candle, regardless of its raw signal quality relative to other symbols.
A shared scorer means the regime-bucketed threshold reflects genuine
aggregate trade-setup frequency across the whole account, not one
symbol's independent random walk.

RL Trade Management (Level 7) is NOT wired in here — v1 holds every
paper trade to its one-candle settlement. See paper_trading_schema's
module docstring for the explicitly agreed v2 scope.

One `PaperTradingOrchestrator` instance owns ALL configured symbols
(rather than one per symbol), mirroring the account-level reality that
there is one risk/equity curve across every instrument traded. Unlike
the scorer, each symbol DOES still get its own probability model —
signal quality genuinely differs per instrument, unlike the trade-setup
frequency the scorer's threshold is calibrating against.

Optional Ensemble Fusion (Bayesian Logistic + Bagged GBM + Monte Carlo)
------------------------------------------------------------------------------
By default (no `fusion_config` passed to `__init__`), this class
behaves EXACTLY as before this feature was added: one
`BayesianLogisticRegression` per symbol, its `ProbabilityEstimate` fed
straight to the EV engine. Passing `fusion_config` (and, optionally,
`bagged_gbm_config` / `monte_carlo_config`) switches every symbol over
to combining evidence from however many of {Bayesian Logistic, Bagged
GBM, Monte Carlo GBM price-path} are configured via
`EnsembleFusionEngine`, using regime-aware weights from a `WeightLearner`
(itself normally kept up to date by the Continuous Learning
Orchestrator's Bayesian weight optimization — see
`continuous_learning/orchestrator.py`). Sequence models (LSTM/Transformer)
are deliberately NOT wired into this live per-candle loop directly (they
are trained/promoted via the Continuous Learning Orchestrator, which is
where the heavier torch-based training belongs); a production deployment
that wants their live predictions in the fused decision would extend
`_predict_fused` below to also consult whatever `ModelRegistry` champion
is currently live for those model types.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np

from configs.duration_selection_schema import DurationSelectionConfig
from configs.ensemble_schema import EnsembleFusionConfig
from configs.ev_schema import ExpectedValueConfig
from configs.execution_schema import ExecutionConfig
from configs.meta_learning_schema import MetaLearningConfig
from configs.monte_carlo_schema import MonteCarloPricePathConfig
from configs.opportunity_schema import OpportunityScoringConfig
from configs.paper_trading_schema import PaperTradingConfig
from configs.post_trade_schema import PostTradeAnalysisConfig
from configs.probability_schema import BayesianLogisticConfig, BaggedGBMConfig
from configs.risk_schema import RiskConfig
from data.types import Candle
from ensemble.fusion_engine import EnsembleFusionEngine, monte_carlo_result_to_probability_estimate
from execution.engine import ExecutionEngine
from execution.outcome_tracker import ContractOutcomeTracker
from execution.types import BrokerClient
from expected_value.engine import ExpectedValueEngine
from expected_value.types import ContractSpec, ContractType
from features.types import FeatureVector
from meta_learning.sufficiency import apply_sufficiency_scaling, compute_sufficiency_ratio
from meta_learning.types import EnsembleWeights
from meta_learning.weight_learner import WeightLearner
from monte_carlo.duration_selector import DurationSelector
from monte_carlo.price_paths import MonteCarloPricePathSimulator, estimate_gbm_parameters_from_returns
from opportunity.scorer import TradeOpportunityScorer
from paper_trading.types import PendingTrade, opportunity_to_pending_trade
from post_trade.analyzer import PostTradeAnalyzer
from post_trade.types import CompletedTrade, PerformanceMetrics
from probability.bayesian_logistic import BayesianLogisticRegression
from probability.gbm import BaggedGBMEstimator
from probability.types import ProbabilityEstimate
from regime.hmm_detector import GaussianHMMRegimeDetector
from regime.rule_based import RuleBasedRegimeDetector
from regime.types import RegimeLabel
from risk.engine import RiskEngine
from risk.types import TradeOutcome
from state_encoder.encoder import MarketStateEncoder
from state_encoder.types import MarketState

logger = logging.getLogger("paper_trading")
NAN = float("nan")


@dataclass(frozen=True, slots=True)
class SymbolEvaluation:
    """Most recently computed opportunity evaluation for one symbol —
    tracked for every symbol on every candle, regardless of approval, so
    cross-symbol ranking always has each symbol's latest known standing
    even for symbols whose own candle hasn't just completed.

    `veto_reasons` carries `TradeOpportunity.veto_reasons` through to this
    ranking snapshot — needed because `quality_score` alone can't
    distinguish "below threshold" from "scored well but vetoed anyway"
    (a hard EV/Risk gate, independent of quality_score — see
    opportunity/scorer.py). Empty for candles that never reached full
    scoring (e.g. a regime-consensus-gate skip, or before bootstrap)."""

    quality_score: float
    epoch: int
    approved: bool
    veto_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SignalReading:
    """One signal source's contribution to a single trade decision — either
    a probability model (bayesian_logistic, bagged_gbm, monte_carlo_gbm) or
    a regime detector (rule_based, hmm). `expected_direction` is only
    meaningful for probability models (+1/-1/0); regime detectors don't
    have a directional prediction, so it's left at 0 and `agrees` is
    reported as None (not False) for those rows — "not applicable" is a
    different answer than "disagrees", and collapsing them would be
    misleading in the logged table.
    """

    name: str
    kind: str  # "probability" | "regime"
    detail: str  # e.g. "up" / "down" / "no edge", or a RegimeLabel value
    confidence: float
    expected_direction: int = 0


class PaperTradingOrchestrator:
    def __init__(
        self,
        paper_config: PaperTradingConfig,
        probability_config: BayesianLogisticConfig,
        ev_config: ExpectedValueConfig,
        risk_config: RiskConfig,
        opportunity_config: OpportunityScoringConfig,
        post_trade_config: PostTradeAnalysisConfig,
        execution_config: ExecutionConfig,
        platform_environment: str,
        regime_detector: RuleBasedRegimeDetector,
        state_encoder: MarketStateEncoder,
        broker_client: BrokerClient | None = None,
        fusion_config: EnsembleFusionConfig | None = None,
        meta_learning_config: MetaLearningConfig | None = None,
        bagged_gbm_config: BaggedGBMConfig | None = None,
        monte_carlo_config: MonteCarloPricePathConfig | None = None,
        weight_learner: WeightLearner | None = None,
        duration_selection_config: DurationSelectionConfig | None = None,
        challenger_regime_detector: GaussianHMMRegimeDetector | None = None,
        sequence_model_estimators: dict[str, object] | None = None,
    ) -> None:
        self._paper_config = paper_config
        self._probability_config = probability_config
        self._bagged_gbm_config = bagged_gbm_config
        self._monte_carlo_config = monte_carlo_config
        self._regime_detector = regime_detector
        self._challenger_regime_detector = challenger_regime_detector
        self._sequence_model_estimators = sequence_model_estimators or {}
        self._state_encoder = state_encoder

        self._ev_engine = ExpectedValueEngine(ev_config)
        self._risk_engine = RiskEngine(risk_config, starting_equity=paper_config.starting_equity)
        self._post_trade_analyzer = PostTradeAnalyzer(post_trade_config)
        self._execution_engine = ExecutionEngine(
            config=execution_config,
            platform_environment=platform_environment,
            broker_client=broker_client,
        )
        # Shared across all symbols — see module docstring for why one
        # per symbol was the direct cause of the degenerate-lock-in bug
        # this replaces.
        self._scorer = TradeOpportunityScorer(opportunity_config)
        # Only needed once trades can actually go live (real contract_id
        # present) — no broker_client means every trade stays paper mode,
        # so nothing ever needs to poll for a real settlement.
        self._outcome_tracker = ContractOutcomeTracker(broker_client) if broker_client is not None else None
        self._contract = ContractSpec(
            contract_type=ContractType.RISE_FALL,
            stake=paper_config.stake,
            payout=paper_config.stake * paper_config.assumed_payout_ratio,
            duration_ticks=paper_config.duration_ticks,
        )

        # Ensemble fusion is entirely opt-in — see module docstring's
        # "Optional Ensemble Fusion" section. `fusion_config is None`
        # (the default) means every code path below that checks
        # `self._fusion_engine is not None` is skipped, and behavior is
        # identical to before this feature existed.
        self._fusion_engine = EnsembleFusionEngine(fusion_config) if fusion_config is not None else None
        model_names = ["bayesian_logistic"]
        if bagged_gbm_config is not None:
            model_names.append("bagged_gbm")
        if monte_carlo_config is not None:
            model_names.append("monte_carlo_gbm")
        self._weight_learner = weight_learner or (
            WeightLearner(meta_learning_config or MetaLearningConfig(model_names=model_names))
            if self._fusion_engine is not None
            else None
        )
        self._mc_simulator = MonteCarloPricePathSimulator(monte_carlo_config) if monte_carlo_config is not None else None

        # Opt-in, same pattern as ensemble fusion above: duration_selection_config
        # is None by default, so self._duration_selector is None and on_candle's
        # duration handling reproduces the original static self._contract behavior
        # exactly. Reuses self._mc_simulator (may itself be None — DurationSelector
        # handles that by using the Hurst fallback for every candidate, see its
        # own module docstring).
        self._duration_selector = (
            DurationSelector(duration_selection_config, ev_config, self._mc_simulator)
            if duration_selection_config is not None
            else None
        )

        self._probability_models: dict[str, BayesianLogisticRegression] = {}
        self._bagged_gbm_models: dict[str, BaggedGBMEstimator] = {}
        self._recent_closes: dict[str, list[float]] = {}
        self._pending_trades: dict[str, PendingTrade | None] = {}
        self._latest_evaluations: dict[str, SymbolEvaluation] = {}
        # symbol -> model_name -> sufficiency ratio (see meta_learning/sufficiency.py).
        # Populated in bootstrap(); consulted by _predict_probability() when fusion
        # is active, to shrink (not sideline) undertrained models' fusion weight.
        self._sufficiency: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------ #
    # Bootstrap
    # ------------------------------------------------------------------ #

    def bootstrap(self, symbol: str, historical_states: list[MarketState], closes: list[float]) -> bool:
        """
        Fits `symbol`'s initial probability model from historical
        (state, close) pairs — the caller is responsible for producing
        `historical_states` by replaying historical candles through the
        SAME live feature-pipeline/state-encoder instances that will keep
        running afterward, so rolling-window warm-up carries continuously
        into live streaming rather than resetting.

        Returns True if enough usable states were available to fit (per
        `paper_config.min_bootstrap_candles`); False (logged) if not — the
        symbol is simply skipped until enough live candles accumulate to
        retry via a later `bootstrap()` call.
        """
        if len(historical_states) != len(closes):
            raise ValueError("historical_states and closes must be the same length")

        if len(historical_states) < self._paper_config.min_bootstrap_candles + 1:
            logger.warning(
                "%s: only %d usable historical states available (need >= %d) — "
                "skipping bootstrap for now.",
                symbol, len(historical_states), self._paper_config.min_bootstrap_candles + 1,
            )
            return False

        usable_states = historical_states[:-1]
        labels = (np.diff(np.array(closes)) > 0).astype(int)
        X = np.array(
            [[getattr(s, dim) for dim in self._probability_config.feature_dims] for s in usable_states]
        )

        model = BayesianLogisticRegression(self._probability_config)
        model.fit(X, labels)
        self._probability_models[symbol] = model

        # bayesian_logistic's own floor (min_bootstrap_candles) already represents
        # "enough for this simple model" by design — reaching it at all (which
        # bootstrap() requires to get this far) means sufficiency=1.0, no shrinkage.
        sufficiency: dict[str, float] = {"bayesian_logistic": 1.0}

        if self._bagged_gbm_config is not None:
            X_gbm = np.array(
                [[getattr(s, dim) for dim in self._bagged_gbm_config.feature_dims] for s in usable_states]
            )
            self._bagged_gbm_models[symbol] = BaggedGBMEstimator(self._bagged_gbm_config).fit(X_gbm, labels)
            sufficiency["bagged_gbm"] = compute_sufficiency_ratio(
                len(usable_states), self._paper_config.bagged_gbm_target_samples,
            )

        self._sufficiency[symbol] = sufficiency

        if self._monte_carlo_config is not None:
            window = self._monte_carlo_config.mu_estimation_window
            self._recent_closes[symbol] = list(closes[-window:])

        self._pending_trades[symbol] = None
        logger.info("%s: bootstrap fit complete on %d historical states.", symbol, len(usable_states))
        return True

    def is_bootstrapped(self, symbol: str) -> bool:
        return symbol in self._probability_models

    async def register_late_contract(self, context: str, buy_result: dict) -> None:
        """
        Receiving end of `DerivWebSocketClient`'s `on_late_contract`
        callback (see data/deriv_client.py's `_on_late_buy_response`): a
        buy already timed out locally and was reported as an error, but
        Deriv's real response then arrived showing a contract WAS opened.
        Without this, that contract was only ever logged — never
        registered anywhere, so it would never settle, never update
        equity, and never appear in post-trade analytics, despite being a
        real open position.

        Honest limitation: by the time this fires, the original
        `TradeOpportunity` context (direction, regime, predicted
        probability, quality score) that led to the buy is gone — it lived
        in `on_candle`'s local variables for the call that errored out, not
        anywhere this method can reach. This registers the contract with
        placeholder values for those descriptive fields (clearly marked
        below) so it can still be correctly settled and counted in
        equity/PnL — which does not depend on them (see
        `_poll_live_settlement_if_any`, which settles from Deriv's own
        reported pnl, never from `pending.direction`). Only the descriptive
        analytics fields on the resulting `CompletedTrade` are unknown, not
        the financial outcome.
        """
        if not context.startswith("buy_direct:"):
            # A real buy() (two-step proposal+buy) call has no bare symbol in
            # `context` (it's a proposal_id) — there's no `_pending_trades` key
            # to register this under. Surfaced loudly rather than guessed at.
            logger.warning(
                "Late contract recovered from a two-step buy() call (context=%s) — "
                "cannot auto-register (no symbol available from a proposal_id). "
                "contract_id=%s is real and open; reconcile it against the account manually.",
                context, buy_result.get("contract_id"),
            )
            return

        symbol = context.removeprefix("buy_direct:")
        contract_id = buy_result.get("contract_id")
        if contract_id is None:
            logger.warning(
                "%s: register_late_contract called with no contract_id in buy_result (%s) — "
                "nothing to register.", symbol, buy_result,
            )
            return

        if self._outcome_tracker is None:
            logger.warning(
                "%s: late contract_id=%s recovered, but no broker_client/outcome_tracker is "
                "configured — cannot poll for settlement. Reconcile against the account manually.",
                symbol, contract_id,
            )
            return

        existing = self._pending_trades.get(symbol)
        if existing is not None:
            logger.warning(
                "%s: late contract_id=%s recovered, but a trade is already pending for this "
                "symbol (contract_id=%s) — refusing to overwrite it. contract_id=%s is still real "
                "and open; reconcile it against the account manually.",
                symbol, contract_id, existing.contract_id, contract_id,
            )
            return

        self._pending_trades[symbol] = PendingTrade(
            symbol=symbol,
            entry_epoch=-1,  # unknown — see docstring; not used by live (broker-polled) settlement
            entry_close=0.0,  # unknown — likewise unused by live settlement
            direction=0,  # unknown — real pnl comes from Deriv directly, not this field
            stake=buy_result.get("buy_price", 0.0) or 0.0,
            payout=buy_result.get("payout", 0.0) or 0.0,
            predicted_probability=NAN,  # unknown — original opportunity context was lost
            regime_at_entry=RegimeLabel.RANDOM_WALK,  # placeholder — see docstring, not "no edge"
            quality_score_at_entry=0.0,  # unknown
            contract_id=contract_id,
        )
        logger.info(
            "%s: late contract_id=%s registered for live settlement polling (stake=%.2f). "
            "Descriptive fields (direction/regime/probability/quality_score) are placeholders — "
            "the original decision context was lost when the buy call timed out — but the real "
            "pnl will be polled from Deriv and correctly applied to equity and post-trade stats.",
            symbol, contract_id, buy_result.get("buy_price", 0.0) or 0.0,
        )

    # ------------------------------------------------------------------ #
    # Live per-candle decision loop
    # ------------------------------------------------------------------ #

    async def on_candle(self, symbol: str, candle: Candle, vector: FeatureVector | None) -> dict:
        """
        Called once per completed candle for `symbol`. `vector` is whatever
        the live feature pipeline produced for this candle (None if still
        warming up on rolling windows — nothing to do yet either way).

        Returns {"settled": CompletedTrade | None, "decision": ExecutionDecision | None}
        for logging; never raises for ordinary "nothing to do" cases.
        """
        result: dict = {
            "settled": None, "decision": None, "rankings": None,
            "duration_selection": None, "regime_consensus": None,
            "signal_agreement": None,
        }

        if symbol not in self._probability_models:
            return result

        if self._mc_simulator is not None:
            window = self._monte_carlo_config.mu_estimation_window
            buf = self._recent_closes.setdefault(symbol, [])
            buf.append(candle.close)
            if len(buf) > window:
                del buf[: len(buf) - window]

        pending = self._pending_trades.get(symbol)
        if pending is not None and pending.is_awaiting_real_settlement:
            settled_trade = await self._poll_live_settlement_if_any(symbol, pending, candle)
        else:
            settled_trade = self._settle_pending_trade_if_any(symbol, candle)
        result["settled"] = settled_trade

        if self._pending_trades.get(symbol) is not None:
            # A live trade is still awaiting its real Deriv settlement —
            # never open a second concurrent position on the same symbol
            # while one is outstanding.
            return result

        if vector is None:
            return result

        state = self._state_encoder.encode(vector)
        if not state.is_valid:
            return result

        regime_rule_based = self._regime_detector.classify(state)

        # Rule-based is informational-only (see module docstring's "Regime
        # consensus" section for the full history: it used to hold equal veto
        # power alongside the HMM in a hard consensus gate, retired after real
        # disagreement-rate evidence — ~158 disagreements in 1.5 hours of live
        # trading — suggested rule-based's memoryless, fixed-threshold reads
        # were adding noise rather than a genuine second opinion against a
        # detector with real temporal memory and a wider feature set). HMM
        # becomes the OPERATIVE regime (fed into probability estimation,
        # duration selection, and scoring) once a challenger is registered;
        # rule-based's own classification is still computed and reported in
        # every result["regime_consensus"], just never blocks or drives
        # anything by itself anymore. Falls back to rule-based as the
        # operative regime only when no HMM challenger is registered yet
        # (e.g. before the continuous-learning cron's first successful run) —
        # same safe-fallback spirit as everywhere else in this file.
        if self._challenger_regime_detector is not None:
            regime_hmm = self._challenger_regime_detector.classify(state)
            regime = regime_hmm
            result["regime_consensus"] = {
                "rule_based": regime_rule_based.regime, "hmm": regime_hmm.regime,
                "agree": regime_hmm.regime == regime_rule_based.regime,
                "operative_source": "hmm",
            }
        else:
            regime = regime_rule_based
            result["regime_consensus"] = {
                "rule_based": regime_rule_based.regime, "hmm": None, "agree": None,
                "operative_source": "rule_based",
            }

        # Reference-horizon MC evidence (self._contract.duration_ticks) for fusion
        # input, since fusion must run BEFORE any duration is chosen — direction
        # is unknown yet, so this uses direction=1 (see the method's own docstring
        # for why that's still directionally meaningful for fusion purposes).
        mc_result = self._compute_mc_result_if_configured(symbol, state, ev_direction_hint=None)
        probability, signal_readings = self._predict_probability(symbol, state, regime, mc_result)
        signal_readings = [
            SignalReading(
                name="regime_rule_based", kind="regime",
                detail=regime_rule_based.regime.value, confidence=regime_rule_based.confidence,
            ),
            *(
                [SignalReading(
                    name="regime_hmm", kind="regime",
                    detail=regime_hmm.regime.value, confidence=regime_hmm.confidence,
                )]
                if self._challenger_regime_detector is not None else []
            ),
            *signal_readings,
        ]

        if self._duration_selector is not None:
            mu, sigma = self._estimate_gbm_params(symbol)
            duration_result = self._duration_selector.select(
                symbol, state.epoch, current_price=candle.close, fused_probability=probability,
                mu_per_tick=mu, sigma_per_tick=sigma, hurst_persistence=state.persistence,
                stake=self._paper_config.stake, assumed_payout_ratio=self._paper_config.assumed_payout_ratio,
            )
            result["duration_selection"] = duration_result
            if duration_result.chosen is None:
                # No candidate duration is risk-adjusted-EV positive right now —
                # same "nothing to do this cycle" treatment as an invalid state or
                # a no-edge probability estimate, not an error.
                self._latest_evaluations[symbol] = SymbolEvaluation(
                    quality_score=0.0, epoch=candle.epoch, approved=False,
                )
                result["rankings"] = dict(self._latest_evaluations)
                return result
            contract = duration_result.contract
            ev = duration_result.chosen.ev_estimate
            probability_for_scoring = duration_result.chosen.probability_estimate
        else:
            contract = self._contract
            ev = self._ev_engine.evaluate(probability, contract)
            probability_for_scoring = probability

        # Re-simulate MC evidence against the direction (and, when duration
        # selection is active, the ACTUAL chosen horizon) the pipeline settled
        # on — this second, direction-aware result is what the scorer's
        # mc_confidence_component should reflect (see the method's docstring).
        mc_result = self._compute_mc_result_if_configured(
            symbol, state, ev_direction_hint=ev.direction, horizon_ticks=contract.duration_ticks,
        )
        risk = self._risk_engine.assess(ev)
        opportunity = self._scorer.evaluate(ev, risk, regime, probability_for_scoring, mc_result)

        self._latest_evaluations[symbol] = SymbolEvaluation(
            quality_score=opportunity.quality_score, epoch=candle.epoch, approved=opportunity.approved,
            veto_reasons=opportunity.veto_reasons,
        )
        result["rankings"] = dict(self._latest_evaluations)

        if opportunity.approved:
            better_symbol = self._better_available_competitor(symbol, opportunity.quality_score)
            if better_symbol is not None:
                opportunity = replace(
                    opportunity,
                    approved=False,
                    veto_reasons=opportunity.veto_reasons + [
                        f"Deferred: {better_symbol} currently ranks higher "
                        f"({self._latest_evaluations[better_symbol].quality_score:.3f} vs "
                        f"{opportunity.quality_score:.3f}) and has no pending trade."
                    ],
                )

        decision = await self._execution_engine.execute(opportunity, ev, risk, contract)
        result["decision"] = decision

        if decision.action == "buy":
            result["signal_agreement"] = {
                "symbol": symbol,
                "direction": ev.direction,
                "readings": [
                    {
                        "name": r.name,
                        "kind": r.kind,
                        "detail": r.detail,
                        "confidence": r.confidence,
                        # Only probability-model rows have a directional prediction to
                        # compare against the trade — regime rows report "n/a" (None),
                        # never a False, since "no direction to agree/disagree with" is
                        # not the same claim as "this regime disagreed with the trade".
                        "agrees": (
                            (r.expected_direction == ev.direction) if r.kind == "probability" else None
                        ),
                    }
                    for r in signal_readings
                ],
            }
            self._pending_trades[symbol] = opportunity_to_pending_trade(
                opportunity,
                entry_close=candle.close,
                direction=ev.direction,
                stake=decision.stake,
                payout=decision.payout,
                probability_used=ev.probability_used,
                contract_id=decision.contract_id,
            )

        return result

    def _estimate_gbm_params(self, symbol: str) -> tuple[float, float]:
        """Shared by `_compute_mc_result_if_configured` and duration
        selection — (NAN, NAN) if there isn't yet enough recent-close
        history buffered for this symbol to estimate mu/sigma."""
        closes = self._recent_closes.get(symbol)
        if closes is None or len(closes) < 2:
            return NAN, NAN
        log_returns = np.diff(np.log(np.asarray(closes)))
        return estimate_gbm_parameters_from_returns(log_returns)

    def _compute_mc_result_if_configured(
        self, symbol: str, state: MarketState, ev_direction_hint: int | None, horizon_ticks: int | None = None,
    ):
        """
        Returns a `PricePathSimulationResult` if Monte Carlo evidence is
        configured for this run, else None. `ev_direction_hint=None` (used
        BEFORE the EV engine has settled on a direction, e.g. to build
        fusion input) simulates against `direction=1` — since GBM
        simulation is direction-symmetric, `prob_favorable` under
        `direction=1` is exactly the absolute P(price ends higher),
        which is exactly what `monte_carlo_result_to_probability_estimate`
        needs to produce a direction-agnostic `prob_up` for the fusion
        engine. Passing the EV engine's actual `ev.direction` afterward
        re-simulates against THAT specific candidate direction, which is
        what the opportunity scorer's `mc_confidence_component` should
        reflect (see `on_candle`).

        `horizon_ticks` defaults to `self._contract.duration_ticks` (the
        static reference contract) — when duration selection is active,
        callers pass the ACTUALLY CHOSEN duration instead, so the scorer's
        MC-confidence component reflects the contract really being traded,
        not the static fallback.
        """
        if self._mc_simulator is None:
            return None
        mu, sigma = self._estimate_gbm_params(symbol)
        if mu != mu:  # NaN — insufficient history
            return None

        closes = self._recent_closes[symbol]
        direction = ev_direction_hint if ev_direction_hint is not None else 1
        if direction == 0:
            return None
        return self._mc_simulator.simulate(
            symbol=symbol, epoch=state.epoch, current_price=closes[-1],
            mu_per_tick=mu, sigma_per_tick=sigma, direction=direction,
            horizon_ticks=horizon_ticks or self._contract.duration_ticks,
        )

    def _predict_probability(
        self, symbol: str, state: MarketState, regime, mc_result,
    ) -> tuple[ProbabilityEstimate, list[SignalReading]]:
        """
        Returns `(estimate, readings)`. `estimate` is the
        `BayesianLogisticRegression`-only prediction when `fusion_config`
        was never passed to `__init__` (the default, fully
        backward-compatible path) — otherwise fuses whichever of
        {Bayesian Logistic, Bagged GBM, Monte Carlo GBM} are configured,
        using this regime's weights from `WeightLearner`. See module
        docstring's "Optional Ensemble Fusion" section.

        `readings` is every individual model's own (pre-fusion) prediction
        as a `SignalReading` — kept separate from the fused result so a
        caller can show which underlying signals actually agreed with the
        direction the fused estimate (and any downstream trade) settled
        on, rather than only ever seeing the single blended number.
        """
        bayesian_estimate = self._probability_models[symbol].predict(state)
        readings = [self._reading_from_estimate("bayesian_logistic", bayesian_estimate)]
        if self._fusion_engine is None:
            return bayesian_estimate, readings

        members: dict[str, ProbabilityEstimate] = {"bayesian_logistic": bayesian_estimate}
        if symbol in self._bagged_gbm_models:
            gbm_estimate = self._bagged_gbm_models[symbol].predict(state)
            members["bagged_gbm"] = gbm_estimate
            readings.append(self._reading_from_estimate("bagged_gbm", gbm_estimate))
        if mc_result is not None and mc_result.is_valid:
            mc_estimate = monte_carlo_result_to_probability_estimate(mc_result)
            members["monte_carlo_gbm"] = mc_estimate
            readings.append(self._reading_from_estimate("monte_carlo_gbm", mc_estimate))
        # LSTM/Transformer sequence models — only present once the continuous-learning
        # cron has actually trained and persisted a champion for that model_type (see
        # main.py's load_sequence_model_challengers_if_available); absent otherwise,
        # same safe "just fuse what's available" fallback as bagged_gbm/monte_carlo_gbm
        # above. WeightLearner already carries weight slots for these two names (they've
        # been in meta_learning.model_names since before this wiring existed) — this is
        # what actually activates them, not a config change.
        for seq_name, seq_model in (self._sequence_model_estimators or {}).items():
            if not getattr(seq_model, "is_fitted", False):
                continue
            seq_estimate = seq_model.predict(state)
            if seq_estimate.is_valid:
                members[seq_name] = seq_estimate
                readings.append(self._reading_from_estimate(seq_name, seq_estimate))

        weights = self._weight_learner.get_weights(regime.regime if regime.is_valid else None)
        symbol_sufficiency = self._sufficiency.get(symbol, {})
        if symbol_sufficiency:
            weights = apply_sufficiency_scaling(weights, symbol_sufficiency)
        fused = self._fusion_engine.fuse(symbol, state.epoch, members, weights, regime.regime if regime.is_valid else None)
        estimate = fused.to_probability_estimate() if fused.is_valid else bayesian_estimate
        return estimate, readings

    @staticmethod
    def _reading_from_estimate(name: str, estimate: ProbabilityEstimate) -> SignalReading:
        detail = {1: "up", -1: "down", 0: "no edge"}.get(estimate.expected_direction, "no edge")
        return SignalReading(
            name=name, kind="probability", detail=detail,
            confidence=estimate.confidence, expected_direction=estimate.expected_direction,
        )

    def _better_available_competitor(self, symbol: str, quality_score: float) -> str | None:
        """
        Returns the symbol name of the best-ranked OTHER symbol that (a)
        currently has no pending trade (i.e. is actually available to
        compete for this decision) and (b) has a strictly higher
        quality_score than `quality_score` from its own most recent
        evaluation — or None if `symbol` is itself the current argmax
        among available competitors. See module docstring for why this is
        an approximate (not perfectly simultaneous) cross-symbol argmax.
        """
        best_symbol: str | None = None
        best_score = quality_score
        for other_symbol, evaluation in self._latest_evaluations.items():
            if other_symbol == symbol:
                continue
            if self._pending_trades.get(other_symbol) is not None:
                continue  # not actually available to trade right now
            if evaluation.quality_score > best_score:
                best_symbol = other_symbol
                best_score = evaluation.quality_score
        return best_symbol

    def _settle_pending_trade_if_any(self, symbol: str, candle: Candle) -> CompletedTrade | None:
        """Paper-mode settlement: fictional next-candle-close direction.
        Never called for a trade with a real contract_id — see on_candle."""
        pending = self._pending_trades.get(symbol)
        if pending is None:
            return None

        actual_direction = 1 if candle.close > pending.entry_close else -1
        won = actual_direction == pending.direction
        pnl = (pending.payout - pending.stake) if won else -pending.stake

        return self._finalize_settled_trade(
            symbol, pending, exit_epoch=candle.epoch, pnl=pnl, exit_reason="settled_next_candle_close",
        )

    async def _poll_live_settlement_if_any(
        self, symbol: str, pending: PendingTrade, candle: Candle
    ) -> CompletedTrade | None:
        """
        Live-mode settlement: ask Deriv, via `ContractOutcomeTracker`, what
        actually happened to `pending.contract_id` — never fabricate an
        outcome the way paper mode does. If the contract hasn't settled
        yet (or the poll itself fails), the trade simply stays pending and
        this returns None; `on_candle` will try again on the next candle.
        """
        assert self._outcome_tracker is not None  # guaranteed whenever a pending trade has a contract_id

        try:
            outcome = await self._outcome_tracker.poll(pending.contract_id)
        except Exception as exc:  # noqa: BLE001 — broker/network failures are logged, not raised
            logger.warning(
                "%s: contract status poll failed for contract_id=%s: %s — will retry next candle.",
                symbol, pending.contract_id, exc,
            )
            return None

        if not outcome.is_sold:
            return None

        return self._finalize_settled_trade(
            symbol, pending, exit_epoch=candle.epoch, pnl=outcome.pnl, exit_reason="broker_settled",
        )

    def _finalize_settled_trade(
        self, symbol: str, pending: PendingTrade, exit_epoch: int, pnl: float, exit_reason: str,
    ) -> CompletedTrade:
        """
        Shared by both settlement paths: feed the real pnl (whichever
        source it came from) into the Risk Engine's equity/circuit-breaker
        tracking and the Post-Trade Analyzer, then clear the pending slot
        for this symbol so a new trade can be opened.
        """
        new_equity = self._risk_engine.equity + pnl
        self._risk_engine.record_trade_result(
            TradeOutcome(epoch=exit_epoch, pnl=pnl, equity_after=new_equity)
        )
        trade = CompletedTrade(
            symbol=symbol,
            entry_epoch=pending.entry_epoch,
            exit_epoch=exit_epoch,
            direction=pending.direction,
            stake=pending.stake,
            pnl=pnl,
            predicted_probability=pending.predicted_probability,
            regime_at_entry=pending.regime_at_entry,
            quality_score_at_entry=pending.quality_score_at_entry,
            exit_reason=exit_reason,
        )
        self._post_trade_analyzer.record_trade(trade)
        self._pending_trades[symbol] = None
        return trade

    # ------------------------------------------------------------------ #
    # Monitoring accessors
    # ------------------------------------------------------------------ #

    def update_regime_detector(self, new_detector) -> None:
        """
        Swaps the live regime detector — called after a successful
        Champion-Challenger promotion (see `regime/promotion.py`). Safe to
        call at any point: `self._regime_detector` is only ever read via
        `.classify(state)` inside `on_candle`, never held onto across
        calls, so there's no stale-reference risk from swapping between
        candles. `new_detector` just needs to implement `.classify(state)
        -> RegimeClassification` — both `RuleBasedRegimeDetector` and
        `GaussianHMMRegimeDetector` do, so either can be passed here.

        Kept for anyone still using the classic promote-and-swap flow.
        Most new setups should prefer `set_challenger_regime_detector`
        below instead — running both detectors side by side rather than
        picking one winner to replace the other.
        """
        old_name = type(self._regime_detector).__name__
        self._regime_detector = new_detector
        logger.info(
            "Regime detector swapped: %s -> %s (Champion-Challenger promotion).",
            old_name, type(new_detector).__name__,
        )

    def set_challenger_regime_detector(self, detector: GaussianHMMRegimeDetector | None) -> None:
        """
        Registers (or clears, if `detector` is None) the HMM as the
        OPERATIVE regime source — the one actually fed into probability
        estimation, duration selection, and scoring — while rule-based's
        own classification is still computed and reported in every
        result["regime_consensus"], purely for observability. Rule-based
        is only ever the operative regime when no HMM challenger is
        registered (e.g. before the continuous-learning cron's first
        successful run) — see `on_candle`'s regime-consensus block for
        the full reasoning behind demoting rule-based from its former
        equal-veto role.

        Safe to call at any point, same as `update_regime_detector` — read
        fresh via `.classify(state)` every `on_candle` call, never cached
        across candles.
        """
        self._challenger_regime_detector = detector
        logger.info(
            "HMM regime detector (operative once set) %s.",
            "cleared — rule-based is now the operative fallback" if detector is None
            else f"set: {type(detector).__name__}",
        )

    @property
    def equity(self) -> float:
        return self._risk_engine.equity

    @property
    def starting_equity(self) -> float:
        return self._paper_config.starting_equity

    @property
    def total_pnl(self) -> float:
        """Cumulative PnL across every settled trade so far — equivalent to
        `equity - starting_equity` (no fees/interest modeled anywhere in
        this pipeline, so the two are always exactly equal); exposed
        directly so callers don't need to know that equivalence holds."""
        return self._risk_engine.equity - self._paper_config.starting_equity

    def current_rankings(self) -> dict[str, SymbolEvaluation]:
        """Every symbol's most recently computed opportunity evaluation —
        for logging the full ranked table each cycle (not just whichever
        symbol traded), so the cross-symbol comparison stays visible
        rather than needing to be reconstructed after the fact."""
        return dict(self._latest_evaluations)

    def metrics(self, window: int | None = None) -> PerformanceMetrics:
        return self._post_trade_analyzer.compute_metrics(window=window)

    @property
    def n_trades_recorded(self) -> int:
        return self._post_trade_analyzer.n_trades_recorded
