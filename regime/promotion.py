"""
Regime detector promotion: Champion-Challenger comparison between the
rule-based detector (champion by default — needs no training data, safe
from day one, per main.py's own bootstrap docstring) and a Gaussian HMM
detector (challenger), fit on the same historical data used to bootstrap
the probability model.

This is architecturally distinct from `continuous_learning/orchestrator.py`'s
model comparison, which compares PROBABILITY models (Bayesian Logistic,
Bagged GBM, sequence models) against each other via `WalkForwardBacktester`
— which itself retrains the probability model on each rolling window.
Regime detection is a different axis: here, the probability model is
FIXED (fit once on the training split) and only the regime LABEL varies
between champion and challenger, because a regime label doesn't feed the
probability model directly — it feeds (a) the opportunity scorer's
per-regime adaptive threshold and (b), when ensemble fusion is active,
the fusion engine's per-regime weights. So "is this regime detector
better" can only be measured by running the same fixed downstream
pipeline twice, once per detector, and comparing realized trade PnL —
not by retraining anything, and not by any regime-classification
accuracy metric (there is no ground-truth regime label to score
against).
"""

from __future__ import annotations

import logging

import numpy as np

from champion_challenger.comparator import ChampionChallengerComparator
from champion_challenger.types import PromotionDecision
from configs.champion_challenger_schema import ChampionChallengerConfig
from configs.ev_schema import ExpectedValueConfig
from configs.opportunity_schema import OpportunityScoringConfig
from configs.probability_schema import BayesianLogisticConfig
from configs.regime_schema import GaussianHMMConfig, RuleBasedRegimeConfig
from configs.risk_schema import RiskConfig
from expected_value.engine import ExpectedValueEngine
from expected_value.types import ContractSpec
from opportunity.scorer import TradeOpportunityScorer
from probability.bayesian_logistic import BayesianLogisticRegression
from regime.hmm_detector import GaussianHMMRegimeDetector
from regime.rule_based import RuleBasedRegimeDetector
from risk.engine import RiskEngine
from risk.types import TradeOutcome
from state_encoder.types import MarketState

logger = logging.getLogger("regime.promotion")


def simulate_trades_with_regime_detector(
    states: list[MarketState],
    closes: list[float],
    model: BayesianLogisticRegression,
    ev_config: ExpectedValueConfig,
    risk_config: RiskConfig,
    opportunity_config: OpportunityScoringConfig,
    contract: ContractSpec,
    regime_detector,
    starting_equity: float,
) -> list[float]:
    """
    Single pass over `states`/`closes` (expected to be a HOLDOUT set — the
    caller is responsible for not overlapping this with whatever `model`
    and `regime_detector` were themselves fit on): for each state (except
    the last, which has no next-close to settle against), classify regime
    via `regime_detector`, predict probability via the FIXED `model`,
    evaluate EV/risk/opportunity, and — if approved — "trade" it, settling
    against the next candle's close direction (the same convention
    `paper_trading/orchestrator.py` uses for paper-mode settlement).

    Returns the list of realized per-trade PnLs — this IS the
    `champion_returns`/`challenger_returns` `ChampionChallengerComparator`
    expects, produced by holding the regime detector as the only variable
    between two calls to this function with everything else identical.
    """
    ev_engine = ExpectedValueEngine(ev_config)
    risk_engine = RiskEngine(risk_config, starting_equity=starting_equity)
    scorer = TradeOpportunityScorer(opportunity_config)

    pnls: list[float] = []
    for i in range(len(states) - 1):
        state = states[i]
        if not state.is_valid:
            continue
        regime = regime_detector.classify(state)
        probability = model.predict(state)
        if not probability.is_valid:
            continue
        ev = ev_engine.evaluate(probability, contract)
        risk = risk_engine.assess(ev)
        opportunity = scorer.evaluate(ev, risk, regime, probability)
        if not opportunity.approved:
            continue

        actual_direction = 1 if closes[i + 1] > closes[i] else -1
        won = actual_direction == ev.direction
        pnl = contract.profit_if_win if won else contract.loss_if_lose
        pnls.append(pnl)

        new_equity = risk_engine.equity + pnl
        risk_engine.record_trade_result(TradeOutcome(epoch=state.epoch, pnl=pnl, equity_after=new_equity))

    return pnls


def compare_regime_detectors(
    states: list[MarketState],
    closes: list[float],
    probability_config: BayesianLogisticConfig,
    ev_config: ExpectedValueConfig,
    risk_config: RiskConfig,
    opportunity_config: OpportunityScoringConfig,
    contract: ContractSpec,
    champion_challenger_config: ChampionChallengerConfig,
    hmm_config: GaussianHMMConfig,
    rule_based_config: RuleBasedRegimeConfig,
    starting_equity: float,
    train_fraction: float = 0.6,
) -> tuple[PromotionDecision, GaussianHMMRegimeDetector | None]:
    """
    Splits `states`/`closes` at `train_fraction`, fits a probability model
    AND the HMM on the train split, runs BOTH detectors (rule-based as
    champion, HMM as challenger) over the SAME holdout split with that one
    fixed probability model, and compares realized trade PnL via
    `ChampionChallengerComparator`'s bootstrap significance test.

    Returns `(decision, fitted_hmm)`. `fitted_hmm` is returned whenever
    fitting succeeded — even when `decision.promote` is False — so a
    caller wanting to log or re-evaluate the same fit later doesn't need
    to refit; only actually swap it into live use (via
    `PaperTradingOrchestrator.update_regime_detector`) when
    `decision.promote` is True.

    If the HMM fails to fit at all (e.g. degenerate/insufficient training
    data — genuinely possible for a newly-listed or thinly-historied
    symbol), this is logged and `decision.promote` is always False with
    `fitted_hmm=None` — rule-based remains champion by default, which is
    the safe outcome, not an error.
    """
    if len(states) != len(closes):
        raise ValueError("states and closes must be the same length")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be in (0, 1)")

    split = int(len(states) * train_fraction)
    train_states, holdout_states = states[:split], states[split:]
    train_closes, holdout_closes = closes[:split], closes[split:]

    if len(holdout_states) < 2 or len(train_states) < 2:
        raise ValueError(
            "Not enough data after the train/holdout split to fit a model and simulate "
            "at least one trade — pass more history or a smaller train_fraction."
        )

    labels = (np.diff(np.array(train_closes)) > 0).astype(int)
    X = np.array(
        [[getattr(s, dim) for dim in probability_config.feature_dims] for s in train_states[:-1]]
    )
    model = BayesianLogisticRegression(probability_config).fit(X, labels)

    hmm_detector: GaussianHMMRegimeDetector | None
    try:
        hmm_detector = GaussianHMMRegimeDetector(hmm_config).fit(train_states)
    except Exception as exc:  # noqa: BLE001 — degenerate training data is expected occasionally, not a bug
        logger.warning("HMM fit failed on this training split (%s) — rule-based stays champion.", exc)
        hmm_detector = None

    rule_based_detector = RuleBasedRegimeDetector(rule_based_config)
    champion_returns = simulate_trades_with_regime_detector(
        holdout_states, holdout_closes, model, ev_config, risk_config, opportunity_config,
        contract, rule_based_detector, starting_equity,
    )

    comparator = ChampionChallengerComparator(champion_challenger_config)

    if hmm_detector is None:
        decision = comparator.compare("rule_based_regime", "hmm_regime", champion_returns, [])
        return decision, None

    challenger_returns = simulate_trades_with_regime_detector(
        holdout_states, holdout_closes, model, ev_config, risk_config, opportunity_config,
        contract, hmm_detector, starting_equity,
    )
    decision = comparator.compare("rule_based_regime", "hmm_regime", champion_returns, challenger_returns)
    return decision, hmm_detector
