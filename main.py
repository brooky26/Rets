"""
Entry point for the platform.

Usage:
    python main.py --config configs/default.yaml

Pipeline: Deriv ticks -> integrity validation -> storage (raw ticks) ->
candle aggregation -> feature engineering -> state encoding -> regime
classification -> [if paper_trading.enabled] Probability (optionally
fused: Bayesian Logistic + Bagged GBM + Monte Carlo GBM) -> [if
paper_trading.use_duration_selection] per-candle duration selection ->
EV -> Risk -> Opportunity Scoring -> cross-symbol ranking -> Execution
(paper or live, per execution.mode) -> settlement -> Post-Trade.

STARTUP SEQUENCE, before live streaming begins:
  1. For every configured symbol, paginated historical-candle fetch (see
     DerivWebSocketClient.fetch_bootstrap_history /
     _fetch_paginated_candles_for_symbol) walks back across multiple
     requests up to whichever per-model-family target is largest
     (Bagged GBM's target when fusion is enabled, else Bayesian
     Logistic's own min_bootstrap_candles) or historical_data.lookback_days,
     whichever comes first — NOT a single capped request.
  2. Each symbol's historical candles are replayed through the SAME
     feature-pipeline/state-encoder instances that continue into live
     streaming afterward (so rolling-window warm-up carries over
     continuously, never resets), producing (states, closes) used to
     bootstrap-fit that symbol's probability model(s). Symbols with too
     little historical data are skipped (logged) rather than silently
     left broken — this is the intended behavior when Deriv's own
     history for a symbol is genuinely shorter than the target, not a
     bug to "fix" by lowering the bar silently.
  3. If regime_detection.enable_hmm_promotion is True: the SAME
     bootstrap (states, closes) are reused (no second fetch) to fit a
     Gaussian HMM per symbol, registered as a standing regime-consensus
     CHALLENGER via PaperTradingOrchestrator.set_challenger_regime_detector
     — NOT a promote-and-swap decision anymore. From here on, every
     candle classifies regime with BOTH rule-based (primary, always
     active) and the HMM; if they disagree and
     regime_detection.enable_regime_consensus_gate is True, that candle
     is treated as nothing-to-trade (logged, not an error). Fitting
     failure (e.g. insufficient bootstrap history for a symbol) leaves
     no challenger registered for that symbol — rule-based operates
     alone, the same safe fallback as before. The classic one-shot
     statistical promotion/swap flow this replaced still exists in
     regime/promotion.py for anyone who wants it directly; it's just not
     what this flag wires up by default anymore. Continuous learning of
     the HMM itself (periodic refit as more data becomes available,
     rather than only ever fitting once at each process startup) is
     handled by continuous_learning/orchestrator.py's daily cron cycle —
     see its module docstring.
  4. Live tick streaming begins, with every model family that was
     successfully bootstrapped already active and voting — nothing is
     sidelined for having less data than another model; see
     meta_learning/sufficiency.py for how an undertrained model's
     fusion weight is shrunk (not zeroed) instead.

Scope/simplifications of this pipeline (v1) are documented in
configs/paper_trading_schema.py's module docstring — most importantly:
in PAPER mode, settlement is next-candle-close direction (not real
duration_ticks expiry); in LIVE mode, real settlement is polled from
Deriv via ContractOutcomeTracker instead (see
paper_trading/orchestrator.py's module docstring). Level 7 (RL Trade
Management) is not wired in — every trade is held to its own
settlement rather than managed with HOLD/SELL decisions.

Set paper_trading.enabled: false to fall back to the original
data-collection + regime-logging-only mode. Set
paper_trading.use_ensemble_fusion: true to combine Bayesian Logistic +
Bagged GBM + Monte Carlo GBM evidence via EnsembleFusionEngine instead
of Bayesian Logistic alone. Set paper_trading.use_duration_selection:
true to pick contract duration per-candle instead of a fixed constant.
Set continuous_learning.enabled: true to run the daily
retrain/promote/re-optimize cycle in-process via APScheduler against a
rolling live MarketState buffer for the first configured symbol.

LIVE TRADING (real Deriv orders, real money): requires BOTH
execution.mode: live AND environment: live to agree (ExecutionEngine
enforces this as an independent second safety rail — a single flag
flip cannot enable live trading by accident), PLUS a real
DERIV_API_TOKEN supplied via environment variables (never commit real
credentials to YAML). DERIV_ACCOUNT_ID is optional — if unset, `run()`
below auto-resolves it from the token via
`data.deriv_client.ensure_account_id` (GET
/trading/v1/options/accounts, matching DERIV_ACCOUNT_TYPE); set it
explicitly only if you have multiple accounts of the same type. Neither
execution.mode nor environment is set to "live" by default in
configs/default.yaml — flipping both is a deliberate, separate action
the operator takes when actually ready, not something this file does
on your behalf.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

from configs.loader import load_config
from continuous_learning.orchestrator import ContinuousLearningOrchestrator
from data.candle_aggregator import CandleAggregator
from data.deriv_client import DerivWebSocketClient, ensure_account_id
from data.integrity import IntegrityValidator
from data.storage import SQLiteTickStore, SupabaseTickStore, TickStore
from data.types import Candle, ConnectionEvent, Tick
from expected_value.types import ContractSpec, ContractType
from features.pipeline import FeatureEngineeringPipeline
from features.types import FeatureVector
from model_registry.registry import ModelRegistry
from model_registry.store import InMemoryModelRegistryStore
from paper_trading.orchestrator import PaperTradingOrchestrator
from regime.hmm_detector import GaussianHMMRegimeDetector
from regime.rule_based import RuleBasedRegimeDetector
from regime.types import RegimeClassification
from state_encoder.encoder import MarketStateEncoder
from state_encoder.types import MarketState

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("main")


def build_tick_store(storage_cfg) -> TickStore:
    """Backend is chosen by configs.market_data.storage.backend (sqlite or
    supabase — see StorageConfig). sqlite is the local-dev default; supabase
    is required for Railway, since a local sqlite file does not survive a
    redeploy/restart on Railway's ephemeral filesystem."""
    if storage_cfg.backend == "sqlite":
        return SQLiteTickStore(
            db_path=storage_cfg.sqlite_path,
            write_batch_size=storage_cfg.write_batch_size,
            flush_interval_seconds=storage_cfg.flush_interval_seconds,
        )
    elif storage_cfg.backend == "supabase":
        return SupabaseTickStore(
            supabase_url=storage_cfg.supabase_url,
            supabase_key=storage_cfg.supabase_key,
            table=storage_cfg.supabase_table,
            write_batch_size=storage_cfg.write_batch_size,
            flush_interval_seconds=storage_cfg.flush_interval_seconds,
        )
    raise ValueError(f"Unknown storage backend: {storage_cfg.backend}")


def compute_bootstrap_target_candle_count(paper_cfg) -> int:
    """
    The largest per-model-family target among whatever's actually
    configured to run — pagination fetches enough for the MOST
    data-hungry active model, and every less-hungry model simply uses
    however much of that same fetched history it needs (see
    orchestrator.bootstrap()). Currently: Bagged GBM's target when
    ensemble fusion is enabled, else Bayesian Logistic's own
    min_bootstrap_candles (which IS its own target — see
    PaperTradingConfig.bagged_gbm_target_samples' docstring).
    """
    target = paper_cfg.min_bootstrap_candles
    if paper_cfg.use_ensemble_fusion:
        target = max(target, paper_cfg.bagged_gbm_target_samples)
    return target


async def bootstrap_paper_trading(
    client: DerivWebSocketClient,
    orchestrator: PaperTradingOrchestrator,
    feature_pipeline: FeatureEngineeringPipeline,
    state_encoder: MarketStateEncoder,
    symbols: list[str],
    granularity_seconds: int,
    target_candle_count: int,
) -> dict[str, tuple[list[MarketState], list[float]]]:
    """
    Returns `{symbol: (states, closes)}` for every symbol that had ANY
    historical candles at all (even if too few to clear
    min_bootstrap_candles) — callers doing regime-detector promotion
    reuse this directly rather than re-fetching.
    """
    logger.info(
        "Bootstrapping probability models from historical candles for: %s (target=%d candles/symbol, "
        "paginated up to historical_data.lookback_days).", symbols, target_candle_count,
    )
    historical = await client.fetch_bootstrap_history(symbols, granularity_seconds, target_candle_count)

    replayed: dict[str, tuple[list[MarketState], list[float]]] = {}
    for symbol, candles in historical.items():
        states: list[MarketState] = []
        closes: list[float] = []
        for candle in candles:
            vector = feature_pipeline.on_candle(candle)
            if vector is not None:
                state = state_encoder.encode(vector)
                if state.is_valid:
                    states.append(state)
                    closes.append(candle.close)
        replayed[symbol] = (states, closes)

        fitted = orchestrator.bootstrap(symbol, states, closes)
        if not fitted:
            logger.warning(
                "%s: skipped for live trading (insufficient bootstrap history); "
                "data collection continues, will retry once enough live candles accumulate.",
                symbol,
            )

    return replayed


def fit_hmm_challenger_if_enabled(
    config, orchestrator: PaperTradingOrchestrator,
    bootstrap_data: dict[str, tuple[list[MarketState], list[float]]],
) -> None:
    """
    Runs once at startup. If `continuous_learning/orchestrator.py`'s daily
    cron (Option C) has persisted a refit HMM (`CL_HMM_PATH`, or
    `<CL_DATA_DIR>/hmm_regime_detector.pkl`), that's loaded and registered
    as the regime-consensus challenger directly — this is how the HMM
    actually keeps learning over time (a fresh daily fit against whatever
    history is available that day), rather than only ever being fit once
    per process restart against a single bootstrap snapshot. Only when no
    persisted HMM is found does this fall back to fitting fresh here,
    reusing the SAME (states, closes) already fetched for
    probability-model bootstrap (no second fetch) — a reasonable
    first-run/no-cron fallback, but note it never improves across
    restarts on its own the way the cron-fed path does.

    Either way, the result is registered via
    `PaperTradingOrchestrator.set_challenger_regime_detector` — NOT a
    promote-and-swap decision. Both rule-based and the HMM run every
    candle from here on; see `on_candle`'s regime-consensus block.

    Same shared-detector scope limit as the classic promotion flow this
    replaced, for the fresh-fit fallback path: `orchestrator` has ONE
    regime detector (and now one challenger) shared across ALL symbols
    (see `run()` below), so only the FIRST symbol with enough bootstrap
    history to fit an HMM actually gets used there — per-symbol
    challengers would be a larger change. Fitting failure (insufficient
    data, or every symbol lacking enough) leaves no challenger registered
    at all — rule-based operates alone, same safe fallback as before.
    """
    regime_cfg = config.regime_detection
    if not regime_cfg.enable_hmm_promotion:
        return

    # Prefer a persisted HMM from the continuous-learning cron (Option C) over
    # fitting fresh here: the cron refits daily against whatever fresh history
    # is available at that moment (see continuous_learning/orchestrator.py's
    # _run_once_async), so its HMM has seen strictly more/more-recent data
    # than this process's own one-off bootstrap fetch ever could — a fresh
    # fit here is only a fallback for symbols/setups without that cron
    # service running at all. Same env var the cron writes to by default;
    # CL_HMM_PATH overrides it directly if you've customized either side.
    persisted_path = os.environ.get("CL_HMM_PATH") or (
        os.path.join(os.environ["CL_DATA_DIR"], "hmm_regime_detector.pkl")
        if os.environ.get("CL_DATA_DIR") else None
    )
    if persisted_path and os.path.exists(persisted_path):
        try:
            import pickle
            with open(persisted_path, "rb") as f:
                hmm_detector = pickle.load(f)
            orchestrator.set_challenger_regime_detector(hmm_detector)
            logger.info(
                "Loaded HMM regime-consensus challenger from %s (the continuous-learning "
                "cron's last refit) — skipping a fresh fit from this process's own bootstrap.",
                persisted_path,
            )
            return
        except Exception as exc:  # noqa: BLE001 — any load failure just falls through to fitting fresh
            logger.warning(
                "Failed to load persisted HMM from %s (%s) — falling back to fitting fresh "
                "from this process's own bootstrap.", persisted_path, exc,
            )

    for symbol, (states, closes) in bootstrap_data.items():
        valid_states = [s for s in states if s.is_valid]
        if len(valid_states) < regime_cfg.hmm.n_states * 2:
            logger.info("%s: too little bootstrap history to fit an HMM challenger — skipping.", symbol)
            continue
        try:
            hmm_detector = GaussianHMMRegimeDetector(regime_cfg.hmm).fit(valid_states)
        except ValueError as exc:
            logger.info("%s: HMM challenger fit failed (%s) — trying the next symbol.", symbol, exc)
            continue

        orchestrator.set_challenger_regime_detector(hmm_detector)
        logger.info(
            "%s: HMM challenger fitted on %d states and registered for regime consensus "
            "(gate %s).", symbol, len(valid_states),
            "enabled" if regime_cfg.enable_regime_consensus_gate else "disabled (observation-only)",
        )
        break  # only the first successful fit actually gets applied, given the shared-detector scope note above


async def run(config_path: str) -> None:
    config = load_config(config_path)
    config = config.model_copy(
        update={"market_data": config.market_data.model_copy(
            update={"connection": await ensure_account_id(config.market_data.connection)}
        )}
    )
    md_cfg = config.market_data
    feature_cfg = config.feature_engineering
    paper_cfg = config.paper_trading

    store = build_tick_store(md_cfg.storage)
    await store.start()

    validator = IntegrityValidator(md_cfg.integrity)
    aggregator = CandleAggregator(
        granularity_seconds=md_cfg.historical.candle_granularity_seconds
    )
    feature_pipeline = FeatureEngineeringPipeline(feature_cfg)
    state_encoder = MarketStateEncoder(config.state_encoder)
    # Rule-based detector is the active default: it needs no training data
    # and is available from the first valid MarketState. The Gaussian HMM
    # (regime/hmm_detector.py) is a challenger — if
    # regime_detection.enable_hmm_promotion is True, run() below compares
    # it against rule-based via regime/promotion.py once at startup and
    # swaps it in (PaperTradingOrchestrator.update_regime_detector) only on
    # a statistically significant improvement; otherwise this stays active.
    regime_detector = RuleBasedRegimeDetector(config.regime_detection.rule_based)

    async def on_connection_event(event: ConnectionEvent) -> None:
        logger.info("Connection event: %s (%s)", event.event, event.detail)

    orchestrator: PaperTradingOrchestrator | None = None

    # See configs/continuous_learning_schema.py's `enabled` docstring — single-symbol wiring
    # scope limitation of main.py, not of ContinuousLearningOrchestrator itself.
    cl_cfg = config.continuous_learning
    cl_orchestrator: ContinuousLearningOrchestrator | None = None
    cl_scheduler = None
    cl_symbol = md_cfg.connection.symbols[0] if cl_cfg.enabled and md_cfg.connection.symbols else None
    cl_states: list[MarketState] = []
    cl_closes: list[float] = []
    cl_buffer_cap = 5000  # bounded rolling window so memory doesn't grow unbounded over a long-running process

    if cl_cfg.enabled and cl_symbol is not None:
        cl_registry = ModelRegistry(InMemoryModelRegistryStore())
        cl_contract = ContractSpec(
            contract_type=ContractType.RISE_FALL, stake=paper_cfg.stake,
            payout=paper_cfg.stake * paper_cfg.assumed_payout_ratio, duration_ticks=paper_cfg.duration_ticks,
        )
        cl_orchestrator = ContinuousLearningOrchestrator(cl_cfg, config, cl_registry, cl_contract)
        logger.info("Continuous learning enabled for symbol %s.", cl_symbol)

    def on_regime(classification: RegimeClassification) -> None:
        status = "valid" if classification.is_valid else "invalid (NaN state)"
        logger.info(
            "Regime [%s @ %d] %s — %s (confidence=%.2f, detector=%s)",
            classification.symbol,
            classification.epoch,
            status,
            classification.regime.value,
            classification.confidence,
            classification.detector_name,
        )

    async def on_candle(candle: Candle) -> None:
        vector = feature_pipeline.on_candle(candle)
        if vector is not None:
            state = state_encoder.encode(vector)
            on_regime(regime_detector.classify(state))

            if cl_symbol is not None and candle.symbol == cl_symbol and state.is_valid:
                cl_states.append(state)
                cl_closes.append(candle.close)
                if len(cl_states) > cl_buffer_cap:
                    del cl_states[: len(cl_states) - cl_buffer_cap]
                    del cl_closes[: len(cl_closes) - cl_buffer_cap]

        if orchestrator is None:
            return

        result = await orchestrator.on_candle(candle.symbol, candle, vector)
        settled = result["settled"]
        if settled is not None:
            logger.info(
                "SETTLED %s: %s pnl=%.2f (equity=%.2f, total_trades=%d)",
                settled.symbol,
                "WIN" if settled.was_win else "LOSS",
                settled.pnl,
                orchestrator.equity,
                orchestrator.n_trades_recorded,
            )
        decision = result["decision"]
        if decision is not None and decision.action != "skip":
            logger.info("DECISION %s: %s (%s)", decision.symbol, decision.action, decision.reason)

        rankings = result["rankings"]
        if rankings is not None:
            ranked = sorted(rankings.items(), key=lambda kv: kv[1].quality_score, reverse=True)
            table = ", ".join(
                f"{sym}={ev.quality_score:.3f}{'*' if ev.approved else ''}" for sym, ev in ranked
            )
            logger.info("RANKINGS (this cycle, * = approved on own merits): %s", table)

            # RANKINGS only shows quality_score vs the adaptive threshold — it can't
            # distinguish "below threshold" from an unconditional EV/Risk gate veto,
            # which blocks a trade regardless of how high quality_score is (see
            # opportunity/scorer.py, opportunity/types.py's TradeOpportunity
            # docstring). Without this, a symbol can sit above threshold every
            # cycle and still never get approved, with literally nothing in the
            # logs explaining why. Log the top candidate's veto reasons whenever
            # it wasn't approved, so that's answerable from logs alone.
            top_symbol, top_evaluation = ranked[0]
            if not top_evaluation.approved and top_evaluation.veto_reasons:
                logger.info(
                    "TOP CANDIDATE %s not approved despite highest quality_score=%.3f: %s",
                    top_symbol, top_evaluation.quality_score, "; ".join(top_evaluation.veto_reasons),
                )

        # Same explainability gap as above, different cause: a regime-consensus-gate
        # skip (see paper_trading/orchestrator.py's on_candle) bails out BEFORE
        # scoring ever runs, so that symbol's SymbolEvaluation has no veto_reasons
        # to log above — without this, it just looks like a symbol silently never
        # scored anything, with nothing in the logs pointing at the actual cause.
        consensus = result["regime_consensus"]
        if consensus is not None and not consensus["agree"]:
            logger.info(
                "%s: regime consensus disagreement — rule_based=%s, hmm=%s.",
                candle.symbol, consensus["rule_based"], consensus["hmm"],
            )

    async def on_tick(tick: Tick) -> None:
        await store.write_ticks([tick])
        completed_candle = aggregator.on_tick(tick)
        if completed_candle is not None:
            await on_candle(completed_candle)

    client = DerivWebSocketClient(
        connection_config=md_cfg.connection,
        historical_config=md_cfg.historical,
        integrity_validator=validator,
        on_tick=on_tick,
        on_connection_event=on_connection_event,
    )

    if paper_cfg.enabled:
        orchestrator = PaperTradingOrchestrator(
            paper_config=paper_cfg,
            probability_config=config.probability_estimation.bayesian_logistic,
            ev_config=config.expected_value,
            risk_config=config.risk,
            opportunity_config=config.opportunity_scoring,
            post_trade_config=config.post_trade_analysis,
            execution_config=config.execution,
            platform_environment=config.environment,
            regime_detector=regime_detector,
            state_encoder=state_encoder,
            broker_client=client,
            # See configs/paper_trading_schema.py's `use_ensemble_fusion` docstring — False
            # by default, which keeps every kwarg below None and reproduces the original
            # single-Bayesian-Logistic-model behavior exactly.
            fusion_config=config.ensemble_fusion if paper_cfg.use_ensemble_fusion else None,
            meta_learning_config=config.meta_learning if paper_cfg.use_ensemble_fusion else None,
            bagged_gbm_config=config.probability_estimation.bagged_gbm if paper_cfg.use_ensemble_fusion else None,
            # Needed for EITHER ensemble fusion's monte_carlo_gbm member OR duration
            # selection's primary (MC) signal — active if either is enabled, not just fusion.
            monte_carlo_config=(
                config.monte_carlo_price_paths
                if (paper_cfg.use_ensemble_fusion or paper_cfg.use_duration_selection) else None
            ),
            duration_selection_config=config.duration_selection if paper_cfg.use_duration_selection else None,
            enable_regime_consensus_gate=config.regime_detection.enable_regime_consensus_gate,
        )

    try:
        if orchestrator is not None:
            target_candle_count = compute_bootstrap_target_candle_count(paper_cfg)
            bootstrap_data = await bootstrap_paper_trading(
                client, orchestrator, feature_pipeline, state_encoder,
                md_cfg.connection.symbols, md_cfg.historical.candle_granularity_seconds,
                target_candle_count,
            )
            fit_hmm_challenger_if_enabled(config, orchestrator, bootstrap_data)

        if cl_orchestrator is not None:
            def _cl_data_provider():
                # Snapshot the rolling buffer — see `ContinuousLearningOrchestrator
                # .start_scheduler`'s docstring for why this is an injected callable
                # rather than the orchestrator owning data collection itself.
                if not cl_states:
                    raise RuntimeError(f"No MarketState history collected yet for {cl_symbol}.")
                import numpy as _np

                return list(cl_states), _np.array(cl_closes), cl_states[-1].epoch

            cl_scheduler = cl_orchestrator.start_scheduler(_cl_data_provider)

        await client.run_forever()
    finally:
        if cl_scheduler is not None:
            cl_scheduler.shutdown()
        await client.stop()
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Deriv Trading Research Platform")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    asyncio.run(run(args.config))


if __name__ == "__main__":
    main()
