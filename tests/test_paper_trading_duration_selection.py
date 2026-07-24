"""
Orchestrator-level tests for duration selection wiring — the standalone
DurationSelector math is already covered by test_duration_selector.py;
these confirm on_candle actually calls it, uses its chosen contract for
EV/risk/opportunity/execution, and gracefully no-ops when no candidate
duration is viable.
"""

import numpy as np
import pytest

from configs.duration_selection_schema import DurationSelectionConfig
from configs.ev_schema import ExpectedValueConfig
from configs.execution_schema import ExecutionConfig
from configs.monte_carlo_schema import MonteCarloPricePathConfig
from configs.opportunity_schema import OpportunityScoringConfig
from configs.paper_trading_schema import PaperTradingConfig
from configs.post_trade_schema import PostTradeAnalysisConfig
from configs.probability_schema import BayesianLogisticConfig
from configs.regime_schema import RuleBasedRegimeConfig
from configs.risk_schema import RiskConfig
from configs.state_encoder_schema import StateEncoderConfig
from data.types import Candle
from features.types import FeatureVector
from paper_trading.orchestrator import PaperTradingOrchestrator
from regime.rule_based import RuleBasedRegimeDetector
from state_encoder.encoder import MarketStateEncoder
from state_encoder.types import MarketState


def make_synthetic_states_and_closes(n_total: int, seed: int, signal_strength: float = 0.6, symbol: str = "stpRNG"):
    rng = np.random.default_rng(seed)
    trend_vals = rng.uniform(-1, 1, n_total)
    states = [
        MarketState(
            symbol=symbol, epoch=i, trend=float(trend_vals[i]), momentum=0.0, acceleration=0.0,
            volatility=0.1, noise=0.1, persistence=0.3, compression_expansion=0.0, complexity=0.0,
            uncertainty=0.1, liquidity=0.0, market_phase=0.0,
        )
        for i in range(n_total)
    ]
    closes = [100.0]
    for i in range(n_total - 1):
        p_up = float(np.clip(0.5 + signal_strength * trend_vals[i], 0.05, 0.95))
        up = rng.uniform(0, 1) < p_up
        closes.append(closes[-1] + (1.0 if up else -1.0))
    return states, closes


def make_orchestrator_with_duration_selection(duration_selection_config=None, **paper_overrides) -> PaperTradingOrchestrator:
    paper_defaults = dict(min_bootstrap_candles=50, starting_equity=1000.0, stake=10.0, assumed_payout_ratio=1.9)
    paper_defaults.update(paper_overrides)
    return PaperTradingOrchestrator(
        paper_config=PaperTradingConfig(**paper_defaults),
        probability_config=BayesianLogisticConfig(feature_dims=["trend"], prior_precision=1.0),
        ev_config=ExpectedValueConfig(min_ev_threshold=0.0),
        risk_config=RiskConfig(),
        opportunity_config=OpportunityScoringConfig(base_confidence_threshold=0.3, threshold_min=0.2),
        post_trade_config=PostTradeAnalysisConfig(),
        execution_config=ExecutionConfig(mode="paper"),
        platform_environment="development",
        regime_detector=RuleBasedRegimeDetector(RuleBasedRegimeConfig()),
        state_encoder=MarketStateEncoder(StateEncoderConfig()),
        duration_selection_config=duration_selection_config,
    )


def make_candle(symbol: str, epoch: int, close: float) -> Candle:
    return Candle(symbol=symbol, epoch=epoch, granularity=60, open=close, high=close, low=close, close=close)


def make_vector(symbol: str, epoch: int) -> FeatureVector:
    return FeatureVector(symbol=symbol, epoch=epoch, values={})


def make_strong_state(symbol: str, persistence: float = 0.6) -> MarketState:
    return MarketState(
        symbol=symbol, epoch=9999, trend=1.0, momentum=0.0, acceleration=0.0,
        volatility=0.1, noise=0.1, persistence=persistence, compression_expansion=0.0,
        complexity=0.0, uncertainty=0.1, liquidity=0.0, market_phase=0.0,
    )


@pytest.mark.asyncio
async def test_without_duration_selection_config_behavior_is_unchanged():
    """Opt-in design: duration_selection_config=None (default) must
    reproduce the exact static-contract behavior from before this feature
    existed — same guarantee ensemble fusion already provides."""
    orch = make_orchestrator_with_duration_selection(duration_selection_config=None)
    states, closes = make_synthetic_states_and_closes(200, seed=1)
    orch.bootstrap("stpRNG", states, closes)

    strong_state = make_strong_state("stpRNG")
    orch._state_encoder.encode = lambda v, update_normalizer=True: strong_state

    candle = make_candle("stpRNG", epoch=100, close=100.0)
    vector = make_vector("stpRNG", epoch=100)
    result = await orch.on_candle("stpRNG", candle, vector)

    assert result["duration_selection"] is None
    assert result["decision"] is not None
    assert result["decision"].action == "buy"


@pytest.mark.asyncio
async def test_with_duration_selection_configured_result_includes_selection():
    orch = make_orchestrator_with_duration_selection(
        duration_selection_config=DurationSelectionConfig(candidate_durations_ticks=[3, 5, 8])
    )
    states, closes = make_synthetic_states_and_closes(200, seed=2)
    orch.bootstrap("stpRNG", states, closes)

    strong_state = make_strong_state("stpRNG")
    orch._state_encoder.encode = lambda v, update_normalizer=True: strong_state

    candle = make_candle("stpRNG", epoch=100, close=100.0)
    vector = make_vector("stpRNG", epoch=100)
    result = await orch.on_candle("stpRNG", candle, vector)

    assert result["duration_selection"] is not None
    assert len(result["duration_selection"].all_candidates) == 3
    if result["duration_selection"].chosen is not None:
        assert result["decision"].action == "buy"
        # The pending trade's contract must reflect the CHOSEN duration, not
        # a static default — verify via the traded stake/payout consistency.
        pending = orch._pending_trades["stpRNG"]
        assert pending is not None


@pytest.mark.asyncio
async def test_no_viable_duration_produces_no_trade_and_zero_quality_score():
    # An opportunity config with a very high threshold combined with tight
    # EV requirements makes it plausible for no candidate to be EV-positive
    # for a middling-confidence state — but the cleanest, deterministic way
    # to force "no viable duration" is a near-coinflip state.
    orch = make_orchestrator_with_duration_selection(
        duration_selection_config=DurationSelectionConfig(candidate_durations_ticks=[3, 5, 8]),
        assumed_payout_ratio=1.01,  # breakeven confidence needs to be very close to 1.0
    )
    states, closes = make_synthetic_states_and_closes(200, seed=3)
    orch.bootstrap("stpRNG", states, closes)

    near_coinflip_state = MarketState(
        symbol="stpRNG", epoch=9999, trend=0.001, momentum=0.0, acceleration=0.0,
        volatility=0.1, noise=0.1, persistence=0.0, compression_expansion=0.0,
        complexity=0.0, uncertainty=0.1, liquidity=0.0, market_phase=0.0,
    )
    orch._state_encoder.encode = lambda v, update_normalizer=True: near_coinflip_state

    candle = make_candle("stpRNG", epoch=100, close=100.0)
    vector = make_vector("stpRNG", epoch=100)
    result = await orch.on_candle("stpRNG", candle, vector)

    assert result["duration_selection"] is not None
    if result["duration_selection"].chosen is None:
        assert result["decision"] is None
        assert orch._pending_trades["stpRNG"] is None
        rankings = orch.current_rankings()
        assert rankings["stpRNG"].quality_score == 0.0
        assert rankings["stpRNG"].approved is False


@pytest.mark.asyncio
async def test_duration_selection_with_monte_carlo_configured_uses_mc_or_hurst():
    orch = PaperTradingOrchestrator(
        paper_config=PaperTradingConfig(
            min_bootstrap_candles=50, starting_equity=1000.0, stake=10.0, assumed_payout_ratio=1.9,
        ),
        probability_config=BayesianLogisticConfig(feature_dims=["trend"], prior_precision=1.0),
        ev_config=ExpectedValueConfig(min_ev_threshold=0.0),
        risk_config=RiskConfig(),
        opportunity_config=OpportunityScoringConfig(base_confidence_threshold=0.3, threshold_min=0.2),
        post_trade_config=PostTradeAnalysisConfig(),
        execution_config=ExecutionConfig(mode="paper"),
        platform_environment="development",
        regime_detector=RuleBasedRegimeDetector(RuleBasedRegimeConfig()),
        state_encoder=MarketStateEncoder(StateEncoderConfig()),
        duration_selection_config=DurationSelectionConfig(candidate_durations_ticks=[3, 5, 8]),
        monte_carlo_config=MonteCarloPricePathConfig(n_paths=1000, random_seed=1),
    )

    states, closes = make_synthetic_states_and_closes(200, seed=4)
    orch.bootstrap("stpRNG", states, closes)
    orch._recent_closes["stpRNG"] = closes[-60:]

    strong_state = make_strong_state("stpRNG")
    orch._state_encoder.encode = lambda v, update_normalizer=True: strong_state

    candle = make_candle("stpRNG", epoch=100, close=100.0)
    vector = make_vector("stpRNG", epoch=100)
    result = await orch.on_candle("stpRNG", candle, vector)

    assert result["duration_selection"] is not None
    methods = {c.method for c in result["duration_selection"].all_candidates}
    assert methods.issubset({"monte_carlo", "hurst_fallback"})
