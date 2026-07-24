import numpy as np
import pytest

from configs.continuous_learning_schema import ContinuousLearningConfig
from configs.schema import DerivConnectionConfig, MarketDataLayerConfig, PlatformConfig
from continuous_learning.orchestrator import ContinuousLearningOrchestrator
from expected_value.types import ContractSpec, ContractType
from model_registry.registry import ModelRegistry
from model_registry.store import InMemoryModelRegistryStore
from state_encoder.types import MarketState


def make_state(symbol: str, epoch: int, trend: float, volatility: float) -> MarketState:
    return MarketState(
        symbol=symbol, epoch=epoch, trend=trend, momentum=trend * 0.5, acceleration=0.0,
        volatility=volatility, noise=0.05, persistence=0.0, compression_expansion=0.0,
        complexity=0.0, uncertainty=0.2, liquidity=0.0, market_phase=0.0,
    )


def make_synthetic_history(n: int, seed: int) -> tuple[list[MarketState], np.ndarray]:
    """`trend` is deliberately predictive of the next close's direction —
    a well-behaved pipeline should be able to learn from it."""
    rng = np.random.default_rng(seed)
    trends = rng.uniform(-1, 1, size=n)
    vols = np.abs(rng.normal(0.1, 0.05, size=n))
    states = [make_state("STPRNG100", i, float(trends[i]), float(vols[i])) for i in range(n)]
    up_prob = 0.5 + 0.35 * trends  # trend > 0 -> more likely up
    moves = rng.random(n) < up_prob
    closes = np.cumsum(np.where(moves, 1.0, -1.0)) + 1000.0
    return states, closes


def make_platform_config() -> PlatformConfig:
    return PlatformConfig(
        market_data=MarketDataLayerConfig(connection=DerivConnectionConfig(app_id="1234")),
    )


def make_orchestrator(model_types, min_states=100, run_weight_optimization=True) -> ContinuousLearningOrchestrator:
    cl_config = ContinuousLearningConfig(
        model_types=model_types, train_fraction=0.7, drift_reference_fraction=0.5,
        min_states_for_cycle=min_states, run_weight_optimization=run_weight_optimization,
    )
    platform_config = make_platform_config()
    registry = ModelRegistry(InMemoryModelRegistryStore())
    contract = ContractSpec(contract_type=ContractType.RISE_FALL, stake=10.0, payout=19.0, duration_ticks=5)
    return ContinuousLearningOrchestrator(cl_config, platform_config, registry, contract)


def test_cycle_skips_with_insufficient_data():
    orchestrator = make_orchestrator(["bayesian_logistic"], min_states=1000)
    states, closes = make_synthetic_history(50, seed=1)
    report = orchestrator.run_daily_cycle(states, closes, cycle_epoch=999)
    assert report.skipped
    assert "50" in report.skip_reason or "Only" in report.skip_reason


def test_cycle_rejects_mismatched_lengths():
    orchestrator = make_orchestrator(["bayesian_logistic"])
    states, closes = make_synthetic_history(50, seed=1)
    with pytest.raises(ValueError):
        orchestrator.run_daily_cycle(states, closes[:-1], cycle_epoch=1)


def test_first_cycle_auto_promotes_statistical_candidates():
    orchestrator = make_orchestrator(["bayesian_logistic", "bagged_gbm"], min_states=100, run_weight_optimization=False)
    states, closes = make_synthetic_history(300, seed=2)
    report = orchestrator.run_daily_cycle(states, closes, cycle_epoch=1)

    assert not report.skipped
    assert len(report.candidate_results) == 2
    for result in report.candidate_results:
        assert result.promoted  # no prior champion -> auto-promote
        assert result.promotion_decision is None

    registry = orchestrator._registry
    assert registry.get_champion("bayesian_logistic") is not None
    assert registry.get_champion("bagged_gbm") is not None


def test_second_cycle_runs_champion_challenger_comparison():
    orchestrator = make_orchestrator(["bayesian_logistic"], min_states=100, run_weight_optimization=False)
    states1, closes1 = make_synthetic_history(300, seed=3)
    report1 = orchestrator.run_daily_cycle(states1, closes1, cycle_epoch=1)
    assert report1.candidate_results[0].promoted

    states2, closes2 = make_synthetic_history(300, seed=4)
    report2 = orchestrator.run_daily_cycle(states2, closes2, cycle_epoch=2)
    assert not report2.skipped
    result = report2.candidate_results[0]
    # Now there IS a prior champion with a live artifact -> a real comparison should have run.
    assert result.promotion_decision is not None
    assert result.promotion_decision.champion_id is not None


def test_weight_optimization_runs_with_multiple_models():
    orchestrator = make_orchestrator(["bayesian_logistic", "bagged_gbm"], min_states=100, run_weight_optimization=True)
    orchestrator._platform_config.bayesian_weight_optimizer.n_trials = 10
    orchestrator._platform_config.bayesian_weight_optimizer.min_trades_per_regime = 5
    states, closes = make_synthetic_history(300, seed=5)
    report = orchestrator.run_daily_cycle(states, closes, cycle_epoch=1)

    assert not report.skipped
    if report.global_weights is not None:
        assert abs(sum(report.global_weights.weights.values()) - 1.0) < 1e-6


def test_summary_property_does_not_raise():
    orchestrator = make_orchestrator(["bayesian_logistic"], min_states=1000)
    states, closes = make_synthetic_history(50, seed=6)
    report = orchestrator.run_daily_cycle(states, closes, cycle_epoch=1)
    assert isinstance(report.summary, str)
    assert "SKIPPED" in report.summary
