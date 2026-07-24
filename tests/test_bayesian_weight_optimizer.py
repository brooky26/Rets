import numpy as np
import pytest

from configs.ensemble_schema import BayesianWeightOptimizerConfig
from ensemble.bayesian_weight_optimizer import (
    BayesianWeightOptimizer,
    compute_sharpe_and_calmar,
    simulate_returns_for_weights,
)
from ensemble.types import WeightOptimizationRecord
from regime.types import RegimeLabel


def make_records(n: int, seed: int, good_model="a", bad_model="b", regime=None) -> list[WeightOptimizationRecord]:
    """Synthetic records where `good_model`'s probability is strongly
    predictive of the realized outcome and `bad_model`'s is pure noise
    — a well-behaved optimizer should learn to weight `good_model` higher."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        direction = 1 if rng.random() > 0.5 else -1
        won = rng.random() < 0.7  # good_model will reflect this; bad_model won't
        good_prob = 0.85 if (won and direction == 1) or (not won and direction == -1) else 0.15
        if direction == -1:
            good_prob = 1 - good_prob if won else good_prob
        # Simplify: good_model prob_up aligns with direction when won, opposes when lost.
        good_prob_up = 0.8 if (direction == 1) == won else 0.2
        bad_prob_up = float(rng.uniform(0.0, 1.0))
        realized_return_pct = 0.9 if won else -1.0

        records.append(
            WeightOptimizationRecord(
                symbol="TEST", epoch=i,
                model_probabilities={good_model: good_prob_up, bad_model: bad_prob_up},
                direction=direction, realized_return_pct=realized_return_pct, regime=regime,
            )
        )
    return records


def test_simulate_returns_skips_records_with_no_matching_model():
    records = [
        WeightOptimizationRecord(symbol="T", epoch=1, model_probabilities={"x": 0.8}, direction=1, realized_return_pct=1.0),
    ]
    returns = simulate_returns_for_weights({"a": 1.0}, records)
    assert len(returns) == 0


def test_simulate_returns_full_confidence_full_return():
    records = [
        WeightOptimizationRecord(symbol="T", epoch=1, model_probabilities={"a": 1.0}, direction=1, realized_return_pct=0.5),
    ]
    returns = simulate_returns_for_weights({"a": 1.0}, records)
    assert returns[0] == pytest.approx(0.5)


def test_simulate_returns_disagreement_zeros_return():
    records = [
        WeightOptimizationRecord(symbol="T", epoch=1, model_probabilities={"a": 0.1}, direction=1, realized_return_pct=0.5),
    ]
    returns = simulate_returns_for_weights({"a": 1.0}, records)
    assert returns[0] == pytest.approx(0.0)


def test_compute_sharpe_and_calmar_empty():
    sharpe, calmar, dd = compute_sharpe_and_calmar(np.array([]))
    assert sharpe == 0.0 and calmar == 0.0 and dd == 0.0


def test_compute_sharpe_and_calmar_no_drawdown():
    returns = np.array([0.1, 0.1, 0.1])
    sharpe, calmar, dd = compute_sharpe_and_calmar(returns)
    assert dd == 0.0
    assert calmar == 100.0


def test_optimizer_prefers_predictive_model_multi_objective():
    records = make_records(150, seed=1)
    config = BayesianWeightOptimizerConfig(n_trials=25, objective_mode="multi_objective", per_regime=False, sampler_seed=0)
    optimizer = BayesianWeightOptimizer(config, model_names=["a", "b"])
    global_weights, regime_weights = optimizer.optimize(records)

    assert global_weights.weights.keys() == {"a", "b"}
    assert abs(sum(global_weights.weights.values()) - 1.0) < 1e-6
    assert global_weights.weights["a"] > global_weights.weights["b"]
    assert regime_weights == {}


def test_optimizer_custom_objective_mode():
    records = make_records(150, seed=2)
    config = BayesianWeightOptimizerConfig(n_trials=25, objective_mode="custom", per_regime=False, sampler_seed=0)
    optimizer = BayesianWeightOptimizer(config, model_names=["a", "b"])
    global_weights, _ = optimizer.optimize(records)
    assert global_weights.weights["a"] > global_weights.weights["b"]


def test_optimizer_per_regime_produces_regime_weights_when_sufficient():
    records = make_records(50, seed=3, regime=RegimeLabel.STRONG_TREND)
    config = BayesianWeightOptimizerConfig(n_trials=15, objective_mode="multi_objective", per_regime=True, min_trades_per_regime=20, sampler_seed=0)
    optimizer = BayesianWeightOptimizer(config, model_names=["a", "b"])
    global_weights, regime_weights = optimizer.optimize(records)
    assert RegimeLabel.STRONG_TREND in regime_weights
    assert abs(sum(regime_weights[RegimeLabel.STRONG_TREND].weights.values()) - 1.0) < 1e-6


def test_optimizer_per_regime_skips_sparse_regimes():
    records = make_records(10, seed=4, regime=RegimeLabel.RANGE)
    config = BayesianWeightOptimizerConfig(n_trials=10, objective_mode="multi_objective", per_regime=True, min_trades_per_regime=30, sampler_seed=0)
    optimizer = BayesianWeightOptimizer(config, model_names=["a", "b"])
    _, regime_weights = optimizer.optimize(records)
    assert RegimeLabel.RANGE not in regime_weights


def test_optimizer_raises_on_empty_records():
    config = BayesianWeightOptimizerConfig(n_trials=5)
    optimizer = BayesianWeightOptimizer(config, model_names=["a", "b"])
    with pytest.raises(ValueError):
        optimizer.optimize([])
