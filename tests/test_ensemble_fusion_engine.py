import pytest

from configs.ensemble_schema import EnsembleFusionConfig
from ensemble.fusion_engine import EnsembleFusionEngine, monte_carlo_result_to_probability_estimate
from meta_learning.types import EnsembleWeights
from monte_carlo.types import PricePathSimulationResult
from probability.types import ProbabilityEstimate


def make_pe(name, prob_up, uncertainty=0.2, direction=None) -> ProbabilityEstimate:
    prob_down = 1 - prob_up
    d = direction if direction is not None else (1 if prob_up > 0.5 else (-1 if prob_up < 0.5 else 0))
    return ProbabilityEstimate(
        symbol="TEST", epoch=1, model_name=name, prob_up=prob_up, prob_down=prob_down,
        uncertainty=uncertainty, expected_direction=d, confidence=max(prob_up, prob_down),
    )


def make_weights(**w) -> EnsembleWeights:
    return EnsembleWeights(weights=w, source="test")


def test_fuse_weighted_average_of_prob_up():
    engine = EnsembleFusionEngine(EnsembleFusionConfig(disagreement_penalty_weight=0.0))
    members = {"a": make_pe("a", 0.8), "b": make_pe("b", 0.4)}
    weights = make_weights(a=0.5, b=0.5)
    result = engine.fuse("TEST", 1, members, weights)
    assert result.is_valid
    assert result.prob_up == pytest.approx(0.6)
    assert result.prob_down == pytest.approx(0.4)


def test_fuse_respects_unequal_weights():
    engine = EnsembleFusionEngine(EnsembleFusionConfig(disagreement_penalty_weight=0.0))
    members = {"a": make_pe("a", 1.0, uncertainty=0.0), "b": make_pe("b", 0.0, uncertainty=0.0)}
    weights = make_weights(a=0.9, b=0.1)
    result = engine.fuse("TEST", 1, members, weights)
    assert result.prob_up == pytest.approx(0.9)


def test_fuse_drops_invalid_members_and_renormalizes():
    engine = EnsembleFusionEngine(EnsembleFusionConfig(disagreement_penalty_weight=0.0))
    invalid = ProbabilityEstimate(symbol="TEST", epoch=1, model_name="c", prob_up=float("nan"), prob_down=float("nan"), uncertainty=float("nan"), expected_direction=0, confidence=float("nan"))
    members = {"a": make_pe("a", 0.8), "c": invalid}
    weights = make_weights(a=0.5, c=0.5)
    result = engine.fuse("TEST", 1, members, weights)
    assert result.is_valid
    assert result.prob_up == pytest.approx(0.8)
    assert "c" not in result.weights_used
    assert result.weights_used["a"] == pytest.approx(1.0)


def test_fuse_disagreement_increases_uncertainty():
    engine_no_disagreement = EnsembleFusionEngine(EnsembleFusionConfig(disagreement_penalty_weight=0.0))
    engine_with_disagreement = EnsembleFusionEngine(EnsembleFusionConfig(disagreement_penalty_weight=1.0))
    # Two models that strongly disagree (0.9 vs 0.1) but each individually report low uncertainty.
    members = {"a": make_pe("a", 0.9, uncertainty=0.05), "b": make_pe("b", 0.1, uncertainty=0.05)}
    weights = make_weights(a=0.5, b=0.5)

    low = engine_no_disagreement.fuse("TEST", 1, members, weights)
    high = engine_with_disagreement.fuse("TEST", 1, members, weights)
    assert high.uncertainty > low.uncertainty


def test_fuse_below_min_members_returns_invalid():
    engine = EnsembleFusionEngine(EnsembleFusionConfig(min_members_required=2))
    members = {"a": make_pe("a", 0.8)}
    weights = make_weights(a=1.0)
    result = engine.fuse("TEST", 1, members, weights)
    assert not result.is_valid


def test_to_probability_estimate_adapter():
    engine = EnsembleFusionEngine(EnsembleFusionConfig())
    members = {"a": make_pe("a", 0.7)}
    weights = make_weights(a=1.0)
    fused = engine.fuse("TEST", 1, members, weights)
    pe = fused.to_probability_estimate()
    assert pe.prob_up == pytest.approx(fused.prob_up)
    assert pe.model_name == "ensemble_fusion"


def test_monte_carlo_adapter_up_direction():
    mc = PricePathSimulationResult(
        symbol="TEST", epoch=1, direction=1, n_paths=1000, horizon_ticks=10, current_price=100.0,
        mu_per_tick=0.001, sigma_per_tick=0.01, prob_favorable=0.65, expected_favorable_duration_ticks=3.0,
        terminal_return_mean=0.01, terminal_return_std=0.03, mfe_mean=0.02, mfe_p95=0.05, mae_mean=-0.01, mae_p95=0.02,
    )
    pe = monte_carlo_result_to_probability_estimate(mc)
    assert pe.is_valid
    assert pe.prob_up == pytest.approx(0.65)
    assert pe.expected_direction == 1


def test_monte_carlo_adapter_down_direction_flips_prob_up():
    mc = PricePathSimulationResult(
        symbol="TEST", epoch=1, direction=-1, n_paths=1000, horizon_ticks=10, current_price=100.0,
        mu_per_tick=-0.001, sigma_per_tick=0.01, prob_favorable=0.7, expected_favorable_duration_ticks=3.0,
        terminal_return_mean=0.01, terminal_return_std=0.03, mfe_mean=0.02, mfe_p95=0.05, mae_mean=-0.01, mae_p95=0.02,
    )
    pe = monte_carlo_result_to_probability_estimate(mc)
    # direction=-1, favorable means price went DOWN, so prob_up (absolute) should be 1 - 0.7 = 0.3
    assert pe.prob_up == pytest.approx(0.3)
    assert pe.expected_direction == -1


def test_monte_carlo_adapter_invalid_result():
    mc = PricePathSimulationResult(
        symbol="TEST", epoch=1, direction=0, n_paths=1000, horizon_ticks=10, current_price=100.0,
        mu_per_tick=float("nan"), sigma_per_tick=0.01, prob_favorable=float("nan"), expected_favorable_duration_ticks=float("nan"),
        terminal_return_mean=float("nan"), terminal_return_std=float("nan"), mfe_mean=float("nan"), mfe_p95=float("nan"),
        mae_mean=float("nan"), mae_p95=float("nan"),
    )
    pe = monte_carlo_result_to_probability_estimate(mc)
    assert not pe.is_valid
