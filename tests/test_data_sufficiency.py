import pytest

from meta_learning.sufficiency import apply_sufficiency_scaling, compute_sufficiency_ratio
from meta_learning.types import EnsembleWeights


def test_compute_sufficiency_ratio_below_target():
    assert compute_sufficiency_ratio(100, 500) == pytest.approx(0.2)


def test_compute_sufficiency_ratio_capped_at_one():
    assert compute_sufficiency_ratio(1000, 500) == 1.0
    assert compute_sufficiency_ratio(500, 500) == 1.0


def test_compute_sufficiency_ratio_rejects_bad_input():
    with pytest.raises(ValueError):
        compute_sufficiency_ratio(-1, 500)
    with pytest.raises(ValueError):
        compute_sufficiency_ratio(100, 0)


def test_apply_sufficiency_scaling_shrinks_undertrained_model():
    weights = EnsembleWeights(
        weights={"bayesian_logistic": 0.5, "bagged_gbm": 0.5}, source="default",
    )
    scaled = apply_sufficiency_scaling(weights, {"bayesian_logistic": 1.0, "bagged_gbm": 0.2})

    assert scaled.weights["bagged_gbm"] < scaled.weights["bayesian_logistic"]
    assert sum(scaled.weights.values()) == pytest.approx(1.0)


def test_apply_sufficiency_scaling_leaves_untracked_models_unaffected():
    weights = EnsembleWeights(
        weights={"bayesian_logistic": 0.4, "bagged_gbm": 0.3, "monte_carlo_gbm": 0.3}, source="default",
    )
    # monte_carlo_gbm not present in sufficiency dict — treated as 1.0 (unaffected).
    scaled = apply_sufficiency_scaling(weights, {"bayesian_logistic": 1.0, "bagged_gbm": 0.5})

    # Both bayesian_logistic and monte_carlo_gbm are scaled by an effective
    # multiplier of 1.0 — their RATIO to each other must be exactly
    # preserved from the original weights (0.4 / 0.3), even though the
    # absolute values change due to renormalization against the shrunk
    # bagged_gbm.
    original_ratio = weights.weights["bayesian_logistic"] / weights.weights["monte_carlo_gbm"]
    scaled_ratio = scaled.weights["bayesian_logistic"] / scaled.weights["monte_carlo_gbm"]
    assert scaled_ratio == pytest.approx(original_ratio)
    assert scaled.weights["bagged_gbm"] < weights.weights["bagged_gbm"]  # shrunk in absolute terms
    assert sum(scaled.weights.values()) == pytest.approx(1.0)


def test_apply_sufficiency_scaling_full_sufficiency_is_a_no_op():
    weights = EnsembleWeights(
        weights={"bayesian_logistic": 0.6, "bagged_gbm": 0.4}, source="default",
    )
    scaled = apply_sufficiency_scaling(weights, {"bayesian_logistic": 1.0, "bagged_gbm": 1.0})

    assert scaled.weights["bayesian_logistic"] == pytest.approx(0.6)
    assert scaled.weights["bagged_gbm"] == pytest.approx(0.4)


def test_apply_sufficiency_scaling_falls_back_to_equal_weights_when_everything_zero():
    weights = EnsembleWeights(
        weights={"bayesian_logistic": 0.6, "bagged_gbm": 0.4}, source="default",
    )
    scaled = apply_sufficiency_scaling(weights, {"bayesian_logistic": 0.0, "bagged_gbm": 0.0})

    assert scaled.weights["bayesian_logistic"] == pytest.approx(0.5)
    assert scaled.weights["bagged_gbm"] == pytest.approx(0.5)


def test_apply_sufficiency_scaling_preserves_source_and_n_trials_lineage():
    weights = EnsembleWeights(
        weights={"bayesian_logistic": 1.0}, source="bayesian_optimizer", n_trials=42,
    )
    scaled = apply_sufficiency_scaling(weights, {"bayesian_logistic": 1.0})

    assert "bayesian_optimizer" in scaled.source
    assert "sufficiency_scaled" in scaled.source
    assert scaled.n_trials == 42
