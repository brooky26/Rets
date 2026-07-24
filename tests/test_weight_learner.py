import pytest

from configs.meta_learning_schema import MetaLearningConfig
from meta_learning.types import EnsembleWeights
from meta_learning.weight_learner import WeightLearner
from regime.types import RegimeLabel


def make_config(**overrides) -> MetaLearningConfig:
    defaults = dict(
        model_names=["bayesian_logistic", "bagged_gbm"],
        min_regime_samples_before_specific_weights=10,
    )
    defaults.update(overrides)
    return MetaLearningConfig(**defaults)


def test_default_weights_are_equal():
    learner = WeightLearner(make_config())
    w = learner.get_weights()
    assert w.weights == {"bayesian_logistic": 0.5, "bagged_gbm": 0.5}
    assert w.source == "default"


def test_ensemble_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        EnsembleWeights(weights={"a": 0.3, "b": 0.3}, source="test")


def test_ensemble_weights_rejects_negative():
    with pytest.raises(ValueError):
        EnsembleWeights(weights={"a": -0.5, "b": 1.5}, source="test")


def test_set_global_weights_updates_get_weights():
    learner = WeightLearner(make_config())
    new_weights = EnsembleWeights(weights={"bayesian_logistic": 0.7, "bagged_gbm": 0.3}, source="bayesian_optimizer", n_trials=50)
    learner.set_global_weights(new_weights)
    assert learner.get_weights().weights == {"bayesian_logistic": 0.7, "bagged_gbm": 0.3}


def test_set_global_weights_rejects_mismatched_keys():
    learner = WeightLearner(make_config())
    bad = EnsembleWeights(weights={"bayesian_logistic": 1.0}, source="test")
    with pytest.raises(ValueError):
        learner.set_global_weights(bad)


def test_regime_weights_fallback_to_global_when_insufficient_trials():
    learner = WeightLearner(make_config(min_regime_samples_before_specific_weights=20))
    sparse = EnsembleWeights(weights={"bayesian_logistic": 0.9, "bagged_gbm": 0.1}, source="bayesian_optimizer", n_trials=5)
    learner.set_regime_weights(RegimeLabel.STRONG_TREND, sparse)
    # Insufficient trials -> falls back to global (equal weights)
    result = learner.get_weights(RegimeLabel.STRONG_TREND)
    assert result.weights == {"bayesian_logistic": 0.5, "bagged_gbm": 0.5}


def test_regime_weights_used_when_sufficient_trials():
    learner = WeightLearner(make_config(min_regime_samples_before_specific_weights=20))
    sufficient = EnsembleWeights(weights={"bayesian_logistic": 0.9, "bagged_gbm": 0.1}, source="bayesian_optimizer", n_trials=25)
    learner.set_regime_weights(RegimeLabel.STRONG_TREND, sufficient)
    result = learner.get_weights(RegimeLabel.STRONG_TREND)
    assert result.weights == {"bayesian_logistic": 0.9, "bagged_gbm": 0.1}
    # A different, unset regime still falls back to global.
    other = learner.get_weights(RegimeLabel.RANGE)
    assert other.weights == {"bayesian_logistic": 0.5, "bagged_gbm": 0.5}


def test_save_and_load_round_trip(tmp_path):
    learner = WeightLearner(make_config())
    learner.set_global_weights(EnsembleWeights(weights={"bayesian_logistic": 0.6, "bagged_gbm": 0.4}, source="bayesian_optimizer", n_trials=30))
    learner.set_regime_weights(
        RegimeLabel.HIGH_VOLATILITY,
        EnsembleWeights(weights={"bayesian_logistic": 0.2, "bagged_gbm": 0.8}, source="bayesian_optimizer", n_trials=40),
    )
    path = tmp_path / "weights.json"
    learner.save_to_file(path)

    reloaded = WeightLearner(make_config())
    reloaded.load_from_file(path)
    assert reloaded.get_weights().weights == {"bayesian_logistic": 0.6, "bagged_gbm": 0.4}
    assert reloaded.get_weights(RegimeLabel.HIGH_VOLATILITY).weights == {"bayesian_logistic": 0.2, "bagged_gbm": 0.8}
