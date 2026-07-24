import numpy as np
import pytest

from configs.probability_schema import BaggedGBMConfig
from probability.gbm import BaggedGBMEstimator
from state_encoder.types import MarketState

FEATURE_DIMS = ["trend", "momentum", "acceleration", "volatility", "noise", "persistence",
                "compression_expansion", "complexity", "uncertainty", "market_phase"]


def make_state(symbol="STPRNG100", epoch=0, **overrides) -> MarketState:
    defaults = {dim: 0.0 for dim in FEATURE_DIMS}
    defaults["liquidity"] = 0.0
    defaults.update(overrides)
    return MarketState(symbol=symbol, epoch=epoch, **defaults)


def make_training_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, len(FEATURE_DIMS)))
    # y depends nonlinearly on trend (index 0) and volatility (index 3)
    z = 3.0 * X[:, 0] - 2.0 * X[:, 3] ** 2
    p = 1 / (1 + np.exp(-z))
    y = (rng.uniform(0, 1, n) < p).astype(int)
    return X, y


@pytest.fixture
def config() -> BaggedGBMConfig:
    return BaggedGBMConfig(
        feature_dims=FEATURE_DIMS,
        n_ensemble_members=8,
        max_boosting_iterations=30,
        random_seed=1,
    )


def test_fit_succeeds_with_sufficient_data(config):
    X, y = make_training_data()
    model = BaggedGBMEstimator(config)
    model.fit(X, y)
    assert model.is_fitted
    assert len(model._models) >= 2


def test_predict_before_fit_raises(config):
    model = BaggedGBMEstimator(config)
    with pytest.raises(RuntimeError):
        model.predict(make_state())


def test_fit_rejects_insufficient_data(config):
    model = BaggedGBMEstimator(config)
    X = np.random.default_rng(0).normal(0, 1, size=(10, len(FEATURE_DIMS)))
    y = np.random.default_rng(0).integers(0, 2, size=10)
    with pytest.raises(ValueError, match="at least"):
        model.fit(X, y)


def test_predict_returns_valid_probability(config):
    X, y = make_training_data()
    model = BaggedGBMEstimator(config).fit(X, y)
    result = model.predict(make_state(trend=0.8))
    assert 0.0 <= result.prob_up <= 1.0
    assert result.prob_down == pytest.approx(1.0 - result.prob_up)
    assert result.uncertainty >= 0.0


def test_higher_trend_yields_higher_prob_up_on_average(config):
    X, y = make_training_data()
    model = BaggedGBMEstimator(config).fit(X, y)
    low = model.predict(make_state(epoch=1, trend=-0.9))
    high = model.predict(make_state(epoch=2, trend=0.9))
    assert high.prob_up > low.prob_up


def test_invalid_state_returns_nan(config):
    X, y = make_training_data()
    model = BaggedGBMEstimator(config).fit(X, y)
    result = model.predict(make_state(trend=float("nan")))
    assert result.prob_up != result.prob_up
    assert not result.is_valid


def test_ensemble_uncertainty_is_nonzero_somewhere():
    """With enough members and stochastic bootstrap resampling, cross-member
    disagreement should not be uniformly zero across a range of inputs."""
    config = BaggedGBMConfig(feature_dims=FEATURE_DIMS, n_ensemble_members=10, random_seed=2)
    X, y = make_training_data(n=300, seed=3)
    model = BaggedGBMEstimator(config).fit(X, y)

    uncertainties = []
    rng = np.random.default_rng(4)
    for i in range(20):
        x = {dim: float(rng.uniform(-1, 1)) for dim in FEATURE_DIMS}
        result = model.predict(make_state(epoch=i, **x))
        uncertainties.append(result.uncertainty)
    assert max(uncertainties) > 0.0


def test_fit_rejects_wrong_feature_dimensionality(config):
    model = BaggedGBMEstimator(config)
    X = np.random.default_rng(0).normal(0, 1, size=(50, 3))
    y = np.random.default_rng(0).integers(0, 2, size=50)
    with pytest.raises(ValueError, match="shape"):
        model.fit(X, y)


def test_min_ensemble_size_enforced_by_config():
    with pytest.raises(ValueError, match="n_ensemble_members"):
        BaggedGBMConfig(feature_dims=FEATURE_DIMS, n_ensemble_members=1)
