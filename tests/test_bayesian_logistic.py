import numpy as np
import pytest

from configs.probability_schema import BayesianLogisticConfig
from probability.bayesian_logistic import BayesianLogisticRegression
from state_encoder.types import MarketState

FEATURE_DIMS = ["trend", "momentum", "acceleration", "volatility", "persistence", "compression_expansion"]


def make_state(symbol="STPRNG100", epoch=0, **overrides) -> MarketState:
    defaults = dict(
        trend=0.0, momentum=0.0, acceleration=0.0, volatility=0.0, noise=0.0,
        persistence=0.0, compression_expansion=0.0, complexity=0.0,
        uncertainty=0.0, liquidity=0.0, market_phase=0.0,
    )
    defaults.update(overrides)
    return MarketState(symbol=symbol, epoch=epoch, **defaults)


def make_separable_data(n=300, seed=0):
    """
    Synthetic data where y is a genuine (noisy) logistic function of
    `trend` alone, other dims pure noise — lets us check the model
    recovers a sensible sign/magnitude on the informative dimension.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, len(FEATURE_DIMS)))
    true_w = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # only "trend" (index 0) matters
    z = X @ true_w
    p = 1 / (1 + np.exp(-z))
    y = (rng.uniform(0, 1, n) < p).astype(int)
    return X, y


@pytest.fixture
def config() -> BayesianLogisticConfig:
    return BayesianLogisticConfig(feature_dims=FEATURE_DIMS, prior_precision=1.0, max_iterations=50)


def test_fit_converges_on_separable_data(config):
    X, y = make_separable_data()
    model = BayesianLogisticRegression(config)
    model.fit(X, y)
    assert model.is_fitted
    assert model.converged


def test_fit_recovers_informative_dimension_sign(config):
    X, y = make_separable_data()
    model = BayesianLogisticRegression(config)
    model.fit(X, y)
    # w[0] is the intercept (include_intercept=True), w[1] corresponds to "trend"
    trend_weight = model.w_map[1]
    assert trend_weight > 0.5  # should recover a clearly positive weight


def test_predict_before_fit_raises(config):
    model = BayesianLogisticRegression(config)
    with pytest.raises(RuntimeError):
        model.predict(make_state())


def test_predict_returns_valid_probability(config):
    X, y = make_separable_data()
    model = BayesianLogisticRegression(config).fit(X, y)
    result = model.predict(make_state(trend=0.8))
    assert 0.0 <= result.prob_up <= 1.0
    assert result.prob_down == pytest.approx(1.0 - result.prob_up)
    assert 0.0 <= result.uncertainty < 1.0


def test_higher_trend_yields_higher_prob_up(config):
    X, y = make_separable_data()
    model = BayesianLogisticRegression(config).fit(X, y)
    low = model.predict(make_state(epoch=1, trend=-0.8))
    high = model.predict(make_state(epoch=2, trend=0.8))
    assert high.prob_up > low.prob_up


def test_uncertainty_higher_for_out_of_distribution_point(config):
    """
    Points far outside the training distribution (all dims pinned to
    extreme values simultaneously) should show higher predictive
    uncertainty than a point similar to the bulk of training data,
    since x^T Sigma x grows with distance from the data in feature space.
    """
    X, y = make_separable_data(seed=1)
    model = BayesianLogisticRegression(config).fit(X, y)

    typical = model.predict(make_state(epoch=1, trend=0.1, momentum=0.05, acceleration=0.0,
                                        volatility=0.1, persistence=0.05, compression_expansion=0.0))
    extreme = model.predict(make_state(epoch=2, trend=1.0, momentum=1.0, acceleration=1.0,
                                        volatility=1.0, persistence=1.0, compression_expansion=1.0))
    assert extreme.uncertainty > typical.uncertainty


def test_invalid_state_returns_nan(config):
    X, y = make_separable_data()
    model = BayesianLogisticRegression(config).fit(X, y)
    result = model.predict(make_state(trend=float("nan")))
    assert result.prob_up != result.prob_up  # NaN
    assert not result.is_valid


def test_fit_rejects_wrong_shape(config):
    model = BayesianLogisticRegression(config)
    X = np.random.default_rng(0).normal(0, 1, size=(50, 3))  # wrong n_features
    y = np.random.default_rng(0).integers(0, 2, size=50)
    with pytest.raises(ValueError, match="shape"):
        model.fit(X, y)


def test_fit_rejects_non_binary_labels(config):
    model = BayesianLogisticRegression(config)
    X = np.random.default_rng(0).normal(0, 1, size=(50, len(FEATURE_DIMS)))
    y = np.random.default_rng(0).integers(0, 3, size=50)  # labels 0,1,2
    with pytest.raises(ValueError, match="binary"):
        model.fit(X, y)


def test_fit_rejects_too_few_observations(config):
    model = BayesianLogisticRegression(config)
    X = np.zeros((3, len(FEATURE_DIMS)))
    y = np.array([0, 1, 0])
    with pytest.raises(ValueError, match="at least"):
        model.fit(X, y)


def test_stronger_prior_shrinks_weights_toward_zero():
    X, y = make_separable_data(seed=2)
    weak_prior_config = BayesianLogisticConfig(feature_dims=FEATURE_DIMS, prior_precision=0.01)
    strong_prior_config = BayesianLogisticConfig(feature_dims=FEATURE_DIMS, prior_precision=100.0)

    weak_model = BayesianLogisticRegression(weak_prior_config).fit(X, y)
    strong_model = BayesianLogisticRegression(strong_prior_config).fit(X, y)

    assert np.linalg.norm(strong_model.w_map) < np.linalg.norm(weak_model.w_map)


def test_confidence_bounded_half_to_one(config):
    X, y = make_separable_data()
    model = BayesianLogisticRegression(config).fit(X, y)
    for trend in [-1.0, -0.3, 0.0, 0.3, 1.0]:
        result = model.predict(make_state(epoch=int(trend * 10) + 100, trend=trend))
        assert 0.5 <= result.confidence <= 1.0


def test_predict_clamps_sigmoid_saturation_from_out_of_distribution_point(config, caplog, monkeypatch):
    """
    Regression test: predict()'s final sigmoid call was unclamped, unlike
    fit()'s internal iterations. Note: the MacKay uncertainty term (kappa)
    self-dampens as a single feature dimension grows extreme (predictive
    variance grows, shrinking mu_a*kappa back down), so naturally
    triggering true float64 saturation through one fitted model's numerics
    is fragile and config-dependent — worth knowing that epistemic
    uncertainty is doing real protective work here. This test instead
    verifies the clamp itself deterministically: force the internal
    sigmoid call to return exactly 1.0 (as it genuinely can in production
    when enough dimensions are simultaneously out-of-distribution in a
    direction the posterior covariance doesn't dampen), and confirm
    predict() clamps it rather than passing it through, and logs a warning.
    """
    X, y = make_separable_data()
    model = BayesianLogisticRegression(config).fit(X, y)

    import probability.bayesian_logistic as blr_module
    monkeypatch.setattr(blr_module, "_sigmoid", lambda z: np.array([1.0]))

    with caplog.at_level("WARNING", logger="probability.bayesian_logistic"):
        result = model.predict(make_state(epoch=999, trend=0.5))

    assert result.confidence < 1.0, (
        f"confidence={result.confidence!r} — sigmoid saturation was not clamped"
    )
    assert result.prob_up < 1.0
    assert result.prob_down > 0.0
    assert any("saturated" in record.message for record in caplog.records)


def test_predict_does_not_warn_for_in_distribution_points(config, caplog):
    """No false-positive warnings for ordinary, in-distribution predictions
    — the warning should be specific to genuine saturation, not routine."""
    X, y = make_separable_data()
    model = BayesianLogisticRegression(config).fit(X, y)

    with caplog.at_level("WARNING", logger="probability.bayesian_logistic"):
        model.predict(make_state(epoch=100, trend=0.5))

    assert not any("saturated" in record.message for record in caplog.records)
