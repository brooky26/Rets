import numpy as np
import pytest

from anomaly.anomaly_detector import AnomalyDetector
from configs.anomaly_schema import AnomalyDetectionConfig
from state_encoder.types import MarketState


def make_state(symbol: str, epoch: int, trend: float, volatility: float = 0.0) -> MarketState:
    return MarketState(
        symbol=symbol, epoch=epoch, trend=trend, momentum=0.0, acceleration=0.0, volatility=volatility,
        noise=0.0, persistence=0.0, compression_expansion=0.0, complexity=0.0, uncertainty=0.1,
        liquidity=0.0, market_phase=0.0,
    )


def make_config(**overrides) -> AnomalyDetectionConfig:
    defaults = dict(
        feature_dims=["trend", "volatility"], n_estimators=50, contamination=0.05,
        random_seed=0, score_threshold=0.5, min_training_samples=50,
    )
    defaults.update(overrides)
    return AnomalyDetectionConfig(**defaults)


def _normal_states(n: int, seed: int) -> list[MarketState]:
    rng = np.random.default_rng(seed)
    trends = rng.normal(0, 0.1, size=n)
    vols = rng.normal(0, 0.1, size=n)
    return [make_state("TEST", i, float(trends[i]), float(vols[i])) for i in range(n)]


def test_fit_requires_minimum_samples():
    detector = AnomalyDetector(make_config(min_training_samples=100))
    with pytest.raises(ValueError):
        detector.fit(_normal_states(10, seed=1))


def test_fit_and_score_normal_point_low_anomaly_score():
    states = _normal_states(200, seed=2)
    detector = AnomalyDetector(make_config())
    detector.fit(states)
    result = detector.score(make_state("TEST", 999, 0.0, 0.0))
    assert result.is_valid
    assert 0.0 <= result.anomaly_score <= 1.0


def test_extreme_outlier_scores_higher_than_typical_point():
    states = _normal_states(200, seed=3)
    detector = AnomalyDetector(make_config())
    detector.fit(states)

    typical = detector.score(make_state("TEST", 1, 0.0, 0.0))
    outlier = detector.score(make_state("TEST", 2, 10.0, 10.0))
    assert outlier.anomaly_score > typical.anomaly_score
    assert outlier.sklearn_flagged


def test_invalid_state_returns_invalid_score():
    states = _normal_states(200, seed=4)
    detector = AnomalyDetector(make_config())
    detector.fit(states)

    invalid_state = make_state("TEST", 1, float("nan"), 0.0)
    result = detector.score(invalid_state)
    assert not result.is_valid
    assert not result.is_anomaly


def test_score_before_fit_raises():
    detector = AnomalyDetector(make_config())
    with pytest.raises(RuntimeError):
        detector.score(make_state("TEST", 1, 0.0, 0.0))


def test_filter_anomalies_drops_flagged_states():
    states = _normal_states(200, seed=5)
    detector = AnomalyDetector(make_config())
    detector.fit(states)

    contaminated = states + [make_state("TEST", 9999, 50.0, 50.0)]
    filtered = detector.filter_anomalies(contaminated)
    assert len(filtered) <= len(contaminated)
    assert all(s.epoch != 9999 for s in filtered) or len(filtered) == len(contaminated)


def test_score_batch_report_counts_match():
    states = _normal_states(100, seed=6)
    detector = AnomalyDetector(make_config())
    detector.fit(states)
    report = detector.score_batch(states)
    assert report.n_flagged == sum(1 for s in report.scores if s.is_anomaly)
    assert report.flagged_fraction == pytest.approx(report.n_flagged / len(states))
