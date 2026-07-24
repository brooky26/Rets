import numpy as np
import pytest

from configs.drift_detection_schema import DriftDetectionConfig
from drift_detection.detector import DriftDetector
from drift_detection.types import DriftAlert, DriftReport, DriftSeverity, DriftType
from state_encoder.types import MarketState


def make_state(trend=0.0, volatility=0.0, persistence=0.0, compression_expansion=0.0, noise=0.0, epoch=0) -> MarketState:
    return MarketState(
        symbol="STPRNG100", epoch=epoch, trend=trend, momentum=0.0, acceleration=0.0,
        volatility=volatility, noise=noise, persistence=persistence, compression_expansion=compression_expansion,
        complexity=0.0, uncertainty=0.0, liquidity=0.0, market_phase=0.0,
    )


def make_config(**overrides) -> DriftDetectionConfig:
    defaults = dict(feature_dims=["trend", "volatility"], min_samples_required=50)
    defaults.update(overrides)
    return DriftDetectionConfig(**defaults)


def test_feature_drift_none_for_stable_distribution():
    rng = np.random.default_rng(0)
    detector = DriftDetector(make_config())
    reference = [make_state(trend=t, volatility=v, epoch=i) for i, (t, v) in enumerate(
        zip(rng.normal(0, 0.3, 300), rng.normal(0, 0.3, 300))
    )]
    current = [make_state(trend=t, volatility=v, epoch=300 + i) for i, (t, v) in enumerate(
        zip(rng.normal(0, 0.3, 300), rng.normal(0, 0.3, 300))
    )]
    alerts = detector.detect_feature_drift(reference, current)
    assert all(a.severity == DriftSeverity.NONE for a in alerts)


def test_feature_drift_detected_for_shifted_distribution():
    rng = np.random.default_rng(1)
    detector = DriftDetector(make_config())
    reference = [make_state(trend=t, epoch=i) for i, t in enumerate(rng.normal(0, 0.2, 300))]
    current = [make_state(trend=t, epoch=300 + i) for i, t in enumerate(rng.normal(0.8, 0.2, 300))]
    alerts = detector.detect_feature_drift(reference, current)
    trend_alert = next(a for a in alerts if a.metric_name == "trend")
    assert trend_alert.severity == DriftSeverity.CRITICAL


def test_feature_drift_alert_count_matches_configured_dims():
    detector = DriftDetector(make_config(feature_dims=["trend", "volatility"]))
    rng = np.random.default_rng(2)
    reference = [make_state(trend=t, volatility=v, epoch=i) for i, (t, v) in enumerate(
        zip(rng.normal(0, 0.3, 100), rng.normal(0, 0.3, 100))
    )]
    current = [make_state(trend=t, volatility=v, epoch=100 + i) for i, (t, v) in enumerate(
        zip(rng.normal(0, 0.3, 100), rng.normal(0, 0.3, 100))
    )]
    alerts = detector.detect_feature_drift(reference, current)
    assert len(alerts) == 2
    assert {a.metric_name for a in alerts} == {"trend", "volatility"}


def test_distribution_drift_none_for_stable_predictions():
    rng = np.random.default_rng(3)
    detector = DriftDetector(make_config())
    reference = list(np.clip(rng.normal(0.55, 0.05, 300), 0.5, 1.0))
    current = list(np.clip(rng.normal(0.55, 0.05, 300), 0.5, 1.0))
    alert = detector.detect_distribution_drift(reference, current)
    assert alert.severity == DriftSeverity.NONE
    assert alert.drift_type == DriftType.DISTRIBUTION


def test_distribution_drift_detected_for_shifted_predictions():
    rng = np.random.default_rng(4)
    detector = DriftDetector(make_config())
    reference = list(np.clip(rng.normal(0.55, 0.03, 300), 0.5, 1.0))
    current = list(np.clip(rng.normal(0.90, 0.03, 300), 0.5, 1.0))
    alert = detector.detect_distribution_drift(reference, current)
    assert alert.severity == DriftSeverity.CRITICAL


def test_concept_drift_none_when_calibration_unchanged():
    rng = np.random.default_rng(5)
    detector = DriftDetector(make_config())
    ref_predicted = [0.6] * 200
    ref_outcomes = list(rng.uniform(0, 1, 200) < 0.6)
    cur_predicted = [0.6] * 200
    cur_outcomes = list(rng.uniform(0, 1, 200) < 0.6)
    alert = detector.detect_concept_drift(ref_predicted, ref_outcomes, cur_predicted, cur_outcomes)
    assert alert.severity == DriftSeverity.NONE


def test_concept_drift_detected_when_calibration_degrades():
    rng = np.random.default_rng(6)
    detector = DriftDetector(make_config())
    ref_predicted = [0.7] * 300
    ref_outcomes = list(rng.uniform(0, 1, 300) < 0.7)
    cur_predicted = [0.7] * 300
    cur_outcomes = list(rng.uniform(0, 1, 300) < 0.35)
    alert = detector.detect_concept_drift(ref_predicted, ref_outcomes, cur_predicted, cur_outcomes)
    assert alert.severity in (DriftSeverity.WARNING, DriftSeverity.CRITICAL)
    assert alert.statistic > 0


def test_performance_drift_none_for_stable_returns():
    rng = np.random.default_rng(7)
    detector = DriftDetector(make_config())
    reference = list(rng.normal(0.05, 0.1, 300))
    current = list(rng.normal(0.05, 0.1, 300))
    alert = detector.detect_performance_drift(reference, current)
    assert alert.severity == DriftSeverity.NONE


def test_performance_drift_detected_for_decline():
    rng = np.random.default_rng(8)
    detector = DriftDetector(make_config())
    reference = list(rng.normal(0.15, 0.05, 300))
    current = list(rng.normal(-0.05, 0.05, 300))
    alert = detector.detect_performance_drift(reference, current)
    assert alert.severity in (DriftSeverity.WARNING, DriftSeverity.CRITICAL)
    assert alert.statistic < 0


def test_performance_drift_not_flagged_for_improvement():
    rng = np.random.default_rng(9)
    detector = DriftDetector(make_config())
    reference = list(rng.normal(0.0, 0.05, 300))
    current = list(rng.normal(0.20, 0.05, 300))
    alert = detector.detect_performance_drift(reference, current)
    assert alert.severity == DriftSeverity.NONE
    assert alert.statistic > 0


def test_full_report_empty_when_insufficient_samples():
    detector = DriftDetector(make_config(min_samples_required=1000))
    rng = np.random.default_rng(10)
    states = [make_state(trend=t, epoch=i) for i, t in enumerate(rng.normal(0, 0.2, 50))]
    report = detector.run_full_report(
        states, states, [0.6] * 50, [0.6] * 50, [True] * 25 + [False] * 25, [True] * 25 + [False] * 25,
        [0.1] * 50, [0.1] * 50,
    )
    assert report.alerts == ()


def test_full_report_combines_all_four_drift_types():
    rng = np.random.default_rng(11)
    detector = DriftDetector(make_config(feature_dims=["trend"], min_samples_required=50))
    states_ref = [make_state(trend=t, epoch=i) for i, t in enumerate(rng.normal(0, 0.2, 200))]
    states_cur = [make_state(trend=t, epoch=200 + i) for i, t in enumerate(rng.normal(0, 0.2, 200))]
    predicted_ref = list(np.clip(rng.normal(0.6, 0.05, 200), 0.5, 1.0))
    predicted_cur = list(np.clip(rng.normal(0.6, 0.05, 200), 0.5, 1.0))
    outcomes_ref = list(rng.uniform(0, 1, 200) < 0.6)
    outcomes_cur = list(rng.uniform(0, 1, 200) < 0.6)
    returns_ref = list(rng.normal(0.05, 0.1, 200))
    returns_cur = list(rng.normal(0.05, 0.1, 200))

    report = detector.run_full_report(
        states_ref, states_cur, predicted_ref, predicted_cur, outcomes_ref, outcomes_cur,
        returns_ref, returns_cur,
    )
    drift_types_present = {a.drift_type for a in report.alerts}
    assert drift_types_present == {DriftType.FEATURE, DriftType.DISTRIBUTION, DriftType.CONCEPT, DriftType.PERFORMANCE}


def test_should_trigger_retraining_false_for_single_warning():
    alerts = (
        DriftAlert(DriftType.FEATURE, "trend", DriftSeverity.WARNING, 0.15, "x"),
        DriftAlert(DriftType.DISTRIBUTION, "predicted_probability", DriftSeverity.NONE, 0.02, "y"),
    )
    report = DriftReport(alerts=alerts, reference_window_size=100, current_window_size=100)
    assert report.should_trigger_retraining is False


def test_should_trigger_retraining_true_for_two_warnings():
    alerts = (
        DriftAlert(DriftType.FEATURE, "trend", DriftSeverity.WARNING, 0.15, "x"),
        DriftAlert(DriftType.DISTRIBUTION, "predicted_probability", DriftSeverity.WARNING, 0.12, "y"),
    )
    report = DriftReport(alerts=alerts, reference_window_size=100, current_window_size=100)
    assert report.should_trigger_retraining is True


def test_should_trigger_retraining_true_for_single_critical():
    alerts = (
        DriftAlert(DriftType.PERFORMANCE, "mean_return_pct", DriftSeverity.CRITICAL, -0.2, "x"),
    )
    report = DriftReport(alerts=alerts, reference_window_size=100, current_window_size=100)
    assert report.should_trigger_retraining is True
