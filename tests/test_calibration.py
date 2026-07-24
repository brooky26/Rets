import numpy as np
import pytest

from configs.probability_schema import CalibrationConfig
from probability.calibration import PlattCalibrator


@pytest.fixture
def config() -> CalibrationConfig:
    return CalibrationConfig(max_iterations=100, tolerance=1e-8)


def test_fit_converges_on_well_behaved_data(config):
    rng = np.random.default_rng(0)
    true_A, true_B = 2.0, -0.5
    s = rng.normal(0, 1, 500)
    z = true_A * s + true_B
    p = 1 / (1 + np.exp(-z))
    y = (rng.uniform(0, 1, 500) < p).astype(int)

    calibrator = PlattCalibrator(config)
    calibrator.fit(s, y)
    assert calibrator.is_fitted
    assert calibrator.converged


def test_fit_recovers_approximate_true_parameters(config):
    rng = np.random.default_rng(1)
    true_A, true_B = 3.0, 0.5
    s = rng.normal(0, 1, 2000)
    z = true_A * s + true_B
    p = 1 / (1 + np.exp(-z))
    y = (rng.uniform(0, 1, 2000) < p).astype(int)

    calibrator = PlattCalibrator(config).fit(s, y)
    assert calibrator.A == pytest.approx(true_A, abs=0.5)
    assert calibrator.B == pytest.approx(true_B, abs=0.3)


def test_transform_before_fit_raises(config):
    calibrator = PlattCalibrator(config)
    with pytest.raises(RuntimeError):
        calibrator.transform(0.5)


def test_transform_monotonic_in_raw_score(config):
    rng = np.random.default_rng(2)
    s = rng.normal(0, 1, 500)
    p = 1 / (1 + np.exp(-2 * s))
    y = (rng.uniform(0, 1, 500) < p).astype(int)
    calibrator = PlattCalibrator(config).fit(s, y)

    scores = np.linspace(-3, 3, 20)
    calibrated = [calibrator.transform(s) for s in scores]
    assert all(a <= b + 1e-9 for a, b in zip(calibrated, calibrated[1:]))


def test_transform_returns_valid_probability(config):
    rng = np.random.default_rng(3)
    s = rng.normal(0, 1, 300)
    p = 1 / (1 + np.exp(-1.5 * s))
    y = (rng.uniform(0, 1, 300) < p).astype(int)
    calibrator = PlattCalibrator(config).fit(s, y)

    for raw in [-5.0, -1.0, 0.0, 1.0, 5.0]:
        p = calibrator.transform(raw)
        assert 0.0 <= p <= 1.0


def test_fit_rejects_mismatched_lengths(config):
    calibrator = PlattCalibrator(config)
    with pytest.raises(ValueError, match="same length"):
        calibrator.fit(np.array([0.1, 0.2, 0.3]), np.array([0, 1]))


def test_fit_rejects_non_binary_labels(config):
    calibrator = PlattCalibrator(config)
    s = np.random.default_rng(0).normal(0, 1, 20)
    y = np.random.default_rng(0).integers(0, 3, 20)
    with pytest.raises(ValueError, match="binary"):
        calibrator.fit(s, y)


def test_fit_rejects_too_few_observations(config):
    calibrator = PlattCalibrator(config)
    with pytest.raises(ValueError, match="at least"):
        calibrator.fit(np.array([0.1, 0.2]), np.array([0, 1]))


def test_calibration_improves_a_badly_scaled_raw_score(config):
    """
    Construct a raw score that's monotonically related to the true
    probability but on the wrong scale (e.g. an un-normalized logit with
    the wrong slope) — after Platt scaling, the calibrated probabilities
    should be closer to the true probabilities than the naive
    sigmoid(raw_score) would be.
    """
    rng = np.random.default_rng(4)
    true_z = rng.normal(0, 1, 1000)
    true_p = 1 / (1 + np.exp(-true_z))
    y = (rng.uniform(0, 1, 1000) < true_p).astype(int)

    # Badly-scaled raw score: true_z compressed by a factor of 10 and shifted.
    raw_score = true_z / 10.0 + 2.0

    naive_p = 1 / (1 + np.exp(-raw_score))
    naive_error = np.mean(np.abs(naive_p - true_p))

    calibrator = PlattCalibrator(config).fit(raw_score, y)
    calibrated_p = np.array([calibrator.transform(s) for s in raw_score])
    calibrated_error = np.mean(np.abs(calibrated_p - true_p))

    assert calibrated_error < naive_error
