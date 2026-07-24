import numpy as np
import pytest

from drift_detection.psi import population_stability_index


def test_psi_near_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 2000)
    current = rng.normal(0, 1, 2000)
    psi = population_stability_index(reference, current, n_bins=10)
    assert psi < 0.05


def test_psi_large_for_very_different_distributions():
    rng = np.random.default_rng(1)
    reference = rng.normal(0, 1, 2000)
    current = rng.normal(5, 1, 2000)
    psi = population_stability_index(reference, current, n_bins=10)
    assert psi > 0.25


def test_psi_increases_monotonically_with_shift_magnitude():
    rng = np.random.default_rng(2)
    reference = rng.normal(0, 1, 2000)
    small_shift = population_stability_index(reference, rng.normal(0.2, 1, 2000), n_bins=10)
    medium_shift = population_stability_index(reference, rng.normal(1.0, 1, 2000), n_bins=10)
    large_shift = population_stability_index(reference, rng.normal(3.0, 1, 2000), n_bins=10)
    assert small_shift < medium_shift < large_shift


def test_psi_zero_for_near_constant_reference():
    reference = np.full(200, 5.0)
    current = np.array([5.0, 5.1, 4.9] * 50)
    psi = population_stability_index(reference, current, n_bins=10)
    assert psi == 0.0


def test_psi_rejects_reference_smaller_than_bin_count():
    with pytest.raises(ValueError, match="too small"):
        population_stability_index(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]), n_bins=10)


def test_psi_rejects_empty_current():
    with pytest.raises(ValueError, match="must not be empty"):
        population_stability_index(np.random.default_rng(0).normal(0, 1, 100), np.array([]), n_bins=10)


def test_psi_handles_current_values_outside_reference_range():
    reference = np.random.default_rng(3).normal(0, 1, 500)
    current = np.array([100.0, -100.0, 50.0] * 50)
    psi = population_stability_index(reference, current, n_bins=10)
    assert psi > 0
    assert np.isfinite(psi)
