import numpy as np
import pytest

from state_encoder.normalizer import OnlineNormalizer


def test_matches_numpy_mean_and_std():
    rng = np.random.default_rng(0)
    data = rng.normal(10, 3, 500)
    normalizer = OnlineNormalizer()
    for x in data:
        normalizer.update("f", float(x))

    expected_mean = np.mean(data)
    expected_std = np.std(data, ddof=1)

    stats = normalizer._stats_for("f")
    assert stats.mean == pytest.approx(expected_mean, rel=1e-9)
    assert stats.std == pytest.approx(expected_std, rel=1e-9)


def test_zscore_matches_manual_computation():
    rng = np.random.default_rng(1)
    data = rng.normal(0, 1, 300)
    normalizer = OnlineNormalizer()
    for x in data:
        normalizer.update("f", float(x))

    mean = np.mean(data)
    std = np.std(data, ddof=1)
    test_value = 2.5
    expected_z = (test_value - mean) / std
    assert normalizer.zscore("f", test_value) == pytest.approx(expected_z, rel=1e-9)


def test_first_observation_zscore_is_zero_std_undefined():
    normalizer = OnlineNormalizer()
    # After a single observation, variance is undefined (n=1) -> std=0 -> zscore defined as 0
    z = normalizer.update_and_zscore("f", 5.0)
    assert z == 0.0


def test_nan_input_does_not_pollute_stats():
    normalizer = OnlineNormalizer()
    normalizer.update("f", 1.0)
    normalizer.update("f", float("nan"))
    normalizer.update("f", 3.0)
    assert normalizer.sample_count("f") == 2
    assert normalizer._stats_for("f").mean == pytest.approx(2.0)


def test_zscore_of_nan_is_nan():
    normalizer = OnlineNormalizer()
    normalizer.update("f", 1.0)
    normalizer.update("f", 2.0)
    result = normalizer.zscore("f", float("nan"))
    assert result != result  # NaN


def test_independent_keys_do_not_interfere():
    normalizer = OnlineNormalizer()
    for x in [1.0, 2.0, 3.0]:
        normalizer.update("a", x)
    for x in [100.0, 200.0, 300.0]:
        normalizer.update("b", x)
    assert normalizer._stats_for("a").mean == pytest.approx(2.0)
    assert normalizer._stats_for("b").mean == pytest.approx(200.0)


def test_serialization_round_trip():
    normalizer = OnlineNormalizer()
    for x in [1.0, 2.0, 3.0, 4.0]:
        normalizer.update("f", x)

    data = normalizer.to_dict()
    restored = OnlineNormalizer.from_dict(data)

    assert restored.sample_count("f") == normalizer.sample_count("f")
    assert restored.zscore("f", 2.5) == pytest.approx(normalizer.zscore("f", 2.5))


def test_update_and_zscore_reflects_the_new_observation():
    """
    update_and_zscore computes the z-score AFTER folding in the new
    value (matches how a streaming system actually experiences data).
    """
    normalizer = OnlineNormalizer()
    normalizer.update("f", 1.0)
    normalizer.update("f", 1.0)
    # Third value is a big outlier relative to [1.0, 1.0]
    z = normalizer.update_and_zscore("f", 100.0)
    assert z > 0  # should register as above the (now-updated) mean
    assert normalizer.sample_count("f") == 3
