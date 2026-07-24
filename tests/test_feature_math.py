import numpy as np
import pytest

from features import math_utils as mu


# --------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------- #


def test_log_returns_basic():
    prices = np.array([100.0, 110.0, 121.0])
    returns = mu.log_returns(prices)
    assert len(returns) == 2
    assert returns[0] == pytest.approx(np.log(1.1))
    assert returns[1] == pytest.approx(np.log(1.1))


def test_log_returns_too_short():
    assert len(mu.log_returns(np.array([100.0]))) == 0


# --------------------------------------------------------------------- #
# Trend / momentum
# --------------------------------------------------------------------- #


def test_sma_known_value():
    prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mu.sma(prices, 5) == pytest.approx(3.0)
    assert mu.sma(prices, 3) == pytest.approx(4.0)  # mean of [3,4,5]


def test_sma_insufficient_history():
    assert np.isnan(mu.sma(np.array([1.0, 2.0]), 5))


def test_ema_seeds_with_sma_then_recurses():
    prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    window = 3
    alpha = 2 / (window + 1)
    expected = np.mean(prices[:3])  # seed
    for p in prices[3:]:
        expected = alpha * p + (1 - alpha) * expected
    assert mu.ema(prices, window) == pytest.approx(expected)


def test_wma_weights_recent_more_heavily():
    prices = np.array([1.0, 2.0, 3.0])
    # weights 1,2,3 on 1,2,3 -> (1*1+2*2+3*3)/6 = 14/6
    assert mu.wma(prices, 3) == pytest.approx(14 / 6)


def test_momentum_basic():
    prices = np.array([100.0, 101.0, 102.0, 105.0])
    assert mu.momentum(prices, 3) == pytest.approx(5.0)


def test_roc_basic():
    prices = np.array([100.0, 101.0, 102.0, 110.0])
    assert mu.roc(prices, 3) == pytest.approx(10.0)


def test_rsi_all_gains_is_100():
    prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    assert mu.rsi(prices, 5) == pytest.approx(100.0)


def test_rsi_all_losses_is_0():
    prices = np.array([105.0, 104.0, 103.0, 102.0, 101.0, 100.0])
    assert mu.rsi(prices, 5) == pytest.approx(0.0)


def test_rsi_flat_is_neutral_50():
    prices = np.array([100.0] * 10)
    assert mu.rsi(prices, 5) == pytest.approx(50.0)


def test_rsi_bounded_0_100():
    rng = np.random.default_rng(1)
    prices = 100 + np.cumsum(rng.normal(0, 1, 200))
    value = mu.rsi(prices, 14)
    assert 0.0 <= value <= 100.0


def test_atr_constant_range():
    # high-low always 2, no gaps beyond that -> ATR should be 2
    closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    highs = closes + 1
    lows = closes - 1
    assert mu.atr(highs, lows, closes, 3) == pytest.approx(2.0)


def test_macd_insufficient_history_returns_nan():
    prices = np.array([100.0, 101.0])
    macd_line, signal, hist = mu.macd(prices, fast=12, slow=26, signal=9)
    assert np.isnan(macd_line)
    assert np.isnan(signal)
    assert np.isnan(hist)


def test_macd_sufficient_history_produces_values():
    rng = np.random.default_rng(2)
    prices = 100 + np.cumsum(rng.normal(0, 0.5, 60))
    macd_line, signal, hist = mu.macd(prices, fast=12, slow=26, signal=9)
    assert not np.isnan(macd_line)
    assert not np.isnan(signal)
    assert hist == pytest.approx(macd_line - signal)


# --------------------------------------------------------------------- #
# Volatility
# --------------------------------------------------------------------- #


def test_rolling_std_known_value():
    prices = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    expected_std = float(np.std(prices, ddof=1))
    assert mu.rolling_std(prices, 8) == pytest.approx(expected_std)


def test_zscore_at_mean_is_zero():
    prices = np.array([10.0, 10.0, 10.0, 10.0])
    assert mu.zscore(prices, 4) == pytest.approx(0.0)


def test_zscore_positive_when_above_mean():
    prices = np.array([1.0, 2.0, 3.0, 10.0])
    assert mu.zscore(prices, 4) > 0


# --------------------------------------------------------------------- #
# Statistical
# --------------------------------------------------------------------- #


def test_shannon_entropy_uniform_is_higher_than_constant():
    rng = np.random.default_rng(3)
    uniform_returns = rng.uniform(-1, 1, 200)
    constant_returns = np.zeros(200)
    h_uniform = mu.shannon_entropy(uniform_returns, bins=10)
    h_constant = mu.shannon_entropy(constant_returns, bins=10)
    assert h_uniform > h_constant


def test_skewness_symmetric_distribution_near_zero():
    rng = np.random.default_rng(4)
    symmetric = rng.normal(0, 1, 5000)
    assert abs(mu.skewness(symmetric)) < 0.15


def test_kurtosis_normal_distribution_near_zero():
    rng = np.random.default_rng(5)
    normal_data = rng.normal(0, 1, 20000)
    assert abs(mu.kurtosis(normal_data)) < 0.15


def test_autocorrelation_of_white_noise_near_zero():
    rng = np.random.default_rng(6)
    noise = rng.normal(0, 1, 5000)
    assert abs(mu.autocorrelation(noise, lag=1)) < 0.05


def test_autocorrelation_lag_zero_equivalent_perfect_trend():
    # A perfectly repeating alternating series has strong lag-2 autocorrelation.
    # Not exactly 1.0 due to finite-sample edge effects at the boundary.
    returns = np.tile([1.0, -1.0], 50)
    assert mu.autocorrelation(returns, lag=2) == pytest.approx(1.0, abs=0.05)


# --------------------------------------------------------------------- #
# Fractal / complexity
# --------------------------------------------------------------------- #


def test_hurst_exponent_random_walk_near_half():
    rng = np.random.default_rng(7)
    # Random walk: cumulative sum of iid noise -> log-price should give H ~ 0.5
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 2000)))
    h = mu.hurst_exponent(prices, min_chunk_size=8)
    assert 0.35 < h < 0.65


def test_hurst_exponent_strong_trend_above_half():
    # Strongly trending series (deterministic drift dominates noise) should give H > 0.5
    trend = np.linspace(100, 200, 2000)
    noise = np.random.default_rng(8).normal(0, 0.01, 2000)
    prices = trend + noise
    h = mu.hurst_exponent(prices, min_chunk_size=8)
    assert h > 0.5


def test_hurst_insufficient_history_is_nan():
    assert np.isnan(mu.hurst_exponent(np.array([1.0, 2.0, 3.0]), min_chunk_size=8))


def test_higuchi_fractal_dimension_bounded():
    rng = np.random.default_rng(9)
    prices = 100 + np.cumsum(rng.normal(0, 1, 300))
    d = mu.higuchi_fractal_dimension(prices, k_max=10)
    assert 1.0 <= d <= 2.2  # allow small numerical slack beyond the theoretical [1,2]


def test_higuchi_insufficient_history_is_nan():
    assert np.isnan(mu.higuchi_fractal_dimension(np.array([1.0, 2.0]), k_max=10))
