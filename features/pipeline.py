"""
Feature Engineering Pipeline.

Maintains a rolling per-symbol buffer of Candles and, on each new candle,
computes the full feature vector by calling the pure functions in
`features/math_utils.py`. This class owns *only* the buffering — every
actual formula lives in math_utils so the same code path can be reused
for offline batch computation (training/backtesting) by calling
`compute_feature_vector()` directly on a numpy array slice, without
going through this stateful wrapper at all.

That split is what satisfies "identical feature generation during
training, validation and live trading": there's one function per
feature, period.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from configs.feature_schema import FeatureEngineeringConfig
from data.types import Candle
from features import math_utils as mu
from features.types import FeatureVector


def compute_feature_vector(
    symbol: str,
    epoch: int,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    config: FeatureEngineeringConfig,
) -> FeatureVector:
    """
    Pure function: given aligned closes/highs/lows arrays (oldest -> newest,
    most recent price last), compute every feature and return a FeatureVector.
    Called identically by the live streaming pipeline and by offline
    batch/backtest code.
    """
    values: dict[str, float] = {}
    returns = mu.log_returns(closes)
    m = config.momentum
    v = config.volatility
    s = config.statistical
    f = config.fractal

    # --- Momentum / trend ---
    for w in m.sma_windows:
        values[f"sma_{w}"] = mu.sma(closes, w)
    for w in m.ema_windows:
        values[f"ema_{w}"] = mu.ema(closes, w)
    for w in m.wma_windows:
        values[f"wma_{w}"] = mu.wma(closes, w)
    for w in m.momentum_windows:
        values[f"momentum_{w}"] = mu.momentum(closes, w)
    for w in m.roc_windows:
        values[f"roc_{w}"] = mu.roc(closes, w)
    values["velocity"] = mu.velocity(closes, m.velocity_window)
    values["acceleration"] = mu.acceleration(closes, m.acceleration_window)

    macd_line, macd_signal, macd_hist = mu.macd(
        closes, m.macd_fast, m.macd_slow, m.macd_signal
    )
    values["macd_line"] = macd_line
    values["macd_signal"] = macd_signal
    values["macd_histogram"] = macd_hist

    values["rsi"] = mu.rsi(closes, m.rsi_window)

    # --- Volatility ---
    values["atr"] = mu.atr(highs, lows, closes, v.atr_window)
    for w in v.std_windows:
        values[f"std_{w}"] = mu.rolling_std(closes, w)
        values[f"variance_{w}"] = mu.rolling_variance(closes, w)
    values["zscore"] = mu.zscore(closes, v.zscore_window)

    # --- Statistical / distributional (computed on returns) ---
    entropy_window_returns = (
        returns[-s.entropy_window :] if len(returns) >= s.entropy_window else returns
    )
    values["entropy"] = mu.shannon_entropy(entropy_window_returns, s.entropy_bins)

    skew_kurt_returns = (
        returns[-s.skew_kurt_window :]
        if len(returns) >= s.skew_kurt_window
        else returns
    )
    values["skewness"] = mu.skewness(skew_kurt_returns)
    values["kurtosis"] = mu.kurtosis(skew_kurt_returns)

    autocorr_returns = (
        returns[-s.autocorrelation_window :]
        if len(returns) >= s.autocorrelation_window
        else returns
    )
    for lag in s.autocorrelation_lags:
        values[f"autocorrelation_lag_{lag}"] = mu.autocorrelation(autocorr_returns, lag)

    # --- Fractal / complexity ---
    hurst_prices = (
        closes[-f.hurst_window :] if len(closes) >= f.hurst_window else closes
    )
    values["hurst_exponent"] = mu.hurst_exponent(hurst_prices, f.hurst_min_chunk_size)

    higuchi_prices = (
        closes[-f.higuchi_window :] if len(closes) >= f.higuchi_window else closes
    )
    values["fractal_dimension"] = mu.higuchi_fractal_dimension(
        higuchi_prices, f.higuchi_k_max
    )

    return FeatureVector(symbol=symbol, epoch=epoch, values=values)


class FeatureEngineeringPipeline:
    """
    Stateful streaming wrapper: buffers candles per symbol and computes a
    FeatureVector on each new candle via `compute_feature_vector`.
    """

    def __init__(self, config: FeatureEngineeringConfig, buffer_size: int | None = None):
        self._config = config
        self._buffer_size = buffer_size or (config.min_history_required + 50)
        self._buffers: dict[str, deque[Candle]] = {}

    def on_candle(self, candle: Candle) -> FeatureVector | None:
        """
        Feed one new candle for its symbol. Returns a FeatureVector once
        enough history has accumulated, else None (not enough data yet —
        distinct from a FeatureVector with NaNs, which callers should
        also treat as "don't trade yet" but is programmatically available
        for inspection/logging).
        """
        buf = self._buffers.setdefault(
            candle.symbol, deque(maxlen=self._buffer_size)
        )
        buf.append(candle)

        if len(buf) < self._config.min_history_required:
            return None

        closes = np.array([c.close for c in buf], dtype=np.float64)
        highs = np.array([c.high for c in buf], dtype=np.float64)
        lows = np.array([c.low for c in buf], dtype=np.float64)

        return compute_feature_vector(
            symbol=candle.symbol,
            epoch=candle.epoch,
            closes=closes,
            highs=highs,
            lows=lows,
            config=self._config,
        )

    def buffered_count(self, symbol: str) -> int:
        return len(self._buffers.get(symbol, ()))
