"""
Feature Engineering — pure math core.

Every function here is a pure function of a numpy array (no state, no
I/O). This is deliberate: it's what guarantees identical feature
generation during training, validation, and live trading — the
streaming pipeline (features/pipeline.py) and any offline batch/backtest
code both call exactly these functions on a price-history window, so
there is only one implementation of each formula to ever get out of sync.

All functions return `float('nan')` when there isn't enough history to
compute a well-defined value, rather than raising — callers decide how
to handle incomplete feature vectors (typically: don't trade until full).
"""

from __future__ import annotations

import numpy as np

NAN = float("nan")


# --------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------- #


def log_returns(prices: np.ndarray) -> np.ndarray:
    """r_t = ln(P_t / P_{t-1}). Length = len(prices) - 1."""
    if len(prices) < 2:
        return np.array([])
    return np.diff(np.log(prices))


# --------------------------------------------------------------------- #
# Trend / momentum
# --------------------------------------------------------------------- #


def sma(prices: np.ndarray, window: int) -> float:
    """Simple moving average of the last `window` prices."""
    if len(prices) < window:
        return NAN
    return float(np.mean(prices[-window:]))


def ema(prices: np.ndarray, window: int) -> float:
    """
    Exponential moving average, seeded with the SMA of the first
    `window` prices, then recursively:
        EMA_t = alpha * P_t + (1 - alpha) * EMA_{t-1},  alpha = 2 / (window + 1)
    """
    if len(prices) < window:
        return NAN
    alpha = 2.0 / (window + 1)
    series = prices[-(len(prices)) :]  # use full available history for a stable seed
    # Seed with SMA of the first `window` points in the available series.
    ema_val = float(np.mean(series[:window]))
    for price in series[window:]:
        ema_val = alpha * price + (1 - alpha) * ema_val
    return ema_val


def wma(prices: np.ndarray, window: int) -> float:
    """Weighted moving average: linearly increasing weights, most recent price weighted highest."""
    if len(prices) < window:
        return NAN
    weights = np.arange(1, window + 1)
    window_prices = prices[-window:]
    return float(np.dot(window_prices, weights) / weights.sum())


def momentum(prices: np.ndarray, window: int) -> float:
    """P_t - P_{t-window}."""
    if len(prices) <= window:
        return NAN
    return float(prices[-1] - prices[-1 - window])


def roc(prices: np.ndarray, window: int) -> float:
    """Rate of change: (P_t - P_{t-window}) / P_{t-window} * 100."""
    if len(prices) <= window or prices[-1 - window] == 0:
        return NAN
    return float((prices[-1] - prices[-1 - window]) / prices[-1 - window] * 100.0)


def velocity(prices: np.ndarray, window: int) -> float:
    """Mean first difference over the trailing window — average tick-to-tick rate of change."""
    if len(prices) < window + 1:
        return NAN
    diffs = np.diff(prices[-(window + 1) :])
    return float(np.mean(diffs))


def acceleration(prices: np.ndarray, window: int) -> float:
    """Mean second difference over the trailing window — rate of change of velocity."""
    if len(prices) < window + 2:
        return NAN
    diffs = np.diff(prices[-(window + 2) :])
    second_diffs = np.diff(diffs)
    return float(np.mean(second_diffs))


def macd(prices: np.ndarray, fast: int, slow: int, signal: int) -> tuple[float, float, float]:
    """
    MACD line = EMA_fast - EMA_slow
    Signal line = EMA_signal(MACD line)
    Histogram = MACD line - Signal line

    Returns (macd_line, signal_line, histogram). NaN-filled if insufficient history.
    """
    min_len = slow + signal
    if len(prices) < min_len:
        return NAN, NAN, NAN

    # Build the MACD line as a series so we can EMA it for the signal line.
    macd_series = []
    for i in range(slow, len(prices) + 1):
        window = prices[:i]
        macd_series.append(ema(window, fast) - ema(window, slow))
    macd_series = np.array(macd_series)

    if len(macd_series) < signal:
        return NAN, NAN, NAN

    macd_line = float(macd_series[-1])
    signal_line = ema(macd_series, signal)
    histogram = macd_line - signal_line if not np.isnan(signal_line) else NAN
    return macd_line, signal_line, histogram


def rsi(prices: np.ndarray, window: int) -> float:
    """
    Wilder's RSI:
        RS = average_gain / average_loss  (Wilder-smoothed over `window` periods)
        RSI = 100 - 100 / (1 + RS)
    Returns 50.0 (neutral) in the degenerate case of zero average loss with
    zero average gain (flat price), 100.0 if there are gains and no losses.
    """
    if len(prices) < window + 1:
        return NAN
    deltas = np.diff(prices[-(window + 1) :])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_gain == 0 and avg_loss == 0:
        return 50.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, window: int) -> float:
    """
    Average True Range.
        TR_t = max(high_t - low_t, |high_t - close_{t-1}|, |low_t - close_{t-1}|)
        ATR = simple mean of TR over the trailing `window` periods.
    """
    if len(closes) < window + 1:
        return NAN
    h = highs[-(window + 1) :]
    l = lows[-(window + 1) :]
    c = closes[-(window + 1) :]
    prev_close = c[:-1]
    tr = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - prev_close), np.abs(l[1:] - prev_close)),
    )
    return float(np.mean(tr))


# --------------------------------------------------------------------- #
# Volatility / dispersion
# --------------------------------------------------------------------- #


def rolling_variance(prices: np.ndarray, window: int) -> float:
    if len(prices) < window:
        return NAN
    return float(np.var(prices[-window:], ddof=1)) if window > 1 else 0.0


def rolling_std(prices: np.ndarray, window: int) -> float:
    var = rolling_variance(prices, window)
    return float(np.sqrt(var)) if not np.isnan(var) else NAN


def zscore(prices: np.ndarray, window: int) -> float:
    """(P_t - rolling_mean) / rolling_std over the trailing window."""
    if len(prices) < window:
        return NAN
    windowed = prices[-window:]
    mean = np.mean(windowed)
    std = np.std(windowed, ddof=1) if window > 1 else 0.0
    if std == 0:
        return 0.0
    return float((prices[-1] - mean) / std)


# --------------------------------------------------------------------- #
# Statistical / distributional
# --------------------------------------------------------------------- #


def shannon_entropy(returns: np.ndarray, bins: int) -> float:
    """
    Discretize returns into `bins` equal-width buckets and compute
    Shannon entropy in bits:
        H = -sum(p_i * log2(p_i))  for p_i > 0
    Higher entropy = more disorder/randomness in the return distribution
    over the window; lower entropy = more structure/predictability.
    """
    if len(returns) < bins:
        return NAN
    hist, _ = np.histogram(returns, bins=bins)
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def skewness(returns: np.ndarray) -> float:
    """
    Sample skewness (third standardized moment):
        skew = mean((x - mean)^3) / std^3
    """
    if len(returns) < 3:
        return NAN
    mean = np.mean(returns)
    std = np.std(returns, ddof=0)
    if std == 0:
        return 0.0
    return float(np.mean((returns - mean) ** 3) / std**3)


def kurtosis(returns: np.ndarray) -> float:
    """
    Sample excess kurtosis (fourth standardized moment minus 3, so a
    normal distribution scores 0):
        kurt = mean((x - mean)^4) / std^4 - 3
    """
    if len(returns) < 4:
        return NAN
    mean = np.mean(returns)
    std = np.std(returns, ddof=0)
    if std == 0:
        return 0.0
    return float(np.mean((returns - mean) ** 4) / std**4 - 3.0)


def autocorrelation(returns: np.ndarray, lag: int) -> float:
    """
    Lag-k sample autocorrelation of returns:
        rho_k = sum((x_t - mean)(x_{t-k} - mean)) / sum((x_t - mean)^2)
    """
    if len(returns) <= lag:
        return NAN
    mean = np.mean(returns)
    centered = returns - mean
    numerator = np.sum(centered[lag:] * centered[:-lag])
    denominator = np.sum(centered**2)
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


# --------------------------------------------------------------------- #
# Fractal / complexity
# --------------------------------------------------------------------- #


def hurst_exponent(prices: np.ndarray, min_chunk_size: int = 8) -> float:
    """
    Hurst exponent via rescaled range (R/S) analysis on log returns.

    For each chunk size n (a set of divisors of the series length,
    logarithmically spaced), split the return series into chunks of that
    size and compute the average rescaled range R/S. Then fit:
        log(R/S) = H * log(n) + c
    via least squares — the slope H is the Hurst exponent.

    Interpretation: H ~ 0.5 = random walk (no memory), H > 0.5 =
    trending/persistent, H < 0.5 = mean-reverting/anti-persistent.
    """
    returns = log_returns(prices)
    n = len(returns)
    if n < min_chunk_size * 2:
        return NAN

    max_chunk = n // 2
    chunk_sizes = sorted(
        set(
            int(size)
            for size in np.unique(
                np.geomspace(min_chunk_size, max_chunk, num=10).astype(int)
            )
            if size >= min_chunk_size
        )
    )
    if len(chunk_sizes) < 2:
        return NAN

    log_n = []
    log_rs = []
    for size in chunk_sizes:
        n_chunks = n // size
        if n_chunks < 1:
            continue
        rs_values = []
        for i in range(n_chunks):
            chunk = returns[i * size : (i + 1) * size]
            mean = np.mean(chunk)
            deviations = np.cumsum(chunk - mean)
            r = np.max(deviations) - np.min(deviations)
            s = np.std(chunk, ddof=0)
            if s > 0:
                rs_values.append(r / s)
        if rs_values:
            log_n.append(np.log(size))
            log_rs.append(np.log(np.mean(rs_values)))

    if len(log_n) < 2:
        return NAN

    slope, _ = np.polyfit(log_n, log_rs, 1)
    return float(slope)


def higuchi_fractal_dimension(prices: np.ndarray, k_max: int = 10) -> float:
    """
    Higuchi's fractal dimension estimate.

    For each k in 1..k_max, build k subsequences of the series (offset by
    m = 0..k-1, step k), compute each subsequence's normalized curve
    length L(k), then average across the k subsequences to get L_avg(k).
    Fit:
        log(L_avg(k)) = -D * log(k) + c
    via least squares — the negated slope D is the fractal dimension
    (bounded in [1, 2]: closer to 1 = smoother/trending, closer to 2 =
    rougher/noisier).
    """
    n = len(prices)
    if n < k_max * 2:
        return NAN

    log_k = []
    log_l = []
    for k in range(1, k_max + 1):
        lengths = []
        for m in range(k):
            indices = np.arange(m, n, k)
            if len(indices) < 2:
                continue
            subsequence = prices[indices]
            length = np.sum(np.abs(np.diff(subsequence)))
            normalization = (n - 1) / (((n - m - 1) // k) * k) if ((n - m - 1) // k) > 0 else 0
            if normalization > 0:
                lengths.append(length * normalization / k)
        if lengths:
            log_k.append(np.log(1.0 / k))
            log_l.append(np.log(np.mean(lengths)))

    if len(log_k) < 2:
        return NAN

    slope, _ = np.polyfit(log_k, log_l, 1)
    return float(slope)
