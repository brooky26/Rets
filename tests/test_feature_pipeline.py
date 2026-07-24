import numpy as np
import pytest

from configs.feature_schema import (
    FeatureEngineeringConfig,
    FractalFeatureConfig,
    MomentumFeatureConfig,
    StatisticalFeatureConfig,
    VolatilityFeatureConfig,
)
from data.types import Candle
from features.pipeline import FeatureEngineeringPipeline, compute_feature_vector


def _small_config() -> FeatureEngineeringConfig:
    """A config with small windows so tests don't need thousands of candles."""
    return FeatureEngineeringConfig(
        momentum=MomentumFeatureConfig(
            sma_windows=[3, 5],
            ema_windows=[3, 5],
            wma_windows=[3],
            momentum_windows=[3],
            roc_windows=[3],
            macd_fast=3,
            macd_slow=6,
            macd_signal=2,
            rsi_window=5,
            velocity_window=3,
            acceleration_window=3,
        ),
        volatility=VolatilityFeatureConfig(
            atr_window=5, std_windows=[5], zscore_window=5
        ),
        statistical=StatisticalFeatureConfig(
            entropy_window=10,
            entropy_bins=4,
            skew_kurt_window=10,
            autocorrelation_window=10,
            autocorrelation_lags=[1, 2],
        ),
        fractal=FractalFeatureConfig(
            hurst_window=20, hurst_min_chunk_size=4, higuchi_window=20, higuchi_k_max=4
        ),
    )


def make_candles(symbol: str, n: int, start_epoch: int = 1000) -> list[Candle]:
    rng = np.random.default_rng(42)
    prices = 100 + np.cumsum(rng.normal(0, 0.3, n))
    candles = []
    for i, p in enumerate(prices):
        candles.append(
            Candle(
                symbol=symbol,
                epoch=start_epoch + i * 60,
                granularity=60,
                open=float(p),
                high=float(p + 0.5),
                low=float(p - 0.5),
                close=float(p),
            )
        )
    return candles


def test_returns_none_before_min_history():
    config = _small_config()
    pipeline = FeatureEngineeringPipeline(config)
    candles = make_candles("STPRNG100", n=config.min_history_required - 5)
    result = None
    for c in candles:
        result = pipeline.on_candle(c)
    assert result is None


def test_returns_complete_vector_once_enough_history():
    config = _small_config()
    pipeline = FeatureEngineeringPipeline(config)
    candles = make_candles("STPRNG100", n=config.min_history_required + 30)
    result = None
    for c in candles:
        result = pipeline.on_candle(c)
    assert result is not None
    assert result.is_complete
    assert result.symbol == "STPRNG100"


def test_separate_symbols_have_independent_buffers():
    config = _small_config()
    pipeline = FeatureEngineeringPipeline(config)
    candles_a = make_candles("STPRNG100", n=config.min_history_required + 5)
    candles_b = make_candles("STPRNG200", n=5)  # not enough history yet

    for c in candles_a:
        pipeline.on_candle(c)
    result_b = None
    for c in candles_b:
        result_b = pipeline.on_candle(c)

    assert pipeline.buffered_count("STPRNG100") > 0
    assert result_b is None  # STPRNG200 hasn't accumulated enough history


def test_streaming_matches_direct_batch_computation():
    """
    The whole point of the pure-function core: feeding candles one at a
    time through the streaming pipeline must produce the exact same
    feature vector as calling compute_feature_vector directly on the
    equivalent price arrays.
    """
    config = _small_config()
    pipeline = FeatureEngineeringPipeline(config)
    candles = make_candles("STPRNG100", n=config.min_history_required + 10)

    result = None
    for c in candles:
        result = pipeline.on_candle(c)
    assert result is not None

    # Direct batch computation over the full buffered history (matches
    # what the pipeline internally buffers: the last `buffer_size` candles,
    # which here is >= all candles fed in).
    closes = np.array([c.close for c in candles], dtype=np.float64)
    highs = np.array([c.high for c in candles], dtype=np.float64)
    lows = np.array([c.low for c in candles], dtype=np.float64)
    direct = compute_feature_vector(
        symbol="STPRNG100",
        epoch=candles[-1].epoch,
        closes=closes,
        highs=highs,
        lows=lows,
        config=config,
    )

    for key in result.values:
        a, b = result.values[key], direct.values[key]
        if np.isnan(a) or np.isnan(b):
            assert np.isnan(a) and np.isnan(b), f"mismatch on {key}: {a} vs {b}"
        else:
            assert a == pytest.approx(b), f"mismatch on {key}: {a} vs {b}"


def test_feature_vector_has_expected_keys():
    config = _small_config()
    pipeline = FeatureEngineeringPipeline(config)
    candles = make_candles("STPRNG100", n=config.min_history_required + 10)
    result = None
    for c in candles:
        result = pipeline.on_candle(c)

    expected_prefixes = [
        "sma_3", "sma_5", "ema_3", "ema_5", "wma_3", "momentum_3", "roc_3",
        "velocity", "acceleration", "macd_line", "macd_signal", "macd_histogram",
        "rsi", "atr", "std_5", "variance_5", "zscore", "entropy", "skewness",
        "kurtosis", "autocorrelation_lag_1", "autocorrelation_lag_2",
        "hurst_exponent", "fractal_dimension",
    ]
    for key in expected_prefixes:
        assert key in result.values, f"missing feature: {key}"
