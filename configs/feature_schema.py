"""
Config additions for the Feature Engineering Pipeline. Appended to the
same PlatformConfig used by the Market Data Layer — nothing in configs
already built (market_data) is touched.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class MomentumFeatureConfig(BaseModel):
    sma_windows: list[int] = Field(default=[5, 10, 20, 50])
    ema_windows: list[int] = Field(default=[5, 10, 20, 50])
    wma_windows: list[int] = Field(default=[10, 20])
    momentum_windows: list[int] = Field(default=[5, 10, 20])
    roc_windows: list[int] = Field(default=[5, 10, 20])
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    rsi_window: int = 14
    velocity_window: int = 5
    acceleration_window: int = 5


class VolatilityFeatureConfig(BaseModel):
    atr_window: int = 14
    std_windows: list[int] = Field(default=[10, 20, 50])
    zscore_window: int = 20


class StatisticalFeatureConfig(BaseModel):
    entropy_window: int = 50
    entropy_bins: int = 10
    skew_kurt_window: int = 50
    autocorrelation_window: int = 50
    autocorrelation_lags: list[int] = Field(default=[1, 2, 5, 10])


class FractalFeatureConfig(BaseModel):
    hurst_window: int = 100
    hurst_min_chunk_size: int = 8
    higuchi_window: int = 100
    higuchi_k_max: int = 10


class FeatureEngineeringConfig(BaseModel):
    momentum: MomentumFeatureConfig = MomentumFeatureConfig()
    volatility: VolatilityFeatureConfig = VolatilityFeatureConfig()
    statistical: StatisticalFeatureConfig = StatisticalFeatureConfig()
    fractal: FractalFeatureConfig = FractalFeatureConfig()

    @property
    def min_history_required(self) -> int:
        """
        The largest window across every feature family — the pipeline
        needs at least this many candles buffered before it can emit a
        complete feature vector with no NaNs.
        """
        candidates = [
            max(self.momentum.sma_windows, default=0),
            max(self.momentum.ema_windows, default=0),
            max(self.momentum.wma_windows, default=0),
            max(self.momentum.momentum_windows, default=0),
            max(self.momentum.roc_windows, default=0),
            self.momentum.macd_slow + self.momentum.macd_signal,
            self.momentum.rsi_window + 1,
            self.volatility.atr_window + 1,
            max(self.volatility.std_windows, default=0),
            self.volatility.zscore_window,
            self.statistical.entropy_window,
            self.statistical.skew_kurt_window,
            self.statistical.autocorrelation_window
            + max(self.statistical.autocorrelation_lags, default=0),
            self.fractal.hurst_window,
            self.fractal.higuchi_window,
        ]
        return max(candidates) + 1  # +1 because most features need n+1 prices for n returns
