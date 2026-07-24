"""
Config for the Market State Encoder.

Each conceptual dimension (trend, momentum, volatility, ...) is built
from a weighted combination of specific feature-vector keys. Two
transform types:

  - "zscore": normalized via the OnlineNormalizer (unbounded raw scale,
    e.g. MACD line, ATR, entropy).
  - "affine": for features that already have a known, meaningful bounded
    range (e.g. Hurst exponent in ~[0,1], RSI in [0,100], fractal
    dimension in [1,2]) — mapped via (value - center) / scale and clipped
    to [-1, 1]. Avoids z-scoring something that's already interpretable
    on its own terms.

IMPORTANT: these default feature keys assume the default
FeatureEngineeringConfig windows (e.g. momentum_20, std_20, roc_20). If
those windows are changed, this mapping must be updated to match, or the
encoder will raise a clear KeyError rather than silently defaulting —
better a loud failure than a state vector quietly built from garbage.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class FeatureMapping(BaseModel):
    feature_key: str
    transform: Literal["zscore", "affine"] = "zscore"
    weight: float = 1.0
    center: float | None = None
    scale: float | None = None

    @model_validator(mode="after")
    def _affine_requires_center_scale(self) -> "FeatureMapping":
        if self.transform == "affine" and (self.center is None or self.scale is None):
            raise ValueError(
                f"feature '{self.feature_key}': affine transform requires center and scale"
            )
        if self.scale is not None and self.scale == 0:
            raise ValueError(f"feature '{self.feature_key}': scale must not be zero")
        return self


class StateEncoderConfig(BaseModel):
    trend: list[FeatureMapping] = Field(
        default=[
            FeatureMapping(feature_key="momentum_20", transform="zscore"),
            FeatureMapping(feature_key="roc_20", transform="zscore"),
            FeatureMapping(feature_key="macd_line", transform="zscore"),
        ]
    )
    momentum: list[FeatureMapping] = Field(
        default=[
            FeatureMapping(feature_key="rsi", transform="affine", center=50.0, scale=50.0),
            FeatureMapping(feature_key="roc_5", transform="zscore"),
            FeatureMapping(feature_key="velocity", transform="zscore"),
        ]
    )
    acceleration: list[FeatureMapping] = Field(
        default=[FeatureMapping(feature_key="acceleration", transform="zscore")]
    )
    volatility: list[FeatureMapping] = Field(
        default=[
            FeatureMapping(feature_key="atr", transform="zscore"),
            FeatureMapping(feature_key="std_20", transform="zscore"),
        ]
    )
    noise: list[FeatureMapping] = Field(
        default=[FeatureMapping(feature_key="entropy", transform="zscore")]
    )
    uncertainty: list[FeatureMapping] = Field(
        default=[
            FeatureMapping(feature_key="entropy", transform="zscore", weight=0.5),
            FeatureMapping(feature_key="std_20", transform="zscore", weight=0.5),
        ]
    )

    # Naturally-bounded features, mapped directly rather than via the list mechanism:
    hurst_feature_key: str = "hurst_exponent"
    fractal_feature_key: str = "fractal_dimension"

    # Compression/expansion: log-ratio of a short-window vol to a long-window vol.
    compression_short_std_key: str = "std_10"
    compression_long_std_key: str = "std_50"

    # Market phase blend weights (applied to the already-computed trend and
    # compression_expansion dimensions — NOT a substitute for Level 1 Regime
    # Detection, just a cheap continuous proxy available at this stage).
    market_phase_trend_weight: float = 0.6
    market_phase_compression_weight: float = 0.4
