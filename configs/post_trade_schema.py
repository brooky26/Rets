"""Config for Level 8 — Post-Trade Analysis."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PostTradeAnalysisConfig(BaseModel):
    n_calibration_bins: int = Field(
        default=10, description="Number of equal-width probability buckets for the reliability diagram / ECE."
    )
    rolling_window_trades: int = Field(
        default=200, description="Number of most-recent trades PerformanceMetrics is computed over by default."
    )

    @field_validator("n_calibration_bins")
    @classmethod
    def _min_bins(cls, v: int) -> int:
        if v < 2:
            raise ValueError("n_calibration_bins must be >= 2")
        return v

    @field_validator("rolling_window_trades")
    @classmethod
    def _min_window(cls, v: int) -> int:
        if v < 1:
            raise ValueError("rolling_window_trades must be >= 1")
        return v
