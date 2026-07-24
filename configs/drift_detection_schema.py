"""Config for Drift Detection."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from state_encoder.types import DIMENSION_NAMES


class PSIConfig(BaseModel):
    n_bins: int = Field(default=10, description="Quantile bins built from the reference distribution.")
    warning_threshold: float = Field(
        default=0.1, description="PSI at or above this is a WARNING-severity alert (industry-standard convention)."
    )
    critical_threshold: float = Field(
        default=0.25, description="PSI at or above this is a CRITICAL-severity alert (industry-standard convention)."
    )

    @field_validator("n_bins")
    @classmethod
    def _min_bins(cls, v: int) -> int:
        if v < 2:
            raise ValueError("n_bins must be >= 2")
        return v

    @field_validator("warning_threshold", "critical_threshold")
    @classmethod
    def _nonnegative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("must be non-negative")
        return v


class DriftBootstrapConfig(BaseModel):
    n_bootstrap_resamples: int = Field(default=1000)
    confidence_level: float = Field(default=0.95)
    random_seed: int = 42

    @field_validator("n_bootstrap_resamples")
    @classmethod
    def _min_resamples(cls, v: int) -> int:
        if v < 100:
            raise ValueError("n_bootstrap_resamples should be at least 100")
        return v

    @field_validator("confidence_level")
    @classmethod
    def _confidence_bounds(cls, v: float) -> float:
        if not (0.5 <= v < 1.0):
            raise ValueError("confidence_level must be in [0.5, 1.0)")
        return v


class DriftDetectionConfig(BaseModel):
    feature_dims: list[str] = Field(
        default=["trend", "volatility", "persistence", "compression_expansion", "noise"],
        description="Which MarketState dimensions to monitor for feature drift.",
    )
    psi: PSIConfig = PSIConfig()
    bootstrap: DriftBootstrapConfig = DriftBootstrapConfig()
    min_samples_required: int = Field(
        default=100, description="Minimum observations in BOTH reference and current windows before any check runs."
    )

    @field_validator("feature_dims")
    @classmethod
    def _dims_must_exist(cls, v: list[str]) -> list[str]:
        for dim in v:
            if dim not in DIMENSION_NAMES:
                raise ValueError(f"'{dim}' is not a valid MarketState dimension. Valid: {DIMENSION_NAMES}")
        if len(v) == 0:
            raise ValueError("feature_dims must not be empty")
        return v

    @field_validator("min_samples_required")
    @classmethod
    def _min_samples_positive(cls, v: int) -> int:
        if v < 10:
            raise ValueError("min_samples_required should be at least 10")
        return v
