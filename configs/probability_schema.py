"""Config for Level 2 — Probability Estimation."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from state_encoder.types import DIMENSION_NAMES


class BayesianLogisticConfig(BaseModel):
    feature_dims: list[str] = Field(
        default=["trend", "momentum", "acceleration", "volatility", "persistence", "compression_expansion"],
        description="Which MarketState dimensions form the regression's feature vector.",
    )
    prior_precision: float = Field(
        default=1.0,
        description="alpha in the Gaussian weight prior w ~ N(0, alpha^-1 I). Higher = stronger shrinkage toward 0.",
    )
    max_iterations: int = Field(default=50, description="Newton-Raphson (IRLS) iterations for MAP estimation.")
    tolerance: float = Field(default=1e-6, description="Stop IRLS when the weight update norm falls below this.")
    include_intercept: bool = True

    @field_validator("feature_dims")
    @classmethod
    def _dims_must_exist(cls, v: list[str]) -> list[str]:
        for dim in v:
            if dim not in DIMENSION_NAMES:
                raise ValueError(f"'{dim}' is not a valid MarketState dimension. Valid: {DIMENSION_NAMES}")
        if len(v) == 0:
            raise ValueError("feature_dims must not be empty")
        return v

    @field_validator("prior_precision")
    @classmethod
    def _positive_precision(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("prior_precision must be positive")
        return v


class BaggedGBMConfig(BaseModel):
    feature_dims: list[str] = Field(
        default=["trend", "momentum", "acceleration", "volatility", "noise", "persistence",
                  "compression_expansion", "complexity", "uncertainty", "market_phase"],
    )
    n_ensemble_members: int = Field(default=10, description="Number of bootstrap-resampled GBM models.")
    max_boosting_iterations: int = Field(default=100)
    learning_rate: float = Field(default=0.1)
    max_depth: int = Field(default=3)
    bootstrap_fraction: float = Field(default=0.8, description="Fraction of training data each ensemble member is resampled from.")
    random_seed: int = 42

    @field_validator("feature_dims")
    @classmethod
    def _dims_must_exist(cls, v: list[str]) -> list[str]:
        for dim in v:
            if dim not in DIMENSION_NAMES:
                raise ValueError(f"'{dim}' is not a valid MarketState dimension. Valid: {DIMENSION_NAMES}")
        if len(v) == 0:
            raise ValueError("feature_dims must not be empty")
        return v

    @field_validator("n_ensemble_members")
    @classmethod
    def _min_ensemble_size(cls, v: int) -> int:
        if v < 2:
            raise ValueError("n_ensemble_members must be >= 2 to estimate cross-member disagreement")
        return v


class CalibrationConfig(BaseModel):
    max_iterations: int = Field(default=100, description="Newton iterations for Platt scaling's 1D logistic fit.")
    tolerance: float = 1e-8


class ProbabilityEstimationConfig(BaseModel):
    bayesian_logistic: BayesianLogisticConfig = BayesianLogisticConfig()
    bagged_gbm: BaggedGBMConfig = BaggedGBMConfig()
    calibration: CalibrationConfig = CalibrationConfig()
