"""Config for the Champion-Challenger Comparator."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ChampionChallengerConfig(BaseModel):
    min_trades_required: int = Field(
        default=30, description="Minimum trades in BOTH the champion's and challenger's evaluation "
        "samples before a comparison is even attempted — below this, sample noise dominates."
    )
    confidence_level: float = Field(
        default=0.95, description="Confidence level for the bootstrap comparison. The challenger's "
        "improvement must be positive across this much of the bootstrap distribution to promote."
    )
    min_improvement_threshold: float = Field(
        default=0.0, description="Minimum required improvement in the comparison metric (e.g. mean "
        "return_pct) — 0.0 means 'any statistically significant improvement,' raise it to require "
        "a meaningful margin of safety on top of significance."
    )
    n_bootstrap_resamples: int = Field(default=2000, description="Bootstrap resamples for the significance test.")
    random_seed: int = 42

    @field_validator("min_trades_required")
    @classmethod
    def _min_trades_positive(cls, v: int) -> int:
        if v < 2:
            raise ValueError("min_trades_required must be at least 2")
        return v

    @field_validator("confidence_level")
    @classmethod
    def _confidence_bounds(cls, v: float) -> float:
        if not (0.5 <= v < 1.0):
            raise ValueError("confidence_level must be in [0.5, 1.0)")
        return v

    @field_validator("n_bootstrap_resamples")
    @classmethod
    def _min_resamples(cls, v: int) -> int:
        if v < 100:
            raise ValueError("n_bootstrap_resamples should be at least 100 for a stable estimate")
        return v
