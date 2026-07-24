"""Config for Level 7 — Trade Management (RL)."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class QLearningConfig(BaseModel):
    n_time_bins: int = Field(default=5, description="Discretization buckets for time_remaining_fraction.")
    n_return_bins: int = Field(default=7, description="Discretization buckets for unrealized_return.")
    n_trend_bins: int = Field(default=5, description="Discretization buckets for favorable_move_pct.")

    return_bin_edges: list[float] = Field(
        default=[-0.8, -0.4, -0.1, 0.1, 0.4, 0.8],
        description="Edges for unrealized_return bins (must have n_return_bins - 1 entries).",
    )
    trend_bin_edges: list[float] = Field(
        default=[-0.02, -0.005, 0.005, 0.02],
        description="Edges for favorable_move_pct bins (must have n_trend_bins - 1 entries).",
    )

    learning_rate: float = Field(default=0.1, description="alpha in the TD update.")
    discount_factor: float = Field(default=0.95, description="gamma in the TD update.")

    epsilon_start: float = Field(default=1.0, description="Initial exploration rate.")
    epsilon_end: float = Field(default=0.05, description="Floor exploration rate after decay.")
    epsilon_decay_episodes: int = Field(
        default=2000, description="Episodes over which epsilon linearly decays from start to end."
    )

    random_seed: int = 42

    @field_validator("n_time_bins", "n_return_bins", "n_trend_bins")
    @classmethod
    def _min_two_bins(cls, v: int) -> int:
        if v < 2:
            raise ValueError("must have at least 2 bins")
        return v

    @field_validator("learning_rate")
    @classmethod
    def _lr_in_bounds(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError("learning_rate must be in (0, 1]")
        return v

    @field_validator("discount_factor")
    @classmethod
    def _gamma_in_bounds(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("discount_factor must be in [0, 1]")
        return v

    @field_validator("epsilon_start", "epsilon_end")
    @classmethod
    def _epsilon_in_bounds(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("epsilon must be in [0, 1]")
        return v
