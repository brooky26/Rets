"""Config for the Backtest Engine — Monte Carlo stress testing and walk-forward analysis."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class MonteCarloStressConfig(BaseModel):
    n_paths: int = Field(default=2000, description="Number of bootstrap-resampled equity paths to simulate.")
    block_size: int = Field(
        default=5,
        description="Circular block bootstrap block length. 1 = ordinary i.i.d. bootstrap "
        "(matches the independence assumption behind the analytical risk-of-ruin formula in "
        "risk/ruin.py — useful for cross-validating that formula). Larger values preserve "
        "short-range autocorrelation in the trade sequence at the cost of a coarser resampling grid.",
    )
    starting_capital: float = Field(default=1000.0, description="Assumed starting account equity for each simulated path.")
    random_seed: int = 42

    @field_validator("n_paths")
    @classmethod
    def _min_paths(cls, v: int) -> int:
        if v < 100:
            raise ValueError("n_paths should be at least 100 for stable percentile estimates")
        return v

    @field_validator("block_size")
    @classmethod
    def _min_block_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError("block_size must be >= 1")
        return v

    @field_validator("starting_capital")
    @classmethod
    def _positive_capital(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("starting_capital must be positive")
        return v


class WalkForwardConfig(BaseModel):
    train_window_trades: int = Field(default=500, description="Number of observations used to fit the probability model per window.")
    test_window_trades: int = Field(default=100, description="Number of observations evaluated out-of-sample per window.")
    step_trades: int = Field(
        default=100, description="How far the window advances each step. Equal to test_window_trades = "
        "non-overlapping test windows (the standard walk-forward setup)."
    )

    @field_validator("train_window_trades", "test_window_trades", "step_trades")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be positive")
        return v
