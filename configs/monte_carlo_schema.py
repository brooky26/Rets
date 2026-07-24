"""
Config for the Monte Carlo Price-Path Simulator (`monte_carlo/price_paths.py`).

Distinct from `configs.backtest_schema.MonteCarloStressConfig`, which
configures resampling of ALREADY-REALIZED trade P&L sequences. This
schema configures FORWARD GBM price simulation from the current price.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class MonteCarloPricePathConfig(BaseModel):
    n_paths: int = Field(default=1000, description="Number of simulated forward price paths.")
    horizon_ticks: int = Field(
        default=10, description="How many ticks forward to simulate — should match the contract's duration_ticks."
    )
    mu_estimation_window: int = Field(
        default=50, description="Number of recent log-returns used to estimate mu_per_tick/sigma_per_tick."
    )
    ev_blend_weight: float = Field(
        default=0.0,
        description="Weight given to the Monte-Carlo-implied probability when refining an EVEstimate via "
        "`monte_carlo.price_paths.refine_ev_with_monte_carlo` (0.0 = disabled, EV stays purely model-based).",
    )
    quality_score_mc_weight: float = Field(
        default=0.0,
        description="If > 0, opportunity/scorer.py adds an MC-confidence component to the quality score, "
        "weighted by this value. The corresponding QualityWeights.mc_confidence_weight must be set to "
        "match — kept as two separate fields (one here, one on QualityWeights) rather than a single shared "
        "field so this config can be used to derive/validate the other, without opportunity_schema importing "
        "from monte_carlo_schema.",
    )
    random_seed: int = 42

    @field_validator("n_paths")
    @classmethod
    def _min_paths(cls, v: int) -> int:
        if v < 100:
            raise ValueError("n_paths should be at least 100 for stable probability estimates")
        return v

    @field_validator("horizon_ticks", "mu_estimation_window")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be positive")
        return v

    @field_validator("ev_blend_weight", "quality_score_mc_weight")
    @classmethod
    def _weight_bounds(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("must be in [0, 1]")
        return v
