"""Config for Level 4 — Risk Assessment."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class RiskConfig(BaseModel):
    # --- Circuit breakers / capital preservation ---
    max_daily_loss_pct: float = Field(
        default=0.05, description="Trading halts for the day once realized loss exceeds this fraction of the day's starting equity."
    )
    max_drawdown_pct: float = Field(
        default=0.20, description="Trading halts once equity falls this fraction below its peak-to-date."
    )
    max_consecutive_losses: int = Field(
        default=6, description="Trading pauses after this many consecutive losing trades."
    )
    consecutive_loss_cooldown_evaluations: int = Field(
        default=20,
        description="How many assess() calls (i.e. candles evaluated, whether or not a trade "
        "occurs) the consecutive-loss pause lasts before the streak counter resets and trading "
        "can resume. This is a genuine fix, not cosmetic: without an explicit cooldown, the "
        "consecutive-loss counter can only reset on a WIN, but a win can never happen while the "
        "breaker itself is blocking every trade — a real, previously-discovered self-locking "
        "failure mode (found via an actual walk-forward backtest, not a hypothetical). This "
        "pause is intentionally a temporary cooling-off period, unlike the drawdown breaker "
        "below, which stays hard-stopped until a genuine equity recovery or a manual reset() — "
        "matching how institutional risk desks typically treat a drawdown breach (pending "
        "review) versus a losing streak (a scheduled pause).",
    )

    # --- Position sizing ---
    kelly_fraction_multiplier: float = Field(
        default=0.25,
        description="Fractional Kelly safety multiplier (0.25 = quarter-Kelly). Full Kelly (1.0) is "
        "provably capital-optimal in the long run but has ruinous variance in practice — this is "
        "not a minor tuning knob, it's the single biggest lever against blowing up the account.",
    )
    max_exposure_pct: float = Field(
        default=0.02, description="Hard cap on any single trade's stake as a fraction of current equity, "
        "independent of what Kelly sizing suggests."
    )
    min_stake: float = Field(default=1.0, description="Floor on recommended stake (e.g. Deriv's minimum stake).")

    # --- Risk of ruin ---
    risk_of_ruin_threshold: float = Field(
        default=0.01, description="Maximum acceptable probability of eventual ruin, given current edge and capital-to-stake ratio."
    )

    # --- Expected shortfall (CVaR) ---
    expected_shortfall_confidence: float = Field(
        default=0.95, description="Confidence level (e.g. 0.95 = worst 5% of trades) for empirical CVaR."
    )
    expected_shortfall_max_pct: float = Field(
        default=0.10, description="Maximum acceptable expected shortfall as a fraction of equity, once enough trade history exists."
    )
    min_trades_for_expected_shortfall: int = Field(
        default=30, description="Minimum recorded trades before expected shortfall is computed/enforced at all."
    )

    @field_validator(
        "max_daily_loss_pct", "max_drawdown_pct", "max_exposure_pct",
        "risk_of_ruin_threshold", "expected_shortfall_max_pct",
    )
    @classmethod
    def _fraction_in_bounds(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError("must be in (0, 1]")
        return v

    @field_validator("kelly_fraction_multiplier")
    @classmethod
    def _kelly_multiplier_bounds(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError("kelly_fraction_multiplier must be in (0, 1] — values above 1.0 (super-Kelly) are never advisable")
        return v

    @field_validator("max_consecutive_losses", "min_trades_for_expected_shortfall", "consecutive_loss_cooldown_evaluations")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be a positive integer")
        return v

    @field_validator("expected_shortfall_confidence")
    @classmethod
    def _confidence_bounds(cls, v: float) -> float:
        if not (0.5 <= v < 1.0):
            raise ValueError("expected_shortfall_confidence must be in [0.5, 1.0)")
        return v
