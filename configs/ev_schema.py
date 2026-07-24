"""Config for Level 3 — Expected Value Estimation."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ExpectedValueConfig(BaseModel):
    min_ev_threshold: float = Field(
        default=0.0,
        description="Minimum expected_value (currency units) to pass the gate. "
        "0.0 enforces the spec's 'never execute negative EV trades' as a hard minimum; "
        "raise it above 0 to also require a margin of safety.",
    )
    min_reward_to_risk: float = Field(
        default=0.0,
        description="Minimum profit_if_win/stake ratio required, independent of the EV threshold.",
    )
    min_probability_confidence: float = Field(
        default=0.5,
        description="Minimum probability_used required to even consider a direction "
        "(0.5 = no additional filter beyond picking the favored side).",
    )

    @field_validator("min_reward_to_risk")
    @classmethod
    def _reward_to_risk_nonnegative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("min_reward_to_risk cannot be negative")
        return v

    @field_validator("min_probability_confidence")
    @classmethod
    def _confidence_in_bounds(cls, v: float) -> float:
        if not (0.5 <= v <= 1.0):
            raise ValueError("min_probability_confidence must be in [0.5, 1.0]")
        return v
