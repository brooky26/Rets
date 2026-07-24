"""Config for Level 6 — Execution Decision."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ExecutionConfig(BaseModel):
    mode: str = Field(
        default="paper",
        description="paper|live. Defaults to paper — live mode additionally requires "
        "PlatformConfig.environment == 'live' (checked by ExecutionEngine at construction, "
        "not just here) as a second, independent safety rail against a single flag flip "
        "accidentally enabling real-money trading.",
    )
    max_payout_drift_pct: float = Field(
        default=0.15,
        description="If the live proposal's reward_to_risk differs from what the EV/Risk/"
        "Opportunity decision was based on by more than this fraction, abort rather than "
        "trade on a stale assumption — the market moved between scoring and execution.",
    )
    currency: str = Field(default="USD")
    price_slippage_tolerance_pct: float = Field(
        default=0.0,
        description="Accept a proposal ask_price up to this much higher than the stake "
        "used for sizing before aborting as stale. 0.0 = require an exact match.",
    )

    @field_validator("mode")
    @classmethod
    def _valid_mode(cls, v: str) -> str:
        allowed = {"paper", "live"}
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}")
        return v

    @field_validator("max_payout_drift_pct", "price_slippage_tolerance_pct")
    @classmethod
    def _nonnegative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("must be non-negative")
        return v
