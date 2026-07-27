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
        description="Was a pre-trade abort check when execution used a separate fetch_proposal "
        "step. Now that the default live path is ExecutionEngine._execute_live's one-step "
        "buy_direct (see execution/engine.py's module docstring for why), there's no live quote "
        "to check BEFORE committing — this now drives a POST-hoc warning "
        "(_log_drift_warning_if_needed) comparing the actual fill's reward_to_risk against the "
        "decision basis, for review, not to abort an already-completed trade.",
    )
    currency: str = Field(default="USD")
    price_slippage_tolerance_pct: float = Field(
        default=0.0,
        description="Currently UNUSED — was a pre-trade check (accept a proposal ask_price up to "
        "this much higher than the sizing stake before aborting as stale) that only made sense "
        "against a separate fetch_proposal step. Kept in the schema rather than removed outright, "
        "in case a future execution path reintroduces a pre-trade quote check that needs it.",
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
