"""
Config for MC/Hurst-informed duration selection (see monte_carlo/duration_selector.py
for the full design rationale).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class DurationSelectionConfig(BaseModel):
    candidate_durations_ticks: list[int] = Field(
        default=[3, 5, 7, 8, 10],
        description="Candidate contract durations (in ticks) evaluated every candle; the one "
        "maximizing risk-adjusted EV among EV-positive candidates is selected. Kept within "
        "Deriv's confirmed live limit of 1-10 ticks for these symbols (a real "
        "ContractBuyValidationError/TicksNumberLimits response, code_args: [1, 10]) — if you "
        "add symbols with a different limit, verify it before widening this.",
    )
    mc_max_standard_error: float = Field(
        default=0.05,
        description="A candidate duration's Monte Carlo prob_favorable estimate is trusted only "
        "when sqrt(p*(1-p)/n_paths) <= this. Above it, the Hurst-based analytical fallback is "
        "used for that candidate instead — deliberately simple (not a rigorous fat-tail-adjusted "
        "interval), same honesty standard as elsewhere in this codebase.",
    )
    mc_min_paths: int = Field(
        default=200,
        description="Below this many simulated paths, MC's standard-error estimate itself isn't "
        "trusted (a small n can produce a spuriously small SE at an extreme p) — falls back to "
        "the Hurst estimate regardless of the computed SE.",
    )
    hurst_reference_duration_ticks: int = Field(
        default=5,
        description="The duration the fused probability estimate is treated as implicitly "
        "'at' when rescaling it across candidate durations via the Hurst exponent — should "
        "roughly match whatever gap the probability models' training labels were defined over "
        "(next-candle direction, by default one tick/candle apart).",
    )
    ev_penalty_per_tick_beyond_reference: float = Field(
        default=0.0008,
        description="A deliberately simple correction for a real structural bias: the Hurst "
        "fallback's analytical_confidence(d) = 0.5 + (p-0.5)*(d/d_ref)^(H-0.5) is monotonically "
        "increasing in duration d whenever H > 0.5 (any persistent/trending regime) — and since "
        "payout is modeled as a FLAT ratio regardless of duration (no real-world 'longer "
        "contracts price in worse payout odds' cost), EV-maximization then mechanically favors "
        "the longest candidate in trending regimes, not because it's genuinely best but because "
        "nothing in the model penalizes waiting longer. This field subtracts "
        "`this * stake * max(0, duration_ticks - hurst_reference_duration_ticks)` from each "
        "candidate's expected_value before ranking — a heuristic time-cost, not a fitted one "
        "(no real per-duration payout curve is wired in yet). Set to 0.0 to disable and restore "
        "the old unpenalized behavior.",
    )

    @field_validator("candidate_durations_ticks")
    @classmethod
    def _valid_candidates(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("candidate_durations_ticks must not be empty")
        if any(d <= 0 for d in v):
            raise ValueError("all candidate_durations_ticks must be positive")
        return sorted(set(v))

    @field_validator("ev_penalty_per_tick_beyond_reference")
    @classmethod
    def _valid_penalty(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("ev_penalty_per_tick_beyond_reference must be >= 0")
        return v

    @field_validator("mc_max_standard_error")
    @classmethod
    def _valid_se(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("mc_max_standard_error must be in (0, 1)")
        return v

    @field_validator("mc_min_paths", "hurst_reference_duration_ticks")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be positive")
        return v
