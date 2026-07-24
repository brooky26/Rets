"""
Level 3 — Expected Value Estimation: shared types.

Unlike every prior stage, this one needs no training and no fitted
parameters — expected value is deterministic arithmetic once you have a
probability and a contract's payout structure. The "model" here is a
formula, not a learned function.

Deriv payout convention: a contract has a `stake` (amount risked) and a
`payout` (total amount received back if the contract wins, stake
included). Net profit on a win is `payout - stake`; a loss forfeits the
full `stake`. This is the standard convention Deriv's own proposal API
returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ContractType(str, Enum):
    RISE_FALL = "rise_fall"
    HIGHER_LOWER = "higher_lower"
    TOUCH_NO_TOUCH = "touch_no_touch"
    IN_OUT = "in_out"  # ends-between / stays-between style contracts


@dataclass(frozen=True, slots=True)
class ContractSpec:
    """
    The economic terms of a specific contract being evaluated — typically
    sourced from a live Deriv proposal (Level 6 / execution will fetch
    these for real; this stage just consumes whatever numbers it's given,
    live or hypothetical).
    """

    contract_type: ContractType
    stake: float
    payout: float  # total returned on a win, stake included
    duration_ticks: int

    def __post_init__(self) -> None:
        if self.stake <= 0:
            raise ValueError("stake must be positive")
        if self.payout < 0:
            raise ValueError("payout cannot be negative")
        if self.duration_ticks <= 0:
            raise ValueError("duration_ticks must be positive")

    @property
    def profit_if_win(self) -> float:
        return self.payout - self.stake

    @property
    def loss_if_lose(self) -> float:
        return -self.stake


@dataclass(frozen=True, slots=True)
class EVEstimate:
    """
    Output of the Expected Value Engine for one candidate trade.

    `probability_used` is whichever side of the ProbabilityEstimate
    corresponds to the contract's direction (prob_up if betting on a
    rise, prob_down if betting on a fall) — EV is meaningless without
    pinning down which outcome "win" refers to.

    `is_positive_ev` is the hard gate the spec requires ("never execute
    negative EV trades") — later stages check this flag rather than
    re-deriving it, and a False here always comes with `rejection_reason`
    populated so downstream explainability has something to say.
    """

    symbol: str
    epoch: int
    direction: int  # +1 (betting up/rise) or -1 (betting down/fall)
    probability_used: float
    stake: float
    payout: float
    expected_value: float          # currency units: p*payout - stake
    expected_value_pct: float      # expected_value / stake
    reward_to_risk: float          # profit_if_win / stake
    win_component: float           # p * profit_if_win
    loss_component: float          # (1-p) * loss_if_lose  (negative)
    outcome_std: float             # sqrt(p(1-p)) * (profit_if_win - loss_if_lose)
    risk_adjusted_score: float     # expected_value / outcome_std (Sharpe-style, single bet)
    is_positive_ev: bool
    rejection_reason: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.expected_value == self.expected_value  # False only for NaN
