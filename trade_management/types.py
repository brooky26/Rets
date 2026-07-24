"""
Level 7 — Trade Management (RL): shared types.

Scope, stated honestly up front: the spec's generic action space (Trade,
Wait, Hold, Exit, Reduce Exposure, Increase Exposure, Scale Out, Scale
In) assumes a continuously-adjustable position, which is how most RL
trading literature frames the problem. A single Deriv Rise/Fall contract
is not that kind of position — once bought, there is nothing to "scale,"
only two things you can actually do before it expires:

    HOLD — let it run to expiry, settling at the fixed payout or losing
           the stake, whichever the underlying did.
    SELL — exit early at Deriv's current bid_price for the contract
           (available for most contract types via Deriv's `sell` API,
           subject to `is_valid_to_sell`), locking in whatever value it
           currently has rather than waiting for expiry.

"Increase/Reduce Exposure" and "Scale In/Out" would only apply if the
platform were buying additional contracts on top of an open position —
that's an Execution-layer decision (Level 6, already built) about
whether to open a *new* trade, not a Trade-Management decision about an
*existing* one, so those actions are deliberately out of scope here
rather than faked.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TradeManagementAction(str, Enum):
    HOLD = "hold"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class OpenContractState:
    """
    The state of one currently-open contract, as the RL agent sees it at
    one decision point (Deriv contracts are typically evaluated once per
    tick while open, via `proposal_open_contract` subscriptions).

    All fields are continuous / raw — discretization for the tabular
    agent happens in `discretizer.py`, kept separate so the state
    representation itself stays interpretable and reusable if a
    different (e.g. function-approximation) agent is added later.
    """

    symbol: str
    epoch: int
    ticks_remaining: int
    ticks_total: int
    stake: float
    current_bid_price: float
    entry_spot: float
    current_spot: float
    direction: int

    @property
    def time_remaining_fraction(self) -> float:
        if self.ticks_total <= 0:
            return 0.0
        return max(0.0, min(1.0, self.ticks_remaining / self.ticks_total))

    @property
    def unrealized_return(self) -> float:
        if self.stake <= 0:
            return 0.0
        return (self.current_bid_price - self.stake) / self.stake

    @property
    def favorable_move_pct(self) -> float:
        if self.entry_spot == 0:
            return 0.0
        raw_move = (self.current_spot - self.entry_spot) / self.entry_spot
        return raw_move * self.direction


@dataclass(frozen=True, slots=True)
class TradeManagementDecision:
    symbol: str
    epoch: int
    action: TradeManagementAction
    q_hold: float
    q_sell: float
    epsilon_used: float
    explored: bool
