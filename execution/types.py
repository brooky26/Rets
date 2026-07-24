"""
Level 6 — Execution Decision: shared types.

`BrokerClient` is a narrow Protocol capturing only what the execution
engine and the outcome tracker actually need (fetch a proposal, buy it,
poll a bought contract's status) — `DerivWebSocketClient` satisfies it
structurally without any inheritance, and a `FakeBrokerClient`
implementing the same methods is what the test suite uses, so none of
the engine's decision logic (or the outcome tracker's polling logic)
needs a live socket to be fully tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class ExecutionMode(str, Enum):
    PAPER = "paper"   # simulate the decision, never call the broker's buy endpoint
    LIVE = "live"      # fetch a real proposal and actually buy


class BrokerClient(Protocol):
    async def fetch_proposal(
        self, symbol: str, contract_type_code: str, stake: float, duration_ticks: int, currency: str
    ) -> dict: ...

    async def buy(self, proposal_id: str, price: float) -> dict: ...

    async def check_contract_status(self, contract_id: str) -> dict: ...


@dataclass(frozen=True, slots=True)
class ExecutionDecision:
    """
    Output of the Execution Engine for one approved TradeOpportunity.

    `action` is "buy" (actually executed or, in paper mode, simulated as
    if executed), "skip" (upstream wasn't approved, or a safety check
    aborted it), or "error" (the broker call itself failed). `contract_id`
    is only ever populated for a real LIVE buy — paper-mode "buy" actions
    never have one, which is itself useful for auditing whether a given
    decision touched real money.
    """

    symbol: str
    epoch: int
    mode: ExecutionMode
    action: str  # "buy" | "skip" | "error"
    stake: float
    payout: float
    contract_id: str | None
    reason: str
