"""
Contract Outcome Tracker — Level 6 (live-mode follow-on).

The gap this closes: a live `buy()` gets a real `contract_id`, but Deriv
— not this platform — decides when and how that contract settles, on
its own real-tick clock, completely decoupled from candle cadence. Until
something asks Deriv what actually happened, the only outcome data this
platform has for a live trade is none at all.

This module is deliberately the *only* place that talks to
`BrokerClient.check_contract_status` and interprets the response. Every
other consumer (the orchestrator, Risk Engine, Post-Trade Analyzer) only
ever sees the small `ContractOutcome` shape below — never Deriv's raw
dict — so nothing downstream needs to know Deriv's field names, and this
is the one place a future change to Deriv's response shape needs fixing.

Polling, not push: `poll()` is a single one-shot check-status call that
returns `None` when the contract is still open. Callers are expected to
call it again on whatever cadence suits them (the orchestrator calls it
once per candle for the symbol with a trade still awaiting real
settlement) — nothing here blocks or loops, so a caller controls its own
timeout/backoff/give-up behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

from execution.types import BrokerClient


@dataclass(frozen=True, slots=True)
class ContractOutcome:
    """
    A settled (or still-open) contract's status, translated out of
    Deriv's raw response shape.

    `pnl` is Deriv's own `profit` field taken as-is: payout minus
    buy_price on a win, negative buy_price on a loss — this is exactly
    what `RiskEngine.record_trade_result` and `CompletedTrade.pnl`
    already expect, so no re-derivation of win/loss happens here or
    anywhere downstream. Deriv is the one authority on whether a live
    contract won or lost; this platform's job is to relay that number
    faithfully, not recompute it.
    """

    contract_id: str
    is_sold: bool
    pnl: float
    payout: float
    buy_price: float
    sell_price: float | None


class ContractOutcomeTracker:
    def __init__(self, broker_client: BrokerClient) -> None:
        self._broker_client = broker_client

    async def poll(self, contract_id: str) -> ContractOutcome:
        """
        Check a contract's current status right now. `is_sold=False`
        means the contract is still open (expected while it's within its
        real duration_ticks window) — the caller should treat that as
        "nothing changed yet," not as an error, and simply poll again
        later.
        """
        raw = await self._broker_client.check_contract_status(contract_id)
        is_sold = bool(raw.get("is_sold"))

        if not is_sold:
            return ContractOutcome(
                contract_id=contract_id,
                is_sold=False,
                pnl=0.0,
                payout=float(raw.get("payout", 0.0)),
                buy_price=float(raw.get("buy_price", 0.0)),
                sell_price=None,
            )

        return ContractOutcome(
            contract_id=contract_id,
            is_sold=True,
            pnl=float(raw["profit"]),
            payout=float(raw.get("payout", 0.0)),
            buy_price=float(raw.get("buy_price", 0.0)),
            sell_price=float(raw["sell_price"]) if raw.get("sell_price") is not None else None,
        )
