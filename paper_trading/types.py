"""Live paper-trading pipeline: shared types."""

from __future__ import annotations

from dataclasses import dataclass

from opportunity.types import TradeOpportunity
from regime.types import RegimeLabel


@dataclass(frozen=True, slots=True)
class PendingTrade:
    """
    One open paper position for a symbol, awaiting settlement.

    Settlement is next-candle-close direction (see paper_trading_schema's
    module docstring for why) ONLY when `contract_id` is None — this is
    created when a candle's decision chain approves a paper trade, and
    settled when that symbol's *next* candle completes, using entry_close
    vs the new candle's close.

    When `contract_id` is not None, this trade was a real live buy — the
    fictional next-candle rule above never applies to it. It stays
    pending until `ContractOutcomeTracker` reports the real Deriv
    settlement for that `contract_id` (see
    `PaperTradingOrchestrator._poll_live_settlement_if_any`), however many
    candles that takes, and no new trade is opened on that symbol in the
    meantime — one open position per symbol at a time.
    """

    symbol: str
    entry_epoch: int
    entry_close: float
    direction: int  # +1 (bet up) or -1 (bet down)
    stake: float
    payout: float
    predicted_probability: float
    regime_at_entry: RegimeLabel
    quality_score_at_entry: float
    contract_id: str | None = None

    @property
    def is_awaiting_real_settlement(self) -> bool:
        """True for a real live buy still awaiting Deriv's own settlement;
        False for a paper trade, which settles fictionally on next candle."""
        return self.contract_id is not None


def opportunity_to_pending_trade(
    opportunity: TradeOpportunity,
    entry_close: float,
    direction: int,
    stake: float,
    payout: float,
    probability_used: float,
    contract_id: str | None = None,
) -> PendingTrade:
    return PendingTrade(
        symbol=opportunity.symbol,
        entry_epoch=opportunity.epoch,
        entry_close=entry_close,
        direction=direction,
        stake=stake,
        payout=payout,
        predicted_probability=probability_used,
        regime_at_entry=opportunity.regime,
        quality_score_at_entry=opportunity.quality_score,
        contract_id=contract_id,
    )
