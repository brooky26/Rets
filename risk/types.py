"""Level 4 — Risk Assessment: shared types."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TradeOutcome:
    """A single completed trade, fed to RiskEngine.record_trade_result()."""

    epoch: int
    pnl: float             # signed: positive = win, negative = loss
    equity_after: float


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    """
    Output of the Risk Engine for one candidate trade. The Risk Engine
    has absolute veto authority: `approved` can be False even when the
    upstream EVEstimate was positive-EV — EV alone is necessary but never
    sufficient. `veto_reasons` is a list (not a single string) because
    multiple independent checks can fail simultaneously (e.g. both a
    drawdown breach and an excessive risk-of-ruin), and downstream
    explainability should be able to show all of them, not just the
    first one encountered.
    """

    symbol: str
    epoch: int
    approved: bool
    recommended_stake: float          # 0.0 whenever approved is False
    kelly_fraction_raw: float          # unclipped Kelly fraction
    kelly_fraction_applied: float      # after fractional-Kelly multiplier and exposure cap
    risk_of_ruin: float
    current_drawdown_pct: float
    daily_loss_pct: float
    consecutive_losses: int
    expected_shortfall_pct: float      # NaN if insufficient trade history
    veto_reasons: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.risk_of_ruin == self.risk_of_ruin  # False only for NaN
