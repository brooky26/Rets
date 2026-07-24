"""Level 8 — Post-Trade Analysis: shared types."""

from __future__ import annotations

from dataclasses import dataclass

from regime.types import RegimeLabel


@dataclass(frozen=True, slots=True)
class CompletedTrade:
    """
    The full record of one finished trade — the shape everything in this
    stage, and eventually the Continuous Learning Pipeline, consumes.
    Deliberately carries the *predicted* probability alongside the
    *realized* outcome, since that pairing is what calibration analysis
    needs and no earlier stage keeps both together in one place.
    """

    symbol: str
    entry_epoch: int
    exit_epoch: int
    direction: int
    stake: float
    pnl: float
    predicted_probability: float
    regime_at_entry: RegimeLabel
    quality_score_at_entry: float
    exit_reason: str

    @property
    def was_win(self) -> bool:
        return self.pnl > 0

    @property
    def return_pct(self) -> float:
        return self.pnl / self.stake if self.stake > 0 else 0.0


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    bin_lower: float
    bin_upper: float
    n_trades: int
    avg_predicted_probability: float
    realized_win_rate: float


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """
    Standard backtest-style performance metrics. All ratio metrics are
    computed *per-trade*, not annualized — Deriv synthetic-index tick
    contracts don't have a natural trading calendar to annualize
    against, so a pretend-annualized Sharpe would be more misleading
    than informative. Documented explicitly here and in analyzer.py.
    """

    n_trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    average_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    max_consecutive_losses: int
    recovery_factor: float
    brier_score: float
    expected_calibration_error: float
    calibration_bins: tuple[CalibrationBin, ...]
