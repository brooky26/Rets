"""Backtest Engine — shared types."""

from __future__ import annotations

from dataclasses import dataclass

from post_trade.types import PerformanceMetrics


@dataclass(frozen=True, slots=True)
class MonteCarloStressResult:
    """
    Distribution of outcomes across `n_paths` block-bootstrap-resampled
    replays of a realized trade sequence — answers "how much of what
    actually happened was luck, and how bad could it plausibly have
    been given the same underlying edge?"

    `probability_of_ruin` here is EMPIRICAL (fraction of simulated paths
    whose dollar equity touched zero or below at any point) — distinct
    from `risk.ruin.risk_of_ruin()`'s ANALYTICAL result, which assumes
    i.i.d. bets and an infinite time horizon. The two should agree
    closely when `block_size=1` (matching the i.i.d. assumption) and can
    diverge at larger block sizes, which is itself informative: a large
    gap indicates the trade sequence has meaningful autocorrelation the
    analytical formula doesn't account for.
    """

    n_paths: int
    block_size: int
    starting_capital: float

    final_equity_mean: float
    final_equity_median: float
    final_equity_p5: float
    final_equity_p95: float

    max_drawdown_pct_mean: float
    max_drawdown_pct_median: float
    max_drawdown_pct_p95: float

    probability_of_ruin: float


@dataclass(frozen=True, slots=True)
class WalkForwardWindowResult:
    window_index: int
    train_start_index: int
    train_end_index: int
    test_start_index: int
    test_end_index: int
    metrics: PerformanceMetrics


@dataclass(frozen=True, slots=True)
class WalkForwardReport:
    windows: tuple[WalkForwardWindowResult, ...]
    aggregate_metrics: PerformanceMetrics

    @property
    def n_windows(self) -> int:
        return len(self.windows)
