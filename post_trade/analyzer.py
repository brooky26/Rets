"""
Post-Trade Analyzer — Level 8.

Stateful: accumulates `CompletedTrade` records and computes
`PerformanceMetrics` over the full history or the most recent rolling
window. Every formula is documented here rather than assumed —
particularly the drawdown-related ones, which have more than one
reasonable convention and it matters which one is being used.

Two different drawdown bases, deliberately
----------------------------------------------
`max_drawdown_pct` (used for Calmar) is computed on a *compounded,
normalized* equity curve — starting from 1.0 and multiplying by
(1 + return_pct) each trade — because Calmar-style ratios are meant to
compare a percentage return to a percentage drawdown, and compounding is
the only way percentage returns combine correctly over a sequence.

`recovery_factor`'s drawdown is computed on the *raw currency* P&L
cumulative sum instead — net profit divided by max dollar drawdown is
itself a currency-based ratio, and mixing in the compounding assumption
there would conflate two different questions ("how much did I make" vs
"how much did my equity ever recover from its worst point").

A structural caveat worth knowing before reading too much into Calmar
-------------------------------------------------------------------------
For a binary/digital option, a LOSING trade's return_pct is always
exactly -1.0 (the entire stake is forfeited) — that's just what losing
one of these contracts means, not a data quality issue. Compounding a
1.0-normalized equity curve at "this trade's return_pct IS the fraction
of the curve's current value being risked" therefore drives the curve to
exactly 0 the moment ANY single loss occurs, capping max_drawdown_pct at
100% and calmar_ratio at a meaningless negative number for basically any
real trade history with at least one loss in it. This isn't a bug — the
formula is doing exactly what compounding means — but it does mean
max_drawdown_pct/calmar_ratio here answer a narrower, more theoretical
question ("what if each trade risked its own full return against a
dedicated $1 bankroll") than the operationally meaningful portfolio
drawdown, which is what `RiskEngine.current_drawdown_pct` already tracks
against real account equity (Stage 7) and remains the metric to actually
monitor risk against. Sharpe/Sortino/profit_factor/expectancy don't have
this degeneracy and stay informative regardless.

No annualization
-------------------
Deriv synthetic-index tick contracts don't sit on a trading calendar the
way daily-bar equities do — there's no clean "trades per year" constant
to multiply a per-trade Sharpe by. All ratio metrics here are reported
per-trade. Pretending to annualize with an arbitrary assumed trade
frequency would manufacture false precision, not add information.

Probability calibration
--------------------------
Brier score:
    BS = mean( (predicted_probability_i - outcome_i)^2 )
    0 = perfect, 0.25 = a coin flip's worth of calibration on a symmetric problem.

Expected Calibration Error (ECE), via equal-width binning of predicted
probability into `n_calibration_bins` buckets over [0, 1]:
    ECE = sum_over_nonempty_bins( (n_bin / N) * |avg_predicted_bin - realized_win_rate_bin| )
"""

from __future__ import annotations

import numpy as np

from configs.post_trade_schema import PostTradeAnalysisConfig
from post_trade.types import CalibrationBin, CompletedTrade, PerformanceMetrics


class PostTradeAnalyzer:
    def __init__(self, config: PostTradeAnalysisConfig) -> None:
        self._config = config
        self._trades: list[CompletedTrade] = []

    def record_trade(self, trade: CompletedTrade) -> None:
        self._trades.append(trade)

    @property
    def n_trades_recorded(self) -> int:
        return len(self._trades)

    def all_trades(self) -> tuple[CompletedTrade, ...]:
        return tuple(self._trades)

    def compute_metrics(self, window: int | None = None) -> PerformanceMetrics:
        w = window if window is not None else self._config.rolling_window_trades
        trades = self._trades[-w:] if w < len(self._trades) else self._trades

        n = len(trades)
        if n == 0:
            return self._empty_metrics()

        pnls = np.array([t.pnl for t in trades])
        returns = np.array([t.return_pct for t in trades])
        wins = np.array([t.was_win for t in trades])
        predicted_probs = np.array([t.predicted_probability for t in trades])

        win_rate = float(np.mean(wins))

        gross_wins = float(np.sum(pnls[pnls > 0]))
        gross_losses = float(-np.sum(pnls[pnls < 0]))
        if gross_losses > 0:
            profit_factor = gross_wins / gross_losses
        else:
            profit_factor = float("inf") if gross_wins > 0 else 0.0

        expectancy = float(np.mean(pnls))
        average_return_pct = float(np.mean(returns))

        return_std = float(np.std(returns, ddof=1)) if n > 1 else 0.0
        sharpe_ratio = average_return_pct / return_std if return_std > 0 else 0.0

        downside = np.minimum(returns, 0.0)
        downside_deviation = float(np.sqrt(np.mean(downside**2))) if n > 0 else 0.0
        sortino_ratio = average_return_pct / downside_deviation if downside_deviation > 0 else 0.0

        equity_curve = np.cumprod(1.0 + returns)
        equity_curve_with_start = np.concatenate([[1.0], equity_curve])
        running_peak_pct = np.maximum.accumulate(equity_curve_with_start)
        drawdown_pct_series = (running_peak_pct - equity_curve_with_start) / running_peak_pct
        max_drawdown_pct = float(np.max(drawdown_pct_series))
        total_return_pct = float(equity_curve[-1] - 1.0)
        if max_drawdown_pct > 0:
            calmar_ratio = total_return_pct / max_drawdown_pct
        else:
            calmar_ratio = float("inf") if total_return_pct > 0 else 0.0

        cumulative_pnl = np.cumsum(pnls)
        cumulative_pnl_with_start = np.concatenate([[0.0], cumulative_pnl])
        running_peak_dollar = np.maximum.accumulate(cumulative_pnl_with_start)
        dollar_drawdown_series = running_peak_dollar - cumulative_pnl_with_start
        max_dollar_drawdown = float(np.max(dollar_drawdown_series))
        total_net_pnl = float(cumulative_pnl[-1])
        if max_dollar_drawdown > 0:
            recovery_factor = total_net_pnl / max_dollar_drawdown
        else:
            recovery_factor = float("inf") if total_net_pnl > 0 else 0.0

        max_consecutive_losses = self._max_consecutive_losses(wins)

        outcomes = wins.astype(float)
        brier_score = float(np.mean((predicted_probs - outcomes) ** 2))

        calibration_bins = self._calibration_bins(predicted_probs, outcomes)
        ece = self._expected_calibration_error(calibration_bins, n)

        return PerformanceMetrics(
            n_trades=n,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            average_return_pct=average_return_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown_pct=max_drawdown_pct,
            calmar_ratio=calmar_ratio,
            max_consecutive_losses=max_consecutive_losses,
            recovery_factor=recovery_factor,
            brier_score=brier_score,
            expected_calibration_error=ece,
            calibration_bins=calibration_bins,
        )

    @staticmethod
    def _max_consecutive_losses(wins: np.ndarray) -> int:
        max_streak = 0
        current_streak = 0
        for won in wins:
            if not won:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    def _calibration_bins(
        self, predicted_probs: np.ndarray, outcomes: np.ndarray
    ) -> tuple[CalibrationBin, ...]:
        n_bins = self._config.n_calibration_bins
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bins = []
        for i in range(n_bins):
            lower, upper = edges[i], edges[i + 1]
            if i == n_bins - 1:
                mask = (predicted_probs >= lower) & (predicted_probs <= upper)
            else:
                mask = (predicted_probs >= lower) & (predicted_probs < upper)
            n_in_bin = int(np.sum(mask))
            if n_in_bin == 0:
                continue
            avg_pred = float(np.mean(predicted_probs[mask]))
            realized = float(np.mean(outcomes[mask]))
            bins.append(
                CalibrationBin(
                    bin_lower=float(lower), bin_upper=float(upper), n_trades=n_in_bin,
                    avg_predicted_probability=avg_pred, realized_win_rate=realized,
                )
            )
        return tuple(bins)

    @staticmethod
    def _expected_calibration_error(bins: tuple[CalibrationBin, ...], n_total: int) -> float:
        if n_total == 0:
            return float("nan")
        ece = 0.0
        for b in bins:
            ece += (b.n_trades / n_total) * abs(b.avg_predicted_probability - b.realized_win_rate)
        return float(ece)

    @staticmethod
    def _empty_metrics() -> PerformanceMetrics:
        nan = float("nan")
        return PerformanceMetrics(
            n_trades=0, win_rate=nan, profit_factor=nan, expectancy=nan, average_return_pct=nan,
            sharpe_ratio=nan, sortino_ratio=nan, max_drawdown_pct=nan, calmar_ratio=nan,
            max_consecutive_losses=0, recovery_factor=nan, brier_score=nan,
            expected_calibration_error=nan, calibration_bins=(),
        )
