"""
Monte Carlo Stress Testing — circular block bootstrap.

Why block bootstrap, not ordinary (i.i.d.) bootstrap
--------------------------------------------------------
A realized trade P&L sequence isn't i.i.d. in general — regime
persistence, streaks from the RL trade-management agent's learned
policy, and the adaptive opportunity-scoring threshold all induce
short-range dependence between consecutive trades. An ordinary
(element-wise) bootstrap destroys that structure by resampling each
trade independently; a **block** bootstrap instead resamples contiguous
chunks, preserving whatever local dependence exists within a block while
still randomizing the overall path.

The circular block bootstrap (Politis & Romano, 1992)
-----------------------------------------------------------
Given a series of length n and block size L:
    1. Compute the number of blocks needed: ceil(n / L)
    2. For each block, draw a uniformly random start index s in [0, n)
    3. Take the L elements starting at s, wrapping around circularly
       (index (s+i) mod n) — the "circular" part, which avoids the
       endpoint bias an ordinary (non-wrapping) block bootstrap has,
       where blocks near the end of the series are underrepresented.
    4. Concatenate all drawn blocks and truncate to length n.

With block_size=1 this degenerates to an ordinary i.i.d. bootstrap —
useful deliberately, since that's the independence assumption the
analytical risk-of-ruin formula (risk/ruin.py) relies on. Running this
module with block_size=1 and comparing its empirical probability_of_ruin
against risk.ruin.risk_of_ruin()'s analytical result is a genuine
cross-validation of that formula, not just two unrelated numbers.
"""

from __future__ import annotations

import numpy as np

from backtesting.types import MonteCarloStressResult
from configs.backtest_schema import MonteCarloStressConfig


def circular_block_bootstrap(
    x: np.ndarray, block_size: int, n_paths: int, rng: np.random.Generator
) -> np.ndarray:
    """Returns an array of shape (n_paths, len(x)) of resampled series."""
    n = len(x)
    if n == 0:
        raise ValueError("x must not be empty")
    if block_size > n:
        raise ValueError(f"block_size ({block_size}) cannot exceed series length ({n})")

    n_blocks = int(np.ceil(n / block_size))
    paths = np.empty((n_paths, n_blocks * block_size))

    for p in range(n_paths):
        starts = rng.integers(0, n, size=n_blocks)
        blocks = [x[np.arange(s, s + block_size) % n] for s in starts]
        paths[p] = np.concatenate(blocks)

    return paths[:, :n]


class MonteCarloStressTester:
    def __init__(self, config: MonteCarloStressConfig) -> None:
        self._config = config
        self._rng = np.random.default_rng(config.random_seed)

    def run(self, historical_pnls: list[float]) -> MonteCarloStressResult:
        if len(historical_pnls) < 2:
            raise ValueError("Need at least 2 historical trades to run a stress test")

        pnls = np.array(historical_pnls)
        c = self._config
        resampled_paths = circular_block_bootstrap(pnls, c.block_size, c.n_paths, self._rng)

        final_equities = np.empty(c.n_paths)
        max_drawdowns_pct = np.empty(c.n_paths)
        ruined = np.zeros(c.n_paths, dtype=bool)

        for i in range(c.n_paths):
            equity_curve = c.starting_capital + np.cumsum(resampled_paths[i])
            equity_with_start = np.concatenate([[c.starting_capital], equity_curve])

            final_equities[i] = equity_with_start[-1]
            ruined[i] = bool(np.any(equity_with_start <= 0))

            running_peak = np.maximum.accumulate(equity_with_start)
            safe_peak = np.maximum(running_peak, 1e-9)
            drawdown_pct_series = (running_peak - equity_with_start) / safe_peak
            max_drawdowns_pct[i] = np.max(drawdown_pct_series)

        return MonteCarloStressResult(
            n_paths=c.n_paths,
            block_size=c.block_size,
            starting_capital=c.starting_capital,
            final_equity_mean=float(np.mean(final_equities)),
            final_equity_median=float(np.median(final_equities)),
            final_equity_p5=float(np.percentile(final_equities, 5)),
            final_equity_p95=float(np.percentile(final_equities, 95)),
            max_drawdown_pct_mean=float(np.mean(max_drawdowns_pct)),
            max_drawdown_pct_median=float(np.median(max_drawdowns_pct)),
            max_drawdown_pct_p95=float(np.percentile(max_drawdowns_pct, 95)),
            probability_of_ruin=float(np.mean(ruined)),
        )
