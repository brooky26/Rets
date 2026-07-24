"""
State discretizer — maps continuous OpenContractState fields to discrete
bins for tabular Q-learning.

Kept as pure functions, separate from the agent itself, so the state
representation is independently testable and swappable (e.g. if a
function-approximation agent replaces the tabular one later, it can
consume the raw continuous OpenContractState directly and skip this
module entirely).

Binning uses `numpy.digitize`, which returns the index of the bin a
value falls into given a sorted array of edges: for edges [e0, e1, ...,
en], digitize returns 0 for values < e0, i for e[i-1] <= value < e[i],
and len(edges) for values >= the last edge — i.e. len(edges)+1 total
bins, which is exactly n_bins when n_bins-1 edges are supplied.
"""

from __future__ import annotations

import numpy as np

from configs.trade_management_schema import QLearningConfig
from trade_management.types import OpenContractState

StateKey = tuple[int, int, int]


def discretize_time_remaining(fraction: float, n_bins: int) -> int:
    """Equal-width bins over [0, 1]."""
    edges = np.linspace(0, 1, n_bins + 1)[1:-1]
    return int(np.digitize(fraction, edges))


def discretize_return(value: float, edges: list[float]) -> int:
    return int(np.digitize(value, edges))


def discretize_trend(value: float, edges: list[float]) -> int:
    return int(np.digitize(value, edges))


def discretize_state(state: OpenContractState, config: QLearningConfig) -> StateKey:
    time_bin = discretize_time_remaining(state.time_remaining_fraction, config.n_time_bins)
    return_bin = discretize_return(state.unrealized_return, config.return_bin_edges)
    trend_bin = discretize_trend(state.favorable_move_pct, config.trend_bin_edges)
    return (time_bin, return_bin, trend_bin)
