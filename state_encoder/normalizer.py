"""
Online Normalizer — Welford's algorithm.

Raw features arrive on wildly different scales (RSI in [0,100], MACD
histogram in fractions of a pip, entropy in bits, ATR in price units).
Before they can be combined into the State Encoder's conceptual
dimensions, they need to be on a common scale. This module provides a
streaming z-score normalizer, updated one observation at a time with
no need to store history:

    n      += 1
    delta   = x - mean
    mean   += delta / n
    delta2  = x - mean            (uses the *updated* mean)
    M2     += delta * delta2
    variance = M2 / (n - 1)        for n > 1, else 0
    std      = sqrt(variance)
    z        = (x - mean) / std    (0 if std == 0, i.e. no variation seen yet)

This is Welford's algorithm — numerically stable (no catastrophic
cancellation from naive sum-of-squares formulas) and O(1) memory/update,
which matters here since this runs on every tick/candle indefinitely.

Causality note: because this is a running statistic, feeding data
chronologically (live streaming, or backtesting in time order) produces
the same sequence of z-scores either way — there's no lookahead, and no
separate "fit" step that could leak future information into past
z-scores. This is what lets the same normalizer object be used
identically in training and live trading, per the platform's core
requirement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class _FeatureStats:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def zscore(self, x: float) -> float:
        std = self.std
        if std == 0.0:
            return 0.0
        return (x - self.mean) / std


class OnlineNormalizer:
    """Maintains independent running stats per feature key."""

    def __init__(self) -> None:
        self._stats: dict[str, _FeatureStats] = {}

    def _stats_for(self, key: str) -> _FeatureStats:
        if key not in self._stats:
            self._stats[key] = _FeatureStats()
        return self._stats[key]

    def update(self, key: str, value: float) -> None:
        if value != value:  # NaN check without importing math.isnan at call sites
            return  # never let a NaN pollute running statistics
        self._stats_for(key).update(value)

    def zscore(self, key: str, value: float) -> float:
        """Z-score `value` against the running stats for `key`. Does NOT update stats."""
        if value != value:
            return float("nan")
        return self._stats_for(key).zscore(value)

    def update_and_zscore(self, key: str, value: float) -> float:
        """Update running stats with `value`, then return its z-score
        (computed against the stats *including* this observation — matches
        how a streaming system actually sees data: it doesn't know a
        value is an outlier until it's already been folded into history)."""
        if value != value:
            return float("nan")
        self._stats_for(key).update(value)
        return self._stats_for(key).zscore(value)

    def sample_count(self, key: str) -> int:
        return self._stats.get(key, _FeatureStats()).n

    def to_dict(self) -> dict:
        """Serialize for persistence (e.g. Supabase model registry)."""
        return {
            key: {"n": s.n, "mean": s.mean, "m2": s.m2} for key, s in self._stats.items()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OnlineNormalizer":
        normalizer = cls()
        for key, s in data.items():
            stats = _FeatureStats(n=s["n"], mean=s["mean"], m2=s["m2"])
            normalizer._stats[key] = stats
        return normalizer
