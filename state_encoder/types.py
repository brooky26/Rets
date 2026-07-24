from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Fixed, ordered dimension names — this order is the contract for
# `as_vector()`. Every downstream model (regime detection, probability
# estimation, sequence models, RL) reads this same ordering.
DIMENSION_NAMES: tuple[str, ...] = (
    "trend",
    "momentum",
    "acceleration",
    "volatility",
    "noise",
    "persistence",
    "compression_expansion",
    "complexity",
    "uncertainty",
    "liquidity",
    "market_phase",
)


@dataclass(frozen=True, slots=True)
class MarketState:
    """
    The compressed, normalized "universal state" for one symbol at one
    point in time. Every dimension is bounded in [-1, 1] (via tanh
    squashing or affine clipping) except where noted, so downstream
    models see a consistent scale regardless of which raw features fed
    into each dimension.
    """

    symbol: str
    epoch: int
    trend: float
    momentum: float
    acceleration: float
    volatility: float
    noise: float
    persistence: float           # Hurst-exponent-derived, in [-1, 1]; 0 = random walk
    compression_expansion: float  # negative = compressing, positive = expanding
    complexity: float             # fractal-dimension-derived, in [-1, 1]
    uncertainty: float
    liquidity: float              # placeholder for synthetic indices — see encoder.py docstring
    market_phase: float           # rough continuous proxy, NOT a substitute for Level 1 regime detection

    def as_vector(self) -> np.ndarray:
        return np.array([getattr(self, name) for name in DIMENSION_NAMES], dtype=np.float64)

    @property
    def is_valid(self) -> bool:
        return all(v == v for v in self.as_vector())  # v == v is False only for NaN
