"""
Level 1 — Market Regime Detection: shared types.

Every regime detector (rule-based, HMM, GMM, ...) implements the same
`RegimeDetector` protocol and returns the same `RegimeClassification`
shape, so they're interchangeable and — once the Ensemble Fusion Engine
exists — combinable as independent evidence sources. No detector here
makes a trading decision; it only classifies the environment and reports
how confident it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from state_encoder.types import MarketState


class RegimeLabel(str, Enum):
    STRONG_TREND = "strong_trend"
    WEAK_TREND = "weak_trend"
    MEAN_REVERSION = "mean_reversion"
    RANGE = "range"
    COMPRESSION = "compression"
    EXPANSION = "expansion"
    BREAKOUT = "breakout"
    FALSE_BREAKOUT = "false_breakout"
    TRANSITION = "transition"
    RANDOM_WALK = "random_walk"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass(frozen=True, slots=True)
class RegimeClassification:
    """
    Output of one regime detector for one symbol at one point in time.

    `probabilities` holds the full distribution over all labels the
    detector considered (not just the argmax) — later stages (Confidence
    Engine, Fusion Engine) need the full distribution, not just a point
    label, to reason about ambiguity between regimes.
    """

    symbol: str
    epoch: int
    detector_name: str
    regime: RegimeLabel
    confidence: float                          # = probabilities[regime], in [0, 1]
    probabilities: dict[RegimeLabel, float] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.confidence == self.confidence  # False only for NaN


class RegimeDetector(Protocol):
    """Common interface every regime detector implements."""

    name: str

    def classify(self, state: MarketState) -> RegimeClassification: ...
