"""
Level 2 — Probability Estimation: shared types.

The primary objective of the whole platform: estimate
    P(next movement is favorable | current market state)
not predict the next tick. Every model here outputs a *probability with
uncertainty*, never a bare point prediction — that uncertainty is what
lets the Confidence Engine and Risk Engine (later stages) distinguish
"the model thinks 70% up and is sure of it" from "the model thinks 70%
up but has barely seen data like this before."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from state_encoder.types import MarketState


@dataclass(frozen=True, slots=True)
class ProbabilityEstimate:
    """
    Output of one probability model for one symbol at one point in time.

    prob_up + prob_down should sum to ~1.0 (binary favorable/unfavorable
    framing). `uncertainty` is model-specific in origin (predictive
    variance for the Bayesian model, cross-member disagreement for the
    bagged ensemble) but always normalized to roughly [0, 1] so it's
    comparable across detectors: 0 = fully confident, 1 = maximally
    uncertain given how the model was calibrated.
    """

    symbol: str
    epoch: int
    model_name: str
    prob_up: float
    prob_down: float
    uncertainty: float
    expected_direction: int  # +1, -1, or 0 (no edge)
    confidence: float        # = max(prob_up, prob_down), in [0.5, 1.0]

    @property
    def is_valid(self) -> bool:
        return self.prob_up == self.prob_up and self.uncertainty == self.uncertainty  # NaN check


class ProbabilityEstimator(Protocol):
    """Common interface every probability model implements."""

    name: str

    def predict(self, state: MarketState) -> ProbabilityEstimate: ...
