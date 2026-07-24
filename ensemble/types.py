"""Ensemble Fusion — shared types."""

from __future__ import annotations

from dataclasses import dataclass, field

from probability.types import ProbabilityEstimate
from regime.types import RegimeLabel


@dataclass(frozen=True, slots=True)
class FusedProbabilityEstimate:
    """
    Output of `EnsembleFusionEngine.fuse` — combines evidence from
    however many member `ProbabilityEstimate`s were available this
    prediction (Bayesian Logistic, Bagged GBM, sequence models, and a
    Monte-Carlo-implied estimate, per the platform spec) into a single
    weighted view, while keeping every individual member estimate and
    the weights actually used attached for explainability (the spec's
    "explainability" requirement means a human/dashboard needs to be
    able to answer "why did the ensemble think 62% up" by looking at
    what each member said and how much it counted, not just at the
    final number).

    Deliberately mirrors `ProbabilityEstimate`'s field names
    (`prob_up`, `prob_down`, `uncertainty`, `expected_direction`,
    `confidence`) plus a `.to_probability_estimate()` adapter, so
    everything downstream that already consumes a `ProbabilityEstimate`
    (`ExpectedValueEngine`, `opportunity/scorer.py`) accepts a fused
    result without modification.
    """

    symbol: str
    epoch: int
    prob_up: float
    prob_down: float
    uncertainty: float
    expected_direction: int
    confidence: float
    member_estimates: dict[str, ProbabilityEstimate] = field(default_factory=dict)
    weights_used: dict[str, float] = field(default_factory=dict)
    regime: RegimeLabel | None = None

    @property
    def is_valid(self) -> bool:
        return self.prob_up == self.prob_up and self.uncertainty == self.uncertainty  # NaN check

    def to_probability_estimate(self) -> ProbabilityEstimate:
        return ProbabilityEstimate(
            symbol=self.symbol, epoch=self.epoch, model_name="ensemble_fusion",
            prob_up=self.prob_up, prob_down=self.prob_down, uncertainty=self.uncertainty,
            expected_direction=self.expected_direction, confidence=self.confidence,
        )


@dataclass(frozen=True, slots=True)
class WeightOptimizationRecord:
    """
    One historical trade's worth of data used by
    `ensemble.bayesian_weight_optimizer.BayesianWeightOptimizer` to
    evaluate a candidate weight vector. Produced by the Continuous
    Learning Orchestrator from walk-forward backtest / paper-trading
    history: for every trade actually taken, the per-model `prob_up`
    each member model reported at decision time, the direction traded,
    the regime active at the time, and the REALIZED return_pct that
    trade achieved (from `post_trade.types.CompletedTrade.return_pct`).

    See `bayesian_weight_optimizer`'s module docstring for exactly how
    `model_probabilities` + `direction` + `realized_return_pct` combine
    into a weight-dependent simulated return used as the optimization
    objective's input.
    """

    symbol: str
    epoch: int
    model_probabilities: dict[str, float]  # {model_name: prob_up}, as reported at decision time
    direction: int  # +1 or -1, the direction actually traded
    realized_return_pct: float  # actual outcome of that trade
    regime: RegimeLabel | None = None

