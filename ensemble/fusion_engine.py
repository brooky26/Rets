"""
Ensemble Fusion Engine — combines multi-model probability evidence.

What gets fused
--------------------
Any set of NAMED `ProbabilityEstimate`s the caller collects this cycle
— per the spec, typically Bayesian Logistic, Bagged GBM, a sequence
model (LSTM and/or Transformer), and a Monte-Carlo-implied estimate
(via `monte_carlo.price_paths` — wrap `PricePathSimulationResult` in a
`ProbabilityEstimate` first using
`monte_carlo_result_to_probability_estimate` below). The engine itself
has no opinion on WHICH models fed it; it only combines whatever
dict of `{model_name: ProbabilityEstimate}` it's given against
whatever `EnsembleWeights` the `WeightLearner` currently serves for the
active regime — which is exactly why `WeightLearner` and this engine
are separate modules (see `meta_learning/weight_learner.py`'s
docstring): re-optimizing weights should never require touching this
combination logic, and vice versa.

Graceful degradation
-------------------------
Not every model is necessarily available every cycle (a sequence model
that hasn't finished its first training run yet, a model that returned
an invalid/NaN estimate for this state). Missing/invalid members are
simply dropped and the REMAINING members' weights are renormalized to
sum to 1 among themselves — the alternative (refusing to fuse unless
every configured member is present) would make the whole platform
brittle to any one model family's startup lag, which the Continuous
Learning Orchestrator's staged training makes a normal, expected
situation rather than an edge case.

Fusion math
---------------
- `prob_up_fused` = weighted average of member `prob_up`, weights
  renormalized over available members.
- `uncertainty_fused` blends two genuinely different sources of doubt,
  per `EnsembleFusionConfig.disagreement_penalty_weight`:
  1. the weighted average of each member's OWN reported uncertainty
     (how unsure each model is, on its own terms), and
  2. cross-model DISAGREEMENT — the weighted variance of member
     `prob_up` values around the fused mean (how much the models
     contradict each other, which a low average of (1) alone could
     completely miss if every model were individually confident but
     confidently disagreeing).
  This mirrors, one level up, the same principle
  `probability/gbm.py`'s bagged ensemble uses within a single model
  family (cross-member disagreement as uncertainty) — applied here
  across independently-trained model FAMILIES instead of across
  bootstrap resamples of one family.
"""

from __future__ import annotations

import math

import numpy as np

from configs.ensemble_schema import EnsembleFusionConfig
from ensemble.types import FusedProbabilityEstimate
from meta_learning.types import EnsembleWeights
from monte_carlo.types import PricePathSimulationResult
from probability.types import ProbabilityEstimate
from regime.types import RegimeLabel

NAN = float("nan")


def monte_carlo_result_to_probability_estimate(mc_result: PricePathSimulationResult) -> ProbabilityEstimate:
    """
    Adapter: wraps a Monte Carlo price-path simulation result as a
    `ProbabilityEstimate` so it can be fed into the fusion engine
    alongside the statistical/sequence models. `prob_up`/`prob_down`
    are derived from `prob_favorable` re-expressed in absolute
    up/down terms (not "favorable to the candidate direction," since
    every OTHER member's prob_up/prob_down is direction-agnostic —
    absolute price-direction probabilities — and fusing must compare
    like with like). `uncertainty` uses `terminal_return_std`
    normalized against `sigma_per_tick * sqrt(horizon_ticks)` (the GBM
    model's own theoretical terminal std under the estimated
    parameters) so it lands in roughly [0, 1] like every other
    member's uncertainty — a ratio near 1 means the simulated outcome
    dispersion is about what the estimated volatility alone would
    predict (no extra info), materially below 1 would mean the paths
    were more decisive than raw volatility alone would suggest.
    """
    if not mc_result.is_valid:
        return ProbabilityEstimate(
            symbol=mc_result.symbol, epoch=mc_result.epoch, model_name="monte_carlo_gbm",
            prob_up=NAN, prob_down=NAN, uncertainty=NAN, expected_direction=0, confidence=NAN,
        )

    prob_favorable_to_direction = mc_result.prob_favorable
    prob_up = prob_favorable_to_direction if mc_result.direction == 1 else (1.0 - prob_favorable_to_direction)
    prob_down = 1.0 - prob_up

    theoretical_std = mc_result.sigma_per_tick * math.sqrt(mc_result.horizon_ticks)
    uncertainty = float(np.clip(mc_result.terminal_return_std / theoretical_std, 0.0, 1.0)) if theoretical_std > 0 else 1.0

    expected_direction = 1 if prob_up > 0.5 else (-1 if prob_up < 0.5 else 0)
    confidence = max(prob_up, prob_down)

    return ProbabilityEstimate(
        symbol=mc_result.symbol, epoch=mc_result.epoch, model_name="monte_carlo_gbm",
        prob_up=prob_up, prob_down=prob_down, uncertainty=uncertainty,
        expected_direction=expected_direction, confidence=confidence,
    )


class EnsembleFusionEngine:
    def __init__(self, config: EnsembleFusionConfig) -> None:
        self._config = config

    def fuse(
        self,
        symbol: str,
        epoch: int,
        member_estimates: dict[str, ProbabilityEstimate],
        weights: EnsembleWeights,
        regime: RegimeLabel | None = None,
    ) -> FusedProbabilityEstimate:
        valid_members = {
            name: est for name, est in member_estimates.items()
            if est.is_valid and name in weights.weights
        }

        if len(valid_members) < self._config.min_members_required:
            return FusedProbabilityEstimate(
                symbol=symbol, epoch=epoch, prob_up=NAN, prob_down=NAN, uncertainty=NAN,
                expected_direction=0, confidence=NAN, member_estimates=dict(member_estimates),
                weights_used={}, regime=regime,
            )

        raw_weights = {name: weights.weights[name] for name in valid_members}
        total_weight = sum(raw_weights.values())
        if total_weight <= 0:
            # All available members happen to carry zero configured weight
            # — fall back to equal weighting among them rather than
            # dividing by zero or silently fusing nothing.
            renormalized = {name: 1.0 / len(valid_members) for name in valid_members}
        else:
            renormalized = {name: w / total_weight for name, w in raw_weights.items()}

        prob_up_fused = sum(renormalized[name] * valid_members[name].prob_up for name in valid_members)
        prob_down_fused = 1.0 - prob_up_fused

        mean_reported_uncertainty = sum(
            renormalized[name] * valid_members[name].uncertainty for name in valid_members
        )
        disagreement = sum(
            renormalized[name] * (valid_members[name].prob_up - prob_up_fused) ** 2 for name in valid_members
        )
        disagreement_std = math.sqrt(disagreement)
        alpha = self._config.disagreement_penalty_weight
        uncertainty_fused = float(np.clip((1 - alpha) * mean_reported_uncertainty + alpha * disagreement_std, 0.0, 1.0))

        expected_direction = 1 if prob_up_fused > 0.5 else (-1 if prob_up_fused < 0.5 else 0)
        confidence = max(prob_up_fused, prob_down_fused)

        return FusedProbabilityEstimate(
            symbol=symbol, epoch=epoch, prob_up=prob_up_fused, prob_down=prob_down_fused,
            uncertainty=uncertainty_fused, expected_direction=expected_direction, confidence=confidence,
            member_estimates=dict(member_estimates), weights_used=renormalized, regime=regime,
        )
