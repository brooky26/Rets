"""
Data-sufficiency shrinkage for ensemble fusion weights.

Design (agreed explicitly, not a default we invented silently): a model
trained on less data than its target is never sidelined entirely — it
still fits, still votes, still contributes to the fused estimate — but
its FUSION WEIGHT is shrunk proportionally to how much data it actually
got, via

    sufficiency = min(1.0, samples_available / target_samples)

This is the same idea `BayesianLogisticConfig.prior_precision` already
applies internally (shrink toward an uninformative prior when evidence
is scarce) — made explicit here and applied uniformly across every
fusion member, including the ones (Bagged GBM, sequence models) that
have no such built-in shrinkage of their own.

Champion-Challenger promotion is a SEPARATE gate, not handled here: a
model can vote in live fusion from the moment it's fit, regardless of
sufficiency, but continuous_learning/orchestrator.py additionally
requires sufficiency to clear its own (typically higher) promotion bar
before a model can become the registered champion for its type. Two
different questions — "can this model contribute a (correctly quiet)
vote right now" vs "should this model's specific fitted parameters
become the thing everything else is measured against" — deliberately
answered in two different places.
"""

from __future__ import annotations

from meta_learning.types import EnsembleWeights


def compute_sufficiency_ratio(samples_available: int, target_samples: int) -> float:
    if target_samples <= 0:
        raise ValueError("target_samples must be positive")
    if samples_available < 0:
        raise ValueError("samples_available cannot be negative")
    return min(1.0, samples_available / target_samples)


def apply_sufficiency_scaling(
    weights: EnsembleWeights, sufficiency_by_model: dict[str, float],
) -> EnsembleWeights:
    """
    Scales each model's weight by `sufficiency_by_model.get(model_name, 1.0)`
    — models not present in `sufficiency_by_model` (e.g. `monte_carlo_gbm`,
    which re-simulates live from recent closes rather than being "trained"
    in the fit-once sense) are treated as fully trusted (1.0), unaffected.

    Renormalizes so the result still sums to 1.0 (required by
    `EnsembleWeights.__post_init__`). If every member's scaled weight is
    (numerically) zero — every trained model undertrained to the point of
    a zero ratio, which given `min()` capping at 1.0 and ratios only ever
    being non-negative would require every ratio to be exactly 0 — falls
    back to equal weights among the original members, same "zero-weight
    fallback" semantics `EnsembleFusionEngine` already documents for its
    own missing-member renormalization.
    """
    scaled = {
        name: w * sufficiency_by_model.get(name, 1.0)
        for name, w in weights.weights.items()
    }
    total = sum(scaled.values())

    if total <= 0.0:
        n = len(weights.weights)
        scaled = {name: 1.0 / n for name in weights.weights}
    else:
        scaled = {name: w / total for name, w in scaled.items()}

    return EnsembleWeights(
        weights=scaled,
        source=f"{weights.source}+sufficiency_scaled",
        n_trials=weights.n_trials,
    )
