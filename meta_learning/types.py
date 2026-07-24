"""Meta Learning — shared types."""

from __future__ import annotations

from dataclasses import dataclass, field

from regime.types import RegimeLabel


@dataclass(frozen=True, slots=True)
class EnsembleWeights:
    """
    A named weight vector over the ensemble's member models — used both
    as the GLOBAL fallback weights and as a per-regime override.
    `weights` keys must exactly match `MetaLearningConfig.model_names`
    and values must be non-negative and sum to 1.0 (validated in
    `__post_init__`, mirroring `configs.opportunity_schema.QualityWeights`'s
    own sum-to-one validation pattern, just enforced on a plain
    dataclass here rather than a Pydantic model since this is a runtime
    artifact the optimizer produces — not a user-facing config file).
    """

    weights: dict[str, float]
    source: str  # "default" | "bayesian_optimizer" | regime label string, for audit/debugging
    n_trials: int = 0  # 0 for the "equal weights" default; > 0 once Bayesian-optimized

    def __post_init__(self) -> None:
        if len(self.weights) == 0:
            raise ValueError("weights must not be empty")
        for name, w in self.weights.items():
            if w < 0:
                raise ValueError(f"weight for '{name}' is negative ({w})")
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {total:.6f}")


@dataclass(frozen=True, slots=True)
class WeightStore:
    """Everything the Meta Learner currently knows: the global fallback
    weights plus zero or more regime-specific overrides."""

    global_weights: EnsembleWeights
    regime_weights: dict[RegimeLabel, EnsembleWeights] = field(default_factory=dict)
