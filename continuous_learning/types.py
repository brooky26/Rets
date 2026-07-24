"""Continuous Learning Orchestrator — shared types."""

from __future__ import annotations

from dataclasses import dataclass, field

from champion_challenger.types import PromotionDecision
from drift_detection.types import DriftReport
from meta_learning.types import EnsembleWeights
from model_registry.types import ModelVersion
from regime.types import RegimeLabel


@dataclass(frozen=True, slots=True)
class CandidateTrainingResult:
    """Outcome of training and evaluating one candidate model during one
    daily cycle — one per entry in `ContinuousLearningConfig.model_types`."""

    model_type: str
    candidate_version: ModelVersion
    n_train_samples: int
    n_holdout_samples: int
    holdout_accuracy: float
    promotion_decision: PromotionDecision | None  # None when there was no existing champion to compare against (auto-promoted)
    promoted: bool


@dataclass(frozen=True, slots=True)
class DailyCycleReport:
    """
    Full audit trail of one Continuous Learning daily cycle — every
    decision it made and why, so a human/dashboard can answer "what did
    the pipeline do last night" without re-deriving it from logs.
    """

    cycle_epoch: int
    skipped: bool
    skip_reason: str | None
    n_states_collected: int
    n_states_after_cleaning: int
    n_anomalies_flagged: int
    drift_report: DriftReport | None
    candidate_results: tuple[CandidateTrainingResult, ...] = field(default_factory=tuple)
    global_weights: EnsembleWeights | None = None
    regime_weights: dict[RegimeLabel, EnsembleWeights] = field(default_factory=dict)

    @property
    def any_promotions(self) -> bool:
        return any(r.promoted for r in self.candidate_results)

    @property
    def summary(self) -> str:
        if self.skipped:
            return f"Cycle at epoch {self.cycle_epoch} SKIPPED: {self.skip_reason}"
        promoted = [r.model_type for r in self.candidate_results if r.promoted]
        drift_note = "no drift report" if self.drift_report is None else (
            "CRITICAL drift detected" if self.drift_report.any_critical
            else ("drift warning(s)" if self.drift_report.any_warning_or_worse else "no drift detected")
        )
        return (
            f"Cycle at epoch {self.cycle_epoch}: {self.n_states_after_cleaning}/{self.n_states_collected} "
            f"states used after cleaning ({self.n_anomalies_flagged} anomalies flagged), {drift_note}, "
            f"promoted: {promoted or 'none'}."
        )
