"""
Model Registry — shared types.

Scope, stated up front: this registry tracks METADATA and versioning
(hyperparameters, training period, validation metrics, lineage,
promotion status) for model versions — it does not serialize or store
the model objects themselves. `artifact_reference` is an opaque string
the caller controls (a file path, an object-store key, whatever fits
the deployment) — the registry's job is knowing which version is live,
what it was trained on, how it performed, and what came before it, not
being a general-purpose pickle store.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ModelStatus(str, Enum):
    CANDIDATE = "candidate"
    CHAMPION = "champion"
    CHALLENGER = "challenger"
    RETIRED = "retired"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ModelVersion:
    model_id: str
    model_type: str
    version: int
    created_at: int
    training_start_epoch: int
    training_end_epoch: int
    hyperparameters: dict[str, object]
    validation_metrics: dict[str, float]
    artifact_reference: str
    status: ModelStatus
    parent_version_id: str | None = None


@dataclass(frozen=True, slots=True)
class PromotionRecord:
    model_type: str
    promoted_model_id: str
    demoted_model_id: str | None
    epoch: int
    reason: str
