"""
Model Registry — high-level API.

Wraps a `ModelRegistryStore` with the operations the platform actually
needs: register a newly-trained candidate, promote it to champion
(demoting whoever held that title), reject it, and roll back to a
previous champion if a promoted version turns out to be bad in
production. Every promotion and demotion is recorded in the promotion
history — an audit trail of "what was live when and why," which matters
for a system making real trading decisions.
"""

from __future__ import annotations

from dataclasses import replace

from model_registry.store import ModelRegistryStore
from model_registry.types import ModelStatus, ModelVersion, PromotionRecord


class ModelNotFoundError(Exception):
    pass


class ModelRegistry:
    def __init__(self, store: ModelRegistryStore) -> None:
        self._store = store

    def register(
        self,
        model_type: str,
        model_name: str,
        hyperparameters: dict[str, object],
        training_start_epoch: int,
        training_end_epoch: int,
        validation_metrics: dict[str, float],
        artifact_reference: str,
        created_at: int,
        parent_version_id: str | None = None,
    ) -> ModelVersion:
        """Register a newly-trained model version as a CANDIDATE (not yet compared to anything)."""
        existing = self._store.list_versions(model_type)
        next_version = (existing[-1].version + 1) if existing else 1
        model_id = f"{model_name}-v{next_version}"

        version = ModelVersion(
            model_id=model_id,
            model_type=model_type,
            version=next_version,
            created_at=created_at,
            training_start_epoch=training_start_epoch,
            training_end_epoch=training_end_epoch,
            hyperparameters=hyperparameters,
            validation_metrics=validation_metrics,
            artifact_reference=artifact_reference,
            status=ModelStatus.CANDIDATE,
            parent_version_id=parent_version_id,
        )
        self._store.save_version(version)
        return version

    def get_champion(self, model_type: str) -> ModelVersion | None:
        return self._store.get_champion(model_type)

    def get_version(self, model_id: str) -> ModelVersion:
        version = self._store.get_version(model_id)
        if version is None:
            raise ModelNotFoundError(f"No model version found with id '{model_id}'")
        return version

    def list_versions(self, model_type: str) -> list[ModelVersion]:
        return self._store.list_versions(model_type)

    def promote(self, model_id: str, epoch: int, reason: str) -> ModelVersion:
        """
        Promote a version to CHAMPION, demoting whoever currently holds
        that title (if anyone) to RETIRED. Records a PromotionRecord
        regardless — even the first-ever promotion for a model_type,
        where `demoted_model_id` is None, gets an audit entry.
        """
        candidate = self.get_version(model_id)
        current_champion = self._store.get_champion(candidate.model_type)

        if current_champion is not None:
            demoted = replace(current_champion, status=ModelStatus.RETIRED)
            self._store.save_version(demoted)

        promoted = replace(candidate, status=ModelStatus.CHAMPION)
        self._store.save_version(promoted)

        self._store.append_promotion_record(
            PromotionRecord(
                model_type=candidate.model_type,
                promoted_model_id=model_id,
                demoted_model_id=current_champion.model_id if current_champion else None,
                epoch=epoch,
                reason=reason,
            )
        )
        return promoted

    def reject(self, model_id: str) -> ModelVersion:
        candidate = self.get_version(model_id)
        rejected = replace(candidate, status=ModelStatus.REJECTED)
        self._store.save_version(rejected)
        return rejected

    def rollback(self, model_type: str, epoch: int, reason: str) -> ModelVersion:
        """
        Revert to the most recently RETIRED version for this model_type —
        i.e. undo the last promotion. Raises if there's no prior champion
        to roll back to (nothing retired yet).
        """
        retired = [
            v for v in self._store.list_versions(model_type)
            if v.status == ModelStatus.RETIRED
        ]
        if not retired:
            raise ModelNotFoundError(
                f"No retired version to roll back to for model_type '{model_type}'."
            )
        most_recent_retired = max(retired, key=lambda v: v.version)
        return self.promote(most_recent_retired.model_id, epoch, reason)

    def promotion_history(self, model_type: str) -> list[PromotionRecord]:
        return self._store.get_promotion_history(model_type)
