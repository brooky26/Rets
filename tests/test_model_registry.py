import tempfile
from pathlib import Path

import pytest

from model_registry.registry import ModelNotFoundError, ModelRegistry
from model_registry.store import InMemoryModelRegistryStore, JSONFileModelRegistryStore
from model_registry.types import ModelStatus


def make_registry(store=None) -> ModelRegistry:
    return ModelRegistry(store or InMemoryModelRegistryStore())


def test_register_assigns_version_1_to_first_model():
    registry = make_registry()
    v = registry.register(
        model_type="probability_estimator", model_name="bayesian_logistic",
        hyperparameters={"prior_precision": 1.0}, training_start_epoch=0, training_end_epoch=1000,
        validation_metrics={"brier_score": 0.2}, artifact_reference="s3://models/v1.pkl", created_at=2000,
    )
    assert v.version == 1
    assert v.model_id == "bayesian_logistic-v1"
    assert v.status == ModelStatus.CANDIDATE


def test_register_increments_version_across_calls():
    registry = make_registry()
    registry.register(
        model_type="probability_estimator", model_name="bayesian_logistic", hyperparameters={},
        training_start_epoch=0, training_end_epoch=1000, validation_metrics={},
        artifact_reference="a", created_at=1000,
    )
    v2 = registry.register(
        model_type="probability_estimator", model_name="bayesian_logistic", hyperparameters={},
        training_start_epoch=1000, training_end_epoch=2000, validation_metrics={},
        artifact_reference="b", created_at=2000,
    )
    assert v2.version == 2


def test_version_numbering_independent_per_model_type():
    registry = make_registry()
    registry.register(model_type="probability_estimator", model_name="a", hyperparameters={},
                       training_start_epoch=0, training_end_epoch=1, validation_metrics={},
                       artifact_reference="x", created_at=1)
    v = registry.register(model_type="regime_detector", model_name="b", hyperparameters={},
                           training_start_epoch=0, training_end_epoch=1, validation_metrics={},
                           artifact_reference="y", created_at=1)
    assert v.version == 1


def test_no_champion_before_any_promotion():
    registry = make_registry()
    registry.register(model_type="probability_estimator", model_name="a", hyperparameters={},
                       training_start_epoch=0, training_end_epoch=1, validation_metrics={},
                       artifact_reference="x", created_at=1)
    assert registry.get_champion("probability_estimator") is None


def test_promote_sets_champion_status():
    registry = make_registry()
    v = registry.register(model_type="probability_estimator", model_name="a", hyperparameters={},
                           training_start_epoch=0, training_end_epoch=1, validation_metrics={},
                           artifact_reference="x", created_at=1)
    promoted = registry.promote(v.model_id, epoch=100, reason="first deployment")
    assert promoted.status == ModelStatus.CHAMPION
    assert registry.get_champion("probability_estimator").model_id == v.model_id


def test_promoting_new_version_demotes_previous_champion():
    registry = make_registry()
    v1 = registry.register(model_type="probability_estimator", model_name="a", hyperparameters={},
                            training_start_epoch=0, training_end_epoch=1, validation_metrics={},
                            artifact_reference="x", created_at=1)
    registry.promote(v1.model_id, epoch=100, reason="initial")
    v2 = registry.register(model_type="probability_estimator", model_name="a", hyperparameters={},
                            training_start_epoch=1, training_end_epoch=2, validation_metrics={},
                            artifact_reference="y", created_at=2)
    registry.promote(v2.model_id, epoch=200, reason="challenger won")

    assert registry.get_champion("probability_estimator").model_id == v2.model_id
    assert registry.get_version(v1.model_id).status == ModelStatus.RETIRED


def test_reject_sets_rejected_status_and_does_not_affect_champion():
    registry = make_registry()
    v1 = registry.register(model_type="probability_estimator", model_name="a", hyperparameters={},
                            training_start_epoch=0, training_end_epoch=1, validation_metrics={},
                            artifact_reference="x", created_at=1)
    registry.promote(v1.model_id, epoch=100, reason="initial")
    v2 = registry.register(model_type="probability_estimator", model_name="a", hyperparameters={},
                            training_start_epoch=1, training_end_epoch=2, validation_metrics={},
                            artifact_reference="y", created_at=2)
    registry.reject(v2.model_id)

    assert registry.get_version(v2.model_id).status == ModelStatus.REJECTED
    assert registry.get_champion("probability_estimator").model_id == v1.model_id


def test_rollback_restores_previous_champion():
    registry = make_registry()
    v1 = registry.register(model_type="probability_estimator", model_name="a", hyperparameters={},
                            training_start_epoch=0, training_end_epoch=1, validation_metrics={},
                            artifact_reference="x", created_at=1)
    registry.promote(v1.model_id, epoch=100, reason="initial")
    v2 = registry.register(model_type="probability_estimator", model_name="a", hyperparameters={},
                            training_start_epoch=1, training_end_epoch=2, validation_metrics={},
                            artifact_reference="y", created_at=2)
    registry.promote(v2.model_id, epoch=200, reason="challenger won")

    rolled_back = registry.rollback("probability_estimator", epoch=300, reason="v2 underperforming live")
    assert rolled_back.model_id == v1.model_id
    assert registry.get_champion("probability_estimator").model_id == v1.model_id
    assert registry.get_version(v2.model_id).status == ModelStatus.RETIRED


def test_rollback_raises_when_nothing_to_roll_back_to():
    registry = make_registry()
    v1 = registry.register(model_type="probability_estimator", model_name="a", hyperparameters={},
                            training_start_epoch=0, training_end_epoch=1, validation_metrics={},
                            artifact_reference="x", created_at=1)
    registry.promote(v1.model_id, epoch=100, reason="initial")
    with pytest.raises(ModelNotFoundError, match="No retired version"):
        registry.rollback("probability_estimator", epoch=200, reason="test")


def test_get_version_raises_for_unknown_id():
    registry = make_registry()
    with pytest.raises(ModelNotFoundError):
        registry.get_version("nonexistent-v99")


def test_promotion_history_records_every_promotion_and_demotion():
    registry = make_registry()
    v1 = registry.register(model_type="probability_estimator", model_name="a", hyperparameters={},
                            training_start_epoch=0, training_end_epoch=1, validation_metrics={},
                            artifact_reference="x", created_at=1)
    registry.promote(v1.model_id, epoch=100, reason="initial deployment")
    v2 = registry.register(model_type="probability_estimator", model_name="a", hyperparameters={},
                            training_start_epoch=1, training_end_epoch=2, validation_metrics={},
                            artifact_reference="y", created_at=2)
    registry.promote(v2.model_id, epoch=200, reason="challenger won")

    history = registry.promotion_history("probability_estimator")
    assert len(history) == 2
    assert history[0].demoted_model_id is None
    assert history[1].demoted_model_id == v1.model_id
    assert history[1].promoted_model_id == v2.model_id


def test_list_versions_sorted_by_version_number():
    registry = make_registry()
    for i in range(3):
        registry.register(model_type="probability_estimator", model_name="a", hyperparameters={},
                           training_start_epoch=i, training_end_epoch=i + 1, validation_metrics={},
                           artifact_reference=f"artifact{i}", created_at=i)
    versions = registry.list_versions("probability_estimator")
    assert [v.version for v in versions] == [1, 2, 3]


def test_parent_version_lineage_tracked():
    registry = make_registry()
    v1 = registry.register(model_type="probability_estimator", model_name="a", hyperparameters={},
                            training_start_epoch=0, training_end_epoch=1, validation_metrics={},
                            artifact_reference="x", created_at=1)
    v2 = registry.register(model_type="probability_estimator", model_name="a", hyperparameters={},
                            training_start_epoch=1, training_end_epoch=2, validation_metrics={},
                            artifact_reference="y", created_at=2, parent_version_id=v1.model_id)
    assert v2.parent_version_id == v1.model_id


def test_json_store_round_trips_across_instances():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "registry.json")

        store1 = JSONFileModelRegistryStore(path)
        registry1 = make_registry(store1)
        v1 = registry1.register(model_type="probability_estimator", model_name="a", hyperparameters={"x": 1},
                                 training_start_epoch=0, training_end_epoch=1, validation_metrics={"brier": 0.2},
                                 artifact_reference="ref1", created_at=1)
        registry1.promote(v1.model_id, epoch=100, reason="initial")

        store2 = JSONFileModelRegistryStore(path)
        registry2 = make_registry(store2)
        champion = registry2.get_champion("probability_estimator")
        assert champion is not None
        assert champion.model_id == v1.model_id
        assert champion.hyperparameters == {"x": 1}
        assert len(registry2.promotion_history("probability_estimator")) == 1
