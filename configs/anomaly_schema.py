"""Config for the Anomaly Detector (`anomaly/anomaly_detector.py`)."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from state_encoder.types import DIMENSION_NAMES


class AnomalyDetectionConfig(BaseModel):
    feature_dims: list[str] = Field(
        default=list(DIMENSION_NAMES),
        description="Which MarketState dimensions form the IsolationForest's input vector.",
    )
    n_estimators: int = Field(default=100, description="Number of isolation trees in the forest.")
    contamination: float = Field(
        default=0.05,
        description="Expected fraction of anomalous points in the training data — sets the "
        "decision-function threshold sklearn uses internally to derive `predict()`'s -1/1 labels. "
        "This is a prior belief about base-rate anomaly frequency, not a hard cap on how many "
        "live points get flagged.",
    )
    random_seed: int = 42
    score_threshold: float = Field(
        default=0.6,
        description="Normalized anomaly_score (see `anomaly/types.py` for the [0,1] normalization) "
        "at or above which a state is flagged `is_anomaly=True`, independent of sklearn's own "
        "contamination-derived threshold — kept as an explicit, tunable second gate so the "
        "platform's own alerting threshold isn't silently coupled to the training-time "
        "contamination assumption.",
    )
    min_training_samples: int = Field(
        default=100, description="Minimum historical states required before fit() will train a model."
    )

    @field_validator("n_estimators", "min_training_samples")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be a positive integer")
        return v

    @field_validator("contamination")
    @classmethod
    def _contamination_bounds(cls, v: float) -> float:
        if not (0.0 < v < 0.5):
            raise ValueError("contamination must be in (0, 0.5) — sklearn's own IsolationForest constraint")
        return v

    @field_validator("score_threshold")
    @classmethod
    def _threshold_bounds(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("score_threshold must be in [0, 1]")
        return v

    @field_validator("feature_dims")
    @classmethod
    def _dims_must_exist(cls, v: list[str]) -> list[str]:
        for dim in v:
            if dim not in DIMENSION_NAMES:
                raise ValueError(f"'{dim}' is not a valid MarketState dimension. Valid: {DIMENSION_NAMES}")
        if len(v) == 0:
            raise ValueError("feature_dims must not be empty")
        return v
