"""Anomaly Detection — shared types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AnomalyScore:
    """
    Output of the Anomaly Detector for one MarketState observation.

    `raw_score` is sklearn's own `IsolationForest.score_samples` output
    (higher = more normal, unbounded above but roughly centered near 0
    for typical points and increasingly negative for anomalies).
    `anomaly_score` is that raw score min-max-normalized against the
    TRAINING set's own score distribution into [0, 1] (0 = as normal as
    training data gets, 1 = as anomalous as the most anomalous training
    point or worse) — this normalization is what makes `score_threshold`
    in `configs/anomaly_schema.py` a portable, dataset-relative
    threshold rather than an opaque raw sklearn number nobody can eyeball.

    `is_anomaly` combines TWO independent signals, both of which must
    agree region-wise (documented in the detector, not silently ANDed
    without explanation): sklearn's own contamination-derived `predict()`
    label (`sklearn_flagged`), AND whether `anomaly_score` clears the
    platform's own `score_threshold`. Requiring both lines up sklearn's
    "how the model was trained to expect anomalies" with the platform's
    "how aggressively we want to actually act on that" — see
    `AnomalyDetector.score` for exactly how they're combined.
    """

    symbol: str
    epoch: int
    raw_score: float
    anomaly_score: float  # normalized to [0, 1], higher = more anomalous
    sklearn_flagged: bool
    is_anomaly: bool

    @property
    def is_valid(self) -> bool:
        return self.anomaly_score == self.anomaly_score  # False only for NaN


@dataclass(frozen=True, slots=True)
class AnomalyDetectionReport:
    """Batch summary — used by the Continuous Learning Orchestrator's
    data-validation step to decide whether to drop/flag a chunk of newly
    collected training data before it reaches feature engineering."""

    scores: tuple[AnomalyScore, ...]
    n_flagged: int
    flagged_fraction: float
