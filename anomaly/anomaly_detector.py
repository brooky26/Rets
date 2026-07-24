"""
Anomaly Detector — IsolationForest on MarketState features.

Purpose in the Continuous Learning Pipeline
-----------------------------------------------
Sits in the "collect/validate data" step, BEFORE candidate models are
trained on newly collected data: a batch of MarketState observations
containing a burst of genuinely anomalous points (a data-feed glitch
that slipped past `data/integrity.py`'s tick-level checks, a brief
period of extreme illiquidity, a synthetic-index generator quirk)
should not be allowed to silently corrupt a freshly retrained
probability model's decision boundary. This is a state-level check
(operates on the same MarketState dimensions the probability/regime
models consume), distinct from and complementary to
`data/integrity.py`'s tick-level checks (gaps, price-jump sigma,
duplicate timestamps) and `drift_detection`'s distributional-shift
checks (which compare two WHOLE windows against each other, not flag
individual points).

Why IsolationForest specifically
-------------------------------------
Isolation Forest (Liu, Ting & Zhou, 2008) isolates points by randomly
partitioning the feature space; anomalies — being few and different —
tend to get isolated in fewer random splits than normal points, so
average path length to isolation across the forest is the anomaly
signal. No distributional assumption (unlike e.g. a Gaussian
Mahalanobis-distance approach) is needed, which matters here because
MarketState dimensions are each independently bounded/squashed
(tanh-ish, per `state_encoder/types.py`) rather than jointly Gaussian.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import IsolationForest

from anomaly.types import AnomalyDetectionReport, AnomalyScore
from configs.anomaly_schema import AnomalyDetectionConfig
from state_encoder.types import MarketState

logger = logging.getLogger("anomaly.anomaly_detector")
NAN = float("nan")


class AnomalyDetector:
    def __init__(self, config: AnomalyDetectionConfig) -> None:
        self._config = config
        self._model: IsolationForest | None = None
        self._train_score_min: float | None = None
        self._train_score_max: float | None = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def _to_matrix(self, states: list[MarketState]) -> np.ndarray:
        dims = self._config.feature_dims
        return np.array([[getattr(s, dim) for dim in dims] for s in states], dtype=np.float64)

    def fit(self, states: list[MarketState]) -> "AnomalyDetector":
        valid_states = [s for s in states if s.is_valid]
        if len(valid_states) < self._config.min_training_samples:
            raise ValueError(
                f"Need at least {self._config.min_training_samples} valid MarketState observations "
                f"to fit the anomaly detector, got {len(valid_states)}."
            )

        X = self._to_matrix(valid_states)
        model = IsolationForest(
            n_estimators=self._config.n_estimators,
            contamination=self._config.contamination,
            random_state=self._config.random_seed,
        )
        model.fit(X)

        train_scores = model.score_samples(X)
        self._model = model
        self._train_score_min = float(np.min(train_scores))
        self._train_score_max = float(np.max(train_scores))
        logger.info(
            "AnomalyDetector fit on %d states — train score_samples range [%.4f, %.4f].",
            len(valid_states), self._train_score_min, self._train_score_max,
        )
        return self

    def _normalize(self, raw_score: float) -> float:
        """Min-max normalize a raw score against the TRAINING distribution,
        into [0, 1] where 1 = as anomalous as the training set's most
        anomalous point (lower raw score = more anomalous, hence the
        flip). Scores below the training min or above the training max
        clip to 1.0 / 0.0 respectively rather than extrapolating past
        the range this detector has ever actually seen."""
        assert self._train_score_min is not None and self._train_score_max is not None
        span = self._train_score_max - self._train_score_min
        if span <= 0:
            return 0.0
        normalized = (self._train_score_max - raw_score) / span
        return float(np.clip(normalized, 0.0, 1.0))

    def score(self, state: MarketState) -> AnomalyScore:
        if self._model is None:
            raise RuntimeError("AnomalyDetector.score() called before fit().")

        if not state.is_valid:
            return AnomalyScore(
                symbol=state.symbol, epoch=state.epoch, raw_score=NAN, anomaly_score=NAN,
                sklearn_flagged=False, is_anomaly=False,
            )

        x = np.array([[getattr(state, dim) for dim in self._config.feature_dims]], dtype=np.float64)
        raw_score = float(self._model.score_samples(x)[0])
        sklearn_label = int(self._model.predict(x)[0])  # -1 = anomaly, 1 = normal, per sklearn convention
        sklearn_flagged = sklearn_label == -1
        anomaly_score = self._normalize(raw_score)

        # Both signals must agree — see AnomalyScore's docstring for why
        # neither is used alone.
        is_anomaly = sklearn_flagged and anomaly_score >= self._config.score_threshold

        return AnomalyScore(
            symbol=state.symbol, epoch=state.epoch, raw_score=raw_score, anomaly_score=anomaly_score,
            sklearn_flagged=sklearn_flagged, is_anomaly=is_anomaly,
        )

    def score_batch(self, states: list[MarketState]) -> AnomalyDetectionReport:
        scores = tuple(self.score(s) for s in states)
        n_flagged = sum(1 for s in scores if s.is_anomaly)
        flagged_fraction = n_flagged / len(scores) if scores else 0.0
        return AnomalyDetectionReport(scores=scores, n_flagged=n_flagged, flagged_fraction=flagged_fraction)

    def filter_anomalies(self, states: list[MarketState]) -> list[MarketState]:
        """Convenience for the Continuous Learning Orchestrator's data-
        validation step: returns only the NON-anomalous states, in the
        same order, dropping any state flagged `is_anomaly=True`."""
        report = self.score_batch(states)
        return [s for s, score in zip(states, report.scores) if not score.is_anomaly]
