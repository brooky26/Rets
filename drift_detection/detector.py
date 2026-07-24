"""
Drift Detector — orchestrates all four drift types the spec names.

Four distinct questions, four distinct methods
---------------------------------------------------
- FEATURE drift: has the distribution of INPUT features shifted?
  (PSI on each monitored MarketState dimension, reference vs current)
- DISTRIBUTION drift: has the distribution of the model's OWN OUTPUTS
  shifted? (PSI on predicted probabilities, reference vs current)
- CONCEPT drift: has the RELATIONSHIP between predictions and reality
  shifted — i.e. is a given predicted probability now less trustworthy
  than it used to be, even if inputs and outputs individually look
  stable? (bootstrap test on the calibration gap: predicted_probability
  minus realized outcome, per trade)
- PERFORMANCE drift: has realized trading performance gotten worse?
  (bootstrap test on mean per-trade return, one-directional — only
  flagged when current is significantly WORSE than reference, unlike
  Champion-Challenger's symmetric "is the challenger better" test)

Concept and performance drift reuse `circular_block_bootstrap` at
block_size=1 (an ordinary i.i.d. bootstrap) — the same utility built for
Monte Carlo stress testing and Champion-Challenger comparison turns out
to be the right tool here too: "is this difference between two samples
larger than sampling noise would explain" is the same question every
time, just asked of different quantities.

Severity
-----------
PSI-based checks use the industry-standard PSI thresholds directly.
Bootstrap-based checks (concept, performance) are WARNING when the
confidence interval on the change excludes zero (a real, if possibly
small, shift) and CRITICAL when the point-estimate change additionally
exceeds a magnitude threshold — statistically detectable is not the same
as practically significant, and both matter differently.
"""

from __future__ import annotations

import numpy as np

from backtesting.monte_carlo import circular_block_bootstrap
from configs.drift_detection_schema import DriftDetectionConfig
from drift_detection.psi import population_stability_index
from drift_detection.types import DriftAlert, DriftReport, DriftSeverity, DriftType
from state_encoder.types import MarketState

_CONCEPT_DRIFT_CRITICAL_GAP_CHANGE = 0.15
_PERFORMANCE_DRIFT_CRITICAL_RETURN_CHANGE = 0.10


class DriftDetector:
    def __init__(self, config: DriftDetectionConfig) -> None:
        self._config = config
        self._rng = np.random.default_rng(config.bootstrap.random_seed)

    def detect_feature_drift(
        self, reference_states: list[MarketState], current_states: list[MarketState]
    ) -> list[DriftAlert]:
        alerts = []
        for dim in self._config.feature_dims:
            reference_values = np.array([getattr(s, dim) for s in reference_states])
            current_values = np.array([getattr(s, dim) for s in current_states])
            reference_values = reference_values[~np.isnan(reference_values)]
            current_values = current_values[~np.isnan(current_values)]
            if len(reference_values) < self._config.psi.n_bins or len(current_values) == 0:
                continue

            psi = population_stability_index(reference_values, current_values, self._config.psi.n_bins)
            severity = self._psi_severity(psi)
            alerts.append(
                DriftAlert(
                    drift_type=DriftType.FEATURE, metric_name=dim, severity=severity, statistic=psi,
                    detail=f"PSI={psi:.4f} for feature '{dim}' (reference n={len(reference_values)}, "
                    f"current n={len(current_values)}).",
                )
            )
        return alerts

    def detect_distribution_drift(
        self, reference_predicted: list[float], current_predicted: list[float]
    ) -> DriftAlert:
        ref = np.array(reference_predicted)
        cur = np.array(current_predicted)
        psi = population_stability_index(ref, cur, self._config.psi.n_bins)
        severity = self._psi_severity(psi)
        return DriftAlert(
            drift_type=DriftType.DISTRIBUTION, metric_name="predicted_probability",
            severity=severity, statistic=psi,
            detail=f"PSI={psi:.4f} on the model's predicted-probability distribution "
            f"(reference n={len(ref)}, current n={len(cur)}).",
        )

    def _psi_severity(self, psi: float) -> DriftSeverity:
        if psi >= self._config.psi.critical_threshold:
            return DriftSeverity.CRITICAL
        if psi >= self._config.psi.warning_threshold:
            return DriftSeverity.WARNING
        return DriftSeverity.NONE

    def detect_concept_drift(
        self,
        reference_predicted: list[float], reference_outcomes: list[bool],
        current_predicted: list[float], current_outcomes: list[bool],
    ) -> DriftAlert:
        ref_gap = np.array(reference_predicted) - np.array(reference_outcomes, dtype=float)
        cur_gap = np.array(current_predicted) - np.array(current_outcomes, dtype=float)

        lower, upper, point_estimate = self._bootstrap_difference_interval(ref_gap, cur_gap)
        significant = not (lower <= 0 <= upper)
        severity = DriftSeverity.NONE
        if significant:
            severity = (
                DriftSeverity.CRITICAL
                if abs(point_estimate) >= _CONCEPT_DRIFT_CRITICAL_GAP_CHANGE
                else DriftSeverity.WARNING
            )

        return DriftAlert(
            drift_type=DriftType.CONCEPT, metric_name="calibration_gap",
            severity=severity, statistic=point_estimate,
            detail=f"Calibration gap (predicted - outcome) changed by {point_estimate:+.4f} "
            f"({self._config.bootstrap.confidence_level:.0%} CI [{lower:.4f}, {upper:.4f}]).",
        )

    def detect_performance_drift(
        self, reference_returns: list[float], current_returns: list[float]
    ) -> DriftAlert:
        ref = np.array(reference_returns)
        cur = np.array(current_returns)

        lower, upper, point_estimate = self._bootstrap_difference_interval(ref, cur)
        significant_decline = upper < 0
        severity = DriftSeverity.NONE
        if significant_decline:
            severity = (
                DriftSeverity.CRITICAL
                if abs(point_estimate) >= _PERFORMANCE_DRIFT_CRITICAL_RETURN_CHANGE
                else DriftSeverity.WARNING
            )

        return DriftAlert(
            drift_type=DriftType.PERFORMANCE, metric_name="mean_return_pct",
            severity=severity, statistic=point_estimate,
            detail=f"Mean return changed by {point_estimate:+.4f} "
            f"({self._config.bootstrap.confidence_level:.0%} CI [{lower:.4f}, {upper:.4f}]).",
        )

    def _bootstrap_difference_interval(
        self, reference: np.ndarray, current: np.ndarray
    ) -> tuple[float, float, float]:
        c = self._config.bootstrap
        ref_resamples = circular_block_bootstrap(reference, 1, c.n_bootstrap_resamples, self._rng)
        cur_resamples = circular_block_bootstrap(current, 1, c.n_bootstrap_resamples, self._rng)
        diffs = cur_resamples.mean(axis=1) - ref_resamples.mean(axis=1)

        alpha = 1.0 - c.confidence_level
        lower = float(np.percentile(diffs, 100 * alpha / 2))
        upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
        point_estimate = float(np.mean(current) - np.mean(reference))
        return lower, upper, point_estimate

    def run_full_report(
        self,
        reference_states: list[MarketState], current_states: list[MarketState],
        reference_predicted: list[float], current_predicted: list[float],
        reference_outcomes: list[bool], current_outcomes: list[bool],
        reference_returns: list[float], current_returns: list[float],
    ) -> DriftReport:
        n_ref = len(reference_states)
        n_cur = len(current_states)
        if n_ref < self._config.min_samples_required or n_cur < self._config.min_samples_required:
            return DriftReport(alerts=(), reference_window_size=n_ref, current_window_size=n_cur)

        alerts: list[DriftAlert] = []
        alerts.extend(self.detect_feature_drift(reference_states, current_states))
        alerts.append(self.detect_distribution_drift(reference_predicted, current_predicted))
        alerts.append(
            self.detect_concept_drift(
                reference_predicted, reference_outcomes, current_predicted, current_outcomes
            )
        )
        alerts.append(self.detect_performance_drift(reference_returns, current_returns))

        return DriftReport(alerts=tuple(alerts), reference_window_size=n_ref, current_window_size=n_cur)
