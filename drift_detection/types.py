"""Drift Detection — shared types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DriftType(str, Enum):
    FEATURE = "feature"
    DISTRIBUTION = "distribution"
    CONCEPT = "concept"
    PERFORMANCE = "performance"


class DriftSeverity(str, Enum):
    NONE = "none"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class DriftAlert:
    drift_type: DriftType
    metric_name: str
    severity: DriftSeverity
    statistic: float
    detail: str


@dataclass(frozen=True, slots=True)
class DriftReport:
    alerts: tuple[DriftAlert, ...]
    reference_window_size: int
    current_window_size: int

    @property
    def any_warning_or_worse(self) -> bool:
        return any(a.severity != DriftSeverity.NONE for a in self.alerts)

    @property
    def any_critical(self) -> bool:
        return any(a.severity == DriftSeverity.CRITICAL for a in self.alerts)

    @property
    def should_trigger_retraining(self) -> bool:
        """Any critical alert, OR two or more simultaneous warnings, triggers retraining —
        a single warning alone is monitored, not acted on, to avoid retraining on noise."""
        if self.any_critical:
            return True
        warning_count = sum(1 for a in self.alerts if a.severity == DriftSeverity.WARNING)
        return warning_count >= 2
