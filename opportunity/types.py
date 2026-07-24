"""Level 5 — Trade Opportunity Scoring: shared types."""

from __future__ import annotations

from dataclasses import dataclass, field

from regime.types import RegimeLabel


@dataclass(frozen=True, slots=True)
class QualityScoreComponents:
    """The normalized (0-1) sub-scores that combine into the overall quality score, kept
    separately for explainability — every executed OR rejected trade should be able to
    show exactly which factors drove the decision, not just a single opaque number."""

    ev_component: float
    risk_adjusted_component: float
    regime_confidence_component: float
    probability_confidence_component: float
    certainty_component: float
    mc_confidence_component: float = 0.0  # 0.0 when Monte Carlo evidence wasn't supplied for this evaluation


@dataclass(frozen=True, slots=True)
class TradeOpportunity:
    """
    Output of the Trade Opportunity Scorer for one candidate trade.

    `approved` requires ALL of: the upstream EV gate passed, the upstream
    Risk gate passed, AND quality_score clears the current adaptive
    threshold for this regime. The adaptive threshold can only make the
    quality-score bar easier or harder to clear — it structurally cannot
    override the upstream EV/Risk gates, which are checked unconditionally.
    """

    symbol: str
    epoch: int
    regime: RegimeLabel
    quality_score: float                # in [0, 1]
    components: QualityScoreComponents
    threshold_applied: float            # the adaptive threshold in effect at evaluation time
    approved: bool
    veto_reasons: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.quality_score == self.quality_score  # False only for NaN


@dataclass(frozen=True, slots=True)
class FrequencyStats:
    """Rolling trade-frequency diagnostics for one regime bucket (or global, if
    per_regime_adjustment is off) — exposed mainly for monitoring/dashboards."""

    regime: RegimeLabel | None    # None = global bucket
    window_size: int
    approved_count: int
    observed_frequency: float     # approved_count / window_size
    current_threshold: float
