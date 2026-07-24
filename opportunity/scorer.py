"""
Trade Opportunity Scorer — Level 5.

Two distinct responsibilities, matching the spec's two named components:

1. Confidence Engine (quality score): combine independent evidence —
   expected value, risk-adjusted return, regime confidence, probability
   confidence, and prediction certainty — into a single normalized [0,1]
   trade quality score.

2. Trade Opportunity Management (adaptive threshold): monitor rolling
   approval frequency and nudge the quality-score bar up or down to stay
   near a target frequency, without ever:
     - forcing a trade to meet a quota (the mechanism only ever changes
       the THRESHOLD a trade must clear — it never fabricates approval
       for a trade that didn't clear it, and never touches the upstream
       EV/Risk gates)
     - lowering standards below the EV/Risk floors (those gates are
       checked unconditionally, before the adaptive threshold is even
       consulted — see `evaluate()`)
     - becoming permanently stuck too strict or too loose (the threshold
       is bounded in [threshold_min, threshold_max] and moves in both
       directions depending on which side of the target band the
       observed frequency falls on)

Quality score formula
-----------------------
Each raw input is normalized to [0,1] first (since EV_pct and
risk_adjusted_score are unbounded, everything else is already bounded):

    ev_component                  = clip(ev_pct / ev_pct_scale, 0, 1)
    risk_adjusted_component       = clip(risk_adjusted_score / risk_adjusted_scale, 0, 1)
    regime_confidence_component   = regime.confidence                      (already in [0,1])
    probability_confidence_component = 2*(probability.confidence - 0.5)    (maps [0.5,1] -> [0,1])
    certainty_component           = 1 - probability.uncertainty            (already in [0,1))

    quality_score = w_ev*ev_component + w_ra*risk_adjusted_component
                  + w_rc*regime_confidence_component + w_pc*probability_confidence_component
                  + w_c*certainty_component            (weights sum to 1, so quality_score in [0,1])

Adaptive threshold mechanism
------------------------------
Maintain a rolling window (deque) of the last `rolling_window_size`
evaluation outcomes (approved/rejected), per regime if
`per_regime_adjustment` is on. Every `adjustment_cooldown` evaluations,
once at least `min_samples_for_adjustment` samples exist in the window:

    observed_frequency = approved_count / window_size

    if observed_frequency < target * frequency_band_low:   # starvation
        threshold -= adjustment_step   (clipped at threshold_min)
    elif observed_frequency > target * frequency_band_high:  # overtrading risk
        threshold += adjustment_step   (clipped at threshold_max)
    # else: within the target band, leave it alone

This is deliberately a slow, bounded, hysteresis-protected drift — not a
per-trade reaction — so it responds to sustained regime shifts in
opportunity quality rather than chasing noise.
"""

from __future__ import annotations

from collections import deque

from configs.opportunity_schema import OpportunityScoringConfig
from expected_value.types import EVEstimate
from monte_carlo.types import PricePathSimulationResult
from opportunity.types import FrequencyStats, QualityScoreComponents, TradeOpportunity
from probability.types import ProbabilityEstimate
from regime.types import RegimeClassification, RegimeLabel
from risk.types import RiskAssessment

NAN = float("nan")


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


class _RegimeBucketState:
    __slots__ = ("threshold", "history", "evaluations_since_adjustment")

    def __init__(self, initial_threshold: float, window_size: int) -> None:
        self.threshold = initial_threshold
        self.history: deque[bool] = deque(maxlen=window_size)
        self.evaluations_since_adjustment = 0


class TradeOpportunityScorer:
    def __init__(self, config: OpportunityScoringConfig) -> None:
        self._config = config
        self._buckets: dict[RegimeLabel | None, _RegimeBucketState] = {}

    def _bucket_key(self, regime: RegimeLabel) -> RegimeLabel | None:
        return regime if self._config.per_regime_adjustment else None

    def _bucket_for(self, regime: RegimeLabel) -> _RegimeBucketState:
        key = self._bucket_key(regime)
        if key not in self._buckets:
            self._buckets[key] = _RegimeBucketState(
                self._config.base_confidence_threshold, self._config.rolling_window_size
            )
        return self._buckets[key]

    # ------------------------------------------------------------------ #
    # Quality score
    # ------------------------------------------------------------------ #

    def _compute_components(
        self,
        ev: EVEstimate,
        risk: RiskAssessment,
        regime: RegimeClassification,
        probability: ProbabilityEstimate,
        mc_result: PricePathSimulationResult | None = None,
    ) -> QualityScoreComponents:
        c = self._config
        ev_component = _clip01(ev.expected_value_pct / c.ev_pct_scale) if ev.is_valid else 0.0
        risk_adjusted_component = (
            _clip01(ev.risk_adjusted_score / c.risk_adjusted_scale) if ev.is_valid else 0.0
        )
        regime_confidence_component = (
            _clip01(regime.confidence) if regime.is_valid else 0.0
        )
        probability_confidence_component = (
            _clip01(2.0 * (probability.confidence - 0.5)) if probability.is_valid else 0.0
        )
        certainty_component = (
            _clip01(1.0 - probability.uncertainty) if probability.is_valid else 0.0
        )
        # Maps mc_result.implied_direction_confidence's [0.5, 1.0] range onto [0, 1],
        # same transform as probability_confidence_component above — 0.0 when no
        # Monte Carlo evidence was supplied (component simply drops out of the
        # weighted sum whenever mc_confidence_weight is also 0, its default).
        mc_confidence_component = (
            _clip01(2.0 * (mc_result.implied_direction_confidence - 0.5))
            if mc_result is not None and mc_result.is_valid
            else 0.0
        )
        return QualityScoreComponents(
            ev_component=ev_component,
            risk_adjusted_component=risk_adjusted_component,
            regime_confidence_component=regime_confidence_component,
            probability_confidence_component=probability_confidence_component,
            certainty_component=certainty_component,
            mc_confidence_component=mc_confidence_component,
        )

    def _weighted_score(self, components: QualityScoreComponents) -> float:
        w = self._config.quality_weights
        return (
            w.ev_weight * components.ev_component
            + w.risk_adjusted_weight * components.risk_adjusted_component
            + w.regime_confidence_weight * components.regime_confidence_component
            + w.probability_confidence_weight * components.probability_confidence_component
            + w.certainty_weight * components.certainty_component
            + w.mc_confidence_weight * components.mc_confidence_component
        )

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        ev: EVEstimate,
        risk: RiskAssessment,
        regime: RegimeClassification,
        probability: ProbabilityEstimate,
        mc_result: PricePathSimulationResult | None = None,
    ) -> TradeOpportunity:
        veto_reasons: list[str] = []

        if not ev.is_valid or not ev.is_positive_ev:
            veto_reasons.append("Upstream EV gate did not approve this trade.")
        if not risk.approved:
            veto_reasons.append("Upstream Risk gate did not approve this trade.")

        components = self._compute_components(ev, risk, regime, probability, mc_result)
        quality_score = self._weighted_score(components)

        bucket = self._bucket_for(regime.regime)
        threshold_applied = bucket.threshold

        if quality_score < threshold_applied:
            veto_reasons.append(
                f"Quality score {quality_score:.3f} is below the current threshold "
                f"{threshold_applied:.3f} for regime '{regime.regime.value}'."
            )

        approved = len(veto_reasons) == 0

        self._record_and_maybe_adjust(bucket, approved)

        return TradeOpportunity(
            symbol=ev.symbol,
            epoch=ev.epoch,
            regime=regime.regime,
            quality_score=quality_score,
            components=components,
            threshold_applied=threshold_applied,
            approved=approved,
            veto_reasons=veto_reasons,
        )

    def _record_and_maybe_adjust(self, bucket: _RegimeBucketState, approved: bool) -> None:
        bucket.history.append(approved)
        bucket.evaluations_since_adjustment += 1

        if bucket.evaluations_since_adjustment < self._config.adjustment_cooldown:
            return
        if len(bucket.history) < self._config.min_samples_for_adjustment:
            return

        bucket.evaluations_since_adjustment = 0
        observed_frequency = sum(bucket.history) / len(bucket.history)
        target = self._config.target_trade_frequency

        if observed_frequency < target * self._config.frequency_band_low:
            bucket.threshold = max(
                self._config.threshold_min,
                bucket.threshold - self._config.threshold_adjustment_step,
            )
        elif observed_frequency > target * self._config.frequency_band_high:
            bucket.threshold = min(
                self._config.threshold_max,
                bucket.threshold + self._config.threshold_adjustment_step,
            )

    # ------------------------------------------------------------------ #
    # Introspection / monitoring
    # ------------------------------------------------------------------ #

    def frequency_stats(self, regime: RegimeLabel | None = None) -> FrequencyStats | None:
        key = regime if self._config.per_regime_adjustment else None
        bucket = self._buckets.get(key)
        if bucket is None or len(bucket.history) == 0:
            return None
        return FrequencyStats(
            regime=key,
            window_size=len(bucket.history),
            approved_count=sum(bucket.history),
            observed_frequency=sum(bucket.history) / len(bucket.history),
            current_threshold=bucket.threshold,
        )

    def current_threshold(self, regime: RegimeLabel) -> float:
        return self._bucket_for(regime).threshold
