"""
Rule-Based Regime Detector.

A deliberately simple, always-available baseline: explicit thresholds on
MarketState dimensions, no training data required. This is the "champion"
until a trained model (e.g. the Gaussian HMM in regime/hmm_detector.py)
is fit, validated, and proven statistically superior via the
Champion-Challenger framework the spec calls for — not something to
throw away once the HMM exists, since it remains a robust, interpretable
fallback when the HMM hasn't seen enough data for a given symbol/regime
yet, or during drift-triggered retraining windows.

Classification logic (in priority order — first matching rule wins):

  1. BREAKOUT: expansion (compression_expansion > expansion_threshold)
     co-occurring with a trend above breakout_trend_threshold. Expansion
     alone isn't breakout — it needs directional confirmation from trend.
  2. STRONG_TREND: |trend| > strong_trend_threshold, reinforced by
     persistence > trending_persistence_threshold.
  3. WEAK_TREND: |trend| > weak_trend_threshold (but not strong).
  4. MEAN_REVERSION: persistence < mean_reversion_persistence_threshold
     (anti-persistent / mean-reverting Hurst signature) with weak/no trend.
  5. COMPRESSION: compression_expansion < compression_threshold.
  6. EXPANSION: compression_expansion > expansion_threshold (without the
     directional confirmation that would have made it a BREAKOUT above).
  7. HIGH_VOLATILITY / LOW_VOLATILITY: volatility dimension beyond its
     thresholds, when nothing more specific matched.
  8. RANDOM_WALK: persistence near zero, trend weak, volatility
     unremarkable — the "nothing structured is happening" case.
  9. RANGE: fallback when nothing else matched — bounded, non-trending,
     non-extreme conditions.

Confidence is a simple, honest function of how far past its threshold
the deciding dimension is (min(1.0, margin / threshold)) — not a
calibrated probability. This detector doesn't pretend to statistical
rigor it doesn't have; the HMM detector is where genuine likelihood-based
confidence comes from.

FALSE_BREAKOUT and TRANSITION are intentionally not reachable by this
detector: distinguishing a real breakout from a false one, or catching a
regime mid-transition, requires temporal context (what happened in the
recent past / what happens next) that a single-snapshot rule engine
structurally cannot see. Those labels are reserved for detectors with
memory (the HMM, or a future dedicated transition detector).
"""

from __future__ import annotations

from configs.regime_schema import RuleBasedRegimeConfig
from regime.types import RegimeClassification, RegimeLabel
from state_encoder.types import MarketState

NAN = float("nan")


class RuleBasedRegimeDetector:
    name = "rule_based"

    def __init__(self, config: RuleBasedRegimeConfig) -> None:
        self._config = config

    def classify(self, state: MarketState) -> RegimeClassification:
        if not state.is_valid:
            return RegimeClassification(
                symbol=state.symbol,
                epoch=state.epoch,
                detector_name=self.name,
                regime=RegimeLabel.RANGE,
                confidence=NAN,
                probabilities={},
            )

        c = self._config
        trend = state.trend
        vol = state.volatility
        persistence = state.persistence
        comp_exp = state.compression_expansion

        regime, confidence = self._decide(c, trend, vol, persistence, comp_exp)

        # Single-point-estimate probabilities: full confidence mass on the
        # decided label, none elsewhere. Honest reflection of what a
        # deterministic rule engine actually knows — it does not model a
        # distribution over alternatives the way the HMM does.
        probabilities = {label: 0.0 for label in RegimeLabel}
        probabilities[regime] = confidence

        return RegimeClassification(
            symbol=state.symbol,
            epoch=state.epoch,
            detector_name=self.name,
            regime=regime,
            confidence=confidence,
            probabilities=probabilities,
        )

    @staticmethod
    def _margin_confidence(value: float, threshold: float, cap_at: float = 1.0) -> float:
        """How far past the threshold, normalized into [0.5, cap_at]."""
        if threshold == 0:
            return cap_at
        margin = (abs(value) - abs(threshold)) / abs(threshold)
        return float(min(cap_at, 0.5 + 0.5 * min(1.0, max(0.0, margin))))

    def _decide(
        self,
        c: RuleBasedRegimeConfig,
        trend: float,
        vol: float,
        persistence: float,
        comp_exp: float,
    ) -> tuple[RegimeLabel, float]:
        # 1. Breakout: expansion + directionally-confirming trend.
        if comp_exp > c.expansion_threshold and abs(trend) > c.breakout_trend_threshold:
            confidence = self._margin_confidence(trend, c.breakout_trend_threshold)
            return RegimeLabel.BREAKOUT, confidence

        # 2/3. Trend strength.
        if abs(trend) > c.strong_trend_threshold:
            confidence = self._margin_confidence(trend, c.strong_trend_threshold)
            if persistence > c.trending_persistence_threshold:
                confidence = min(1.0, confidence + 0.1)  # persistence reinforces trend confidence
            return RegimeLabel.STRONG_TREND, confidence

        if abs(trend) > c.weak_trend_threshold:
            confidence = self._margin_confidence(trend, c.weak_trend_threshold)
            return RegimeLabel.WEAK_TREND, confidence

        # 4. Mean reversion: anti-persistent Hurst signature, no strong trend.
        if persistence < c.mean_reversion_persistence_threshold:
            confidence = self._margin_confidence(persistence, c.mean_reversion_persistence_threshold)
            return RegimeLabel.MEAN_REVERSION, confidence

        # 5/6. Compression / expansion (without breakout-level trend confirmation).
        if comp_exp < c.compression_threshold:
            confidence = self._margin_confidence(comp_exp, c.compression_threshold)
            return RegimeLabel.COMPRESSION, confidence

        if comp_exp > c.expansion_threshold:
            confidence = self._margin_confidence(comp_exp, c.expansion_threshold)
            return RegimeLabel.EXPANSION, confidence

        # 7. Volatility extremes.
        if vol > c.high_volatility_threshold:
            confidence = self._margin_confidence(vol, c.high_volatility_threshold)
            return RegimeLabel.HIGH_VOLATILITY, confidence

        if vol < c.low_volatility_threshold:
            confidence = self._margin_confidence(vol, c.low_volatility_threshold)
            return RegimeLabel.LOW_VOLATILITY, confidence

        # 8. Random walk: nothing structured.
        if (
            abs(persistence) < c.random_walk_persistence_band
            and abs(trend) < c.weak_trend_threshold
        ):
            confidence = 1.0 - (abs(persistence) / c.random_walk_persistence_band)
            return RegimeLabel.RANDOM_WALK, max(0.5, confidence)

        # 9. Fallback.
        return RegimeLabel.RANGE, 0.5
