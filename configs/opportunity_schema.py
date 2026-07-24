"""Config for Level 5 — Trade Opportunity Scoring."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class QualityWeights(BaseModel):
    """
    Weights combining independent evidence into a single Trade Quality
    Score. Each component is normalized to [0,1] before weighting (see
    opportunity/scorer.py for the normalization formulas) so the weights
    themselves are directly interpretable as "how much this factor
    matters," not confounded by differing raw scales.
    """

    ev_weight: float = 0.30
    risk_adjusted_weight: float = 0.25
    regime_confidence_weight: float = 0.15
    probability_confidence_weight: float = 0.15
    certainty_weight: float = 0.15  # 1 - probability.uncertainty
    mc_confidence_weight: float = Field(
        default=0.0,
        description="Weight given to Monte Carlo price-path evidence (see "
        "`monte_carlo/price_paths.py`) in the quality score. Defaults to 0.0 so existing configs "
        "that don't mention Monte Carlo integration keep summing to 1.0 unchanged; set this "
        "(and rebalance the others so the total still sums to 1.0) to actually use it — the "
        "corresponding `MonteCarloPricePathConfig.quality_score_mc_weight` should be set to the "
        "same value, since that config is what tells `TradeOpportunityScorer.evaluate()` whether "
        "an `mc_result` argument should be expected at all.",
    )

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "QualityWeights":
        total = (
            self.ev_weight + self.risk_adjusted_weight + self.regime_confidence_weight
            + self.probability_confidence_weight + self.certainty_weight + self.mc_confidence_weight
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Quality weights must sum to 1.0, got {total}")
        return self


class OpportunityScoringConfig(BaseModel):
    quality_weights: QualityWeights = QualityWeights()

    ev_pct_scale: float = Field(
        default=0.20, description="EV_pct at or above this normalizes to 1.0 in the quality score."
    )
    risk_adjusted_scale: float = Field(
        default=1.0, description="risk_adjusted_score at or above this normalizes to 1.0."
    )

    base_confidence_threshold: float = Field(
        default=0.55, description="Starting quality-score threshold a trade must clear to be approved."
    )
    threshold_min: float = Field(
        default=0.40,
        description="Hard floor the adaptive threshold can never drop below, regardless of trade "
        "starvation — this is layered on top of, not a replacement for, the unconditional EV/Risk "
        "gates from Stages 6-7, which this stage can never bypass no matter how the threshold moves.",
    )
    threshold_max: float = Field(default=0.85, description="Hard ceiling on the adaptive threshold.")
    threshold_adjustment_step: float = Field(
        default=0.02, description="How much the threshold moves per adjustment when frequency drifts out of band."
    )

    rolling_window_size: int = Field(default=50, description="Number of recent opportunities considered for frequency stats.")
    target_trade_frequency: float = Field(
        default=0.30, description="Target fraction of evaluated opportunities that should be approved."
    )
    frequency_band_low: float = Field(
        default=0.7, description="Below target*this multiplier => starvation => threshold eases down."
    )
    frequency_band_high: float = Field(
        default=1.3, description="Above target*this multiplier => overtrading risk => threshold tightens up."
    )
    min_samples_for_adjustment: int = Field(
        default=20, description="Minimum rolling-window samples before the adaptive mechanism acts at all."
    )
    adjustment_cooldown: int = Field(
        default=20, description="Minimum evaluations between threshold adjustments — hysteresis against thrashing."
    )

    per_regime_adjustment: bool = Field(
        default=True,
        description="Track rolling frequency and threshold separately per regime label, per the spec's "
        "'adapts differently for each detected market regime.'",
    )

    @model_validator(mode="after")
    def _threshold_bounds_consistent(self) -> "OpportunityScoringConfig":
        if not (0.0 <= self.threshold_min <= self.base_confidence_threshold <= self.threshold_max <= 1.0):
            raise ValueError(
                "Require 0 <= threshold_min <= base_confidence_threshold <= threshold_max <= 1"
            )
        return self

    @model_validator(mode="after")
    def _frequency_bands_sane(self) -> "OpportunityScoringConfig":
        if not (0.0 < self.frequency_band_low < 1.0 < self.frequency_band_high):
            raise ValueError("Require 0 < frequency_band_low < 1 < frequency_band_high")
        if not (0.0 < self.target_trade_frequency < 1.0):
            raise ValueError("target_trade_frequency must be in (0, 1)")
        return self
