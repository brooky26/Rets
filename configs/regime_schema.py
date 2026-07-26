"""
Config for Level 1 — Market Regime Detection.

Two detectors, two config blocks:
  - RuleBasedRegimeConfig: explicit thresholds on MarketState dimensions.
    Every threshold is a named, documented parameter — no magic numbers
    buried in the detector logic.
  - GaussianHMMConfig: hyperparameters for the Baum-Welch-trained HMM
    (number of hidden states, EM convergence criteria, which state
    dimensions feed the observation vector).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from state_encoder.types import DIMENSION_NAMES


class RuleBasedRegimeConfig(BaseModel):
    """
    Thresholds applied directly to MarketState dimensions (each already
    bounded in [-1, 1] except liquidity, which this detector ignores).
    """

    strong_trend_threshold: float = Field(
        default=0.6, description="abs(trend) above this -> strong trend candidate."
    )
    weak_trend_threshold: float = Field(
        default=0.25, description="abs(trend) above this (but below strong) -> weak trend."
    )
    high_volatility_threshold: float = 0.6
    low_volatility_threshold: float = -0.6
    compression_threshold: float = Field(
        default=-0.5, description="compression_expansion below this -> compression regime."
    )
    expansion_threshold: float = Field(
        default=0.5, description="compression_expansion above this -> expansion regime."
    )
    breakout_trend_threshold: float = Field(
        default=0.5,
        description="Expansion + trend above this magnitude, same sign -> breakout.",
    )
    mean_reversion_persistence_threshold: float = Field(
        default=-0.3, description="persistence below this -> mean-reverting regime candidate."
    )
    trending_persistence_threshold: float = Field(
        default=0.3, description="persistence above this reinforces a trend classification."
    )
    random_walk_persistence_band: float = Field(
        default=0.15,
        description="abs(persistence) below this AND weak trend AND unremarkable vol -> random walk.",
    )

    @field_validator(
        "strong_trend_threshold",
        "weak_trend_threshold",
        "high_volatility_threshold",
        "compression_threshold",
        "expansion_threshold",
        "breakout_trend_threshold",
        "random_walk_persistence_band",
    )
    @classmethod
    def _must_be_in_bounds(cls, v: float) -> float:
        if not (-1.0 <= v <= 1.0):
            raise ValueError("threshold must be within [-1, 1] (MarketState dimensions are bounded there)")
        return v


class GaussianHMMConfig(BaseModel):
    n_states: int = Field(default=4, description="Number of hidden regime states.")
    observation_dims: list[str] = Field(
        default=[
            "trend", "volatility", "persistence", "compression_expansion",
            "momentum", "acceleration", "complexity", "uncertainty",
        ],
        description=(
            "Which MarketState dimensions form the HMM's observation vector. Deliberately wider "
            "than the 4 dims RuleBasedRegimeConfig's thresholds use (trend, volatility, "
            "persistence, compression_expansion) — those 4 are still what _label_states maps "
            "learned clusters back onto (see hmm_detector.py), but fitting on the wider set gives "
            "the HMM real information the rule-based detector never sees (momentum, acceleration, "
            "complexity, uncertainty), rather than the same 4 numbers run through a different "
            "formula. This is what makes the two detectors a genuine second opinion for the "
            "regime-consensus gate (paper_trading.enable_regime_consensus_gate), not a mostly-"
            "correlated echo of each other. Excludes 'noise' (low signal), 'liquidity' (a "
            "synthetic-indices placeholder, see state_encoder/encoder.py), and 'market_phase' "
            "(itself a rough derived proxy, not an independent primary signal)."
        ),
    )
    em_max_iterations: int = 100
    em_tolerance: float = Field(
        default=1e-4, description="Stop EM when log-likelihood improvement falls below this."
    )
    min_variance: float = Field(
        default=1e-3,
        description="Floor applied to estimated per-state variances to avoid degenerate/singular states.",
    )
    random_seed: int = 42

    @field_validator("observation_dims")
    @classmethod
    def _dims_must_exist(cls, v: list[str]) -> list[str]:
        for dim in v:
            if dim not in DIMENSION_NAMES:
                raise ValueError(
                    f"'{dim}' is not a valid MarketState dimension. Valid: {DIMENSION_NAMES}"
                )
        if len(v) == 0:
            raise ValueError("observation_dims must not be empty")
        return v

    @field_validator("n_states")
    @classmethod
    def _n_states_reasonable(cls, v: int) -> int:
        if v < 2:
            raise ValueError("n_states must be >= 2 (need at least 2 regimes to distinguish)")
        return v


class RegimeDetectionConfig(BaseModel):
    rule_based: RuleBasedRegimeConfig = RuleBasedRegimeConfig()
    hmm: GaussianHMMConfig = GaussianHMMConfig()
    enable_hmm_promotion: bool = Field(
        default=False,
        description="If True, main.py fits a GaussianHMMRegimeDetector at startup on the same "
        "historical states/closes gathered for probability-model bootstrap, and registers it as "
        "a standing regime-consensus challenger (see PaperTradingOrchestrator"
        ".set_challenger_regime_detector) — both rule-based and HMM run every candle from then "
        "on, rather than one replacing the other. (Prior to the regime-consensus design, this "
        "flag instead triggered a one-shot Champion-Challenger promotion that could swap the HMM "
        "in as the sole detector; that flow still exists in regime/promotion.py for anyone who "
        "wants it, but is no longer what this flag wires up by default.) Fitting failure (e.g. "
        "insufficient bootstrap history) leaves no challenger registered — rule-based operates "
        "alone, the same safe fallback as before.",
    )
    enable_regime_consensus_gate: bool = Field(
        default=False,
        description="Only meaningful once enable_hmm_promotion has actually registered a "
        "challenger. If True, a candle where rule-based and the HMM challenger disagree on the "
        "regime label is treated as 'nothing to trade this cycle' (logged, not an error) — "
        "PaperTradingOrchestrator.on_candle bails out before scoring/execution. If False, both "
        "classifications are still computed and logged (result['regime_consensus']) for "
        "observability, but disagreement does not block trading — useful for watching how often "
        "the two would actually disagree before committing to the more conservative gated mode.",
    )
    hmm_promotion_train_fraction: float = Field(
        default=0.6,
        description="Fraction of the bootstrap history used to FIT the challenger HMM and the "
        "comparison's probability model; the remainder is the holdout both detectors are "
        "evaluated against — see regime/promotion.py.",
    )

    @field_validator("hmm_promotion_train_fraction")
    @classmethod
    def _valid_train_fraction(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("hmm_promotion_train_fraction must be in (0, 1)")
        return v
