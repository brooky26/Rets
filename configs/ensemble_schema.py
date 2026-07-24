"""Config for `ensemble/fusion_engine.py` and `ensemble/bayesian_weight_optimizer.py`."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class EnsembleFusionConfig(BaseModel):
    disagreement_penalty_weight: float = Field(
        default=0.5,
        description="How much weighted cross-model disagreement (variance of member prob_up "
        "around the fused mean) contributes to the fused uncertainty, relative to the members' "
        "own weighted-average reported uncertainty. 0 = ignore disagreement entirely (fused "
        "uncertainty is purely the weighted average of member uncertainties); 1 = disagreement "
        "and reported uncertainty contribute equally.",
    )
    min_members_required: int = Field(
        default=1,
        description="Minimum number of VALID member estimates required to produce a fused result. "
        "Below this, `EnsembleFusionEngine.fuse` returns an invalid (NaN) FusedProbabilityEstimate "
        "rather than fusing over too little evidence.",
    )

    @field_validator("disagreement_penalty_weight")
    @classmethod
    def _bounds(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("disagreement_penalty_weight must be in [0, 1]")
        return v

    @field_validator("min_members_required")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("min_members_required must be >= 1")
        return v


class BayesianWeightOptimizerConfig(BaseModel):
    n_trials: int = Field(default=100, description="Number of Optuna trials to run per optimization call.")
    objective_mode: str = Field(
        default="multi_objective",
        description="'multi_objective' runs true multi-objective optimization over (Sharpe, Calmar) "
        "and selects a Pareto-front trial by scalarization at the end; 'custom' runs single-objective "
        "optimization directly on `quality_score_proxy * expectancy - lambda_drawdown * max_drawdown_pct`.",
    )
    lambda_drawdown: float = Field(
        default=1.0, description="Drawdown penalty weight used both in 'custom' mode's objective and "
        "in the scalarization used to pick a single trial off the Pareto front in 'multi_objective' mode.",
    )
    per_regime: bool = Field(
        default=True,
        description="If True, `optimize()` additionally runs one optimization per regime present in "
        "the training records (in addition to the global optimization) and returns both.",
    )
    min_trades_per_regime: int = Field(
        default=30,
        description="A regime's slice of training records must have at least this many trades before "
        "a regime-specific optimization is attempted for it — matches "
        "`MetaLearningConfig.min_regime_samples_before_specific_weights` in spirit (don't optimize "
        "against too little data) but is enforced independently since the two live in different "
        "modules with different responsibilities.",
    )
    sampler_seed: int = 42

    @field_validator("n_trials", "min_trades_per_regime")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be a positive integer")
        return v

    @field_validator("objective_mode")
    @classmethod
    def _valid_mode(cls, v: str) -> str:
        if v not in ("multi_objective", "custom"):
            raise ValueError("objective_mode must be 'multi_objective' or 'custom'")
        return v

    @field_validator("lambda_drawdown")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("lambda_drawdown must be >= 0")
        return v
