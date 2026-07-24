"""Config for the Meta Learner (`meta_learning/weight_learner.py`)."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class MetaLearningConfig(BaseModel):
    model_names: list[str] = Field(
        default=["bayesian_logistic", "bagged_gbm", "lstm_sequence", "transformer_sequence"],
        description="Names of the probability-model family members the Ensemble Fusion Engine "
        "combines — must match each model's `.name` attribute. Defines the fixed key set every "
        "stored weight vector (global or per-regime) is expected to have.",
    )
    default_weight_mode: str = Field(
        default="equal",
        description="How to initialize weights before any Bayesian optimization has run: 'equal' "
        "(1/n_models each) is the only supported starting point — an informative prior would just "
        "be a second untested hyperparameter to get wrong before any real optimization data exists.",
    )
    min_regime_samples_before_specific_weights: int = Field(
        default=30,
        description="A regime needs at least this many optimized-weight-eligible trade records "
        "before `WeightLearner` will serve its own regime-specific weights rather than falling "
        "back to the global weight vector — avoids overfitting a regime's weights to a handful of "
        "trades.",
    )

    @field_validator("model_names")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if len(v) == 0:
            raise ValueError("model_names must not be empty")
        if len(set(v)) != len(v):
            raise ValueError("model_names must not contain duplicates")
        return v

    @field_validator("default_weight_mode")
    @classmethod
    def _valid_mode(cls, v: str) -> str:
        if v != "equal":
            raise ValueError("default_weight_mode currently only supports 'equal'")
        return v
