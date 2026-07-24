"""Config for the Continuous Learning Orchestrator (`continuous_learning/orchestrator.py`)."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ContinuousLearningConfig(BaseModel):
    enabled: bool = Field(
        default=False,
        description="If True, main.py starts the in-process APScheduler daily cycle (see "
        "`ContinuousLearningOrchestrator.start_scheduler`) against a rolling buffer of live "
        "MarketState history for the FIRST configured symbol (`market_data.connection.symbols[0]`) "
        "— this single-symbol scope is a main.py wiring simplification, not a limitation of the "
        "orchestrator itself, which accepts whatever (states, closes) a caller supplies. False "
        "(default) leaves main.py's original behavior untouched.",
    )
    model_types: list[str] = Field(
        default=["bayesian_logistic", "bagged_gbm", "lstm_sequence", "transformer_sequence"],
        description="Candidate model families trained every daily cycle. Must be a subset of "
        "what `ContinuousLearningOrchestrator` knows how to fit — see its `_fit_candidate` method.",
    )
    train_fraction: float = Field(
        default=0.7,
        description="Fraction of this cycle's cleaned data used to train candidates; the remainder "
        "is held out for validation metrics, champion-vs-challenger comparison, and building "
        "Bayesian weight-optimizer training records.",
    )
    drift_reference_fraction: float = Field(
        default=0.5,
        description="Of the cleaned data, the first this-fraction becomes the drift detector's "
        "REFERENCE window and the rest becomes CURRENT — a separate split from train_fraction "
        "since drift detection compares two windows of the SAME already-deployed champion's "
        "behavior, not a train/holdout split for a new candidate.",
    )
    min_states_for_cycle: int = Field(
        default=200,
        description="Minimum cleaned (valid, non-anomalous) MarketState observations required to "
        "run a daily cycle at all — below this, `run_daily_cycle` returns a report noting the "
        "cycle was skipped rather than training on too little data.",
    )
    run_weight_optimization: bool = Field(
        default=True, description="Whether to run Bayesian ensemble-weight optimization at the end of each cycle."
    )
    schedule_cron_hour_utc: int = Field(
        default=1,
        description="Hour (UTC, 0-23) the in-process APScheduler job runs at, when scheduling is "
        "enabled via `ContinuousLearningOrchestrator.start_scheduler`. Ignored entirely when using "
        "the external Railway-cron alternative — see the orchestrator module docstring and README "
        "for both options.",
    )
    weight_store_path: str = Field(
        default="./data_store/ensemble_weights.json",
        description="Where `WeightLearner.save_to_file`/`load_from_file` persist learned weights "
        "between cycles/restarts. Same ephemeral-Railway-filesystem caveat "
        "`configs.schema.StorageConfig`'s docstring already documents for sqlite — a real "
        "deployment should point this at a persistent volume or move it to Supabase alongside the "
        "other documented-deferred storage migration.",
    )

    @field_validator("model_types")
    @classmethod
    def _valid_model_types(cls, v: list[str]) -> list[str]:
        allowed = {"bayesian_logistic", "bagged_gbm", "lstm_sequence", "transformer_sequence"}
        unknown = set(v) - allowed
        if unknown:
            raise ValueError(f"Unknown model_types {unknown}. Allowed: {allowed}")
        if len(v) == 0:
            raise ValueError("model_types must not be empty")
        return v

    @field_validator("train_fraction", "drift_reference_fraction")
    @classmethod
    def _fraction_bounds(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("must be in (0, 1)")
        return v

    @field_validator("min_states_for_cycle")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be positive")
        return v

    @field_validator("schedule_cron_hour_utc")
    @classmethod
    def _valid_hour(cls, v: int) -> int:
        if not (0 <= v <= 23):
            raise ValueError("schedule_cron_hour_utc must be in [0, 23]")
        return v
