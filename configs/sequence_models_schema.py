"""
Config for the Sequence Models family (`sequence_models/`).

Both models consume a SEQUENCE of recent MarketState vectors (not a
single snapshot, unlike Bayesian Logistic / Bagged GBM) — the spec's
rationale for including sequence models at all is that some structure
(e.g. a slow regime transition) is only visible across a short window
of consecutive states, not in any single one. `sequence_length` is the
number of trailing MarketState snapshots each forward pass consumes.

Uncertainty for both models comes from MC-dropout (Gal & Ghahramani,
2016): run `mc_dropout_samples` stochastic forward passes with dropout
left ACTIVE at inference time (not the usual eval-mode dropout-off),
and use the std-dev of the resulting predicted probabilities across
those samples as the uncertainty estimate — the same "disagreement
across independent stochastic estimates" principle
`probability/gbm.py` uses for its bagged ensemble, applied here via
dropout-induced stochasticity instead of bootstrap resampling, since a
single neural network doesn't have an analogous ensemble of members to
disagree with each other.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from state_encoder.types import DIMENSION_NAMES


class LSTMSequenceConfig(BaseModel):
    feature_dims: list[str] = Field(
        default=list(DIMENSION_NAMES),
        description="Which MarketState dimensions form each timestep's input vector.",
    )
    sequence_length: int = Field(default=20, description="Number of trailing MarketState snapshots per input sequence.")
    hidden_size: int = Field(default=32, description="LSTM hidden state dimensionality.")
    num_layers: int = Field(default=1, description="Number of stacked LSTM layers.")
    dropout: float = Field(default=0.2, description="Dropout applied between LSTM layers and before the output head.")
    learning_rate: float = Field(default=1e-3)
    weight_decay: float = Field(default=1e-5, description="L2 regularization on the optimizer.")
    epochs: int = Field(default=30)
    batch_size: int = Field(default=32)
    mc_dropout_samples: int = Field(
        default=20, description="Number of stochastic forward passes (dropout active) averaged for prediction and used for uncertainty."
    )
    random_seed: int = 42
    device: str = Field(default="cpu", description="'cpu' or 'cuda' — 'cuda' silently falls back to 'cpu' if unavailable.")

    @field_validator("feature_dims")
    @classmethod
    def _dims_must_exist(cls, v: list[str]) -> list[str]:
        for dim in v:
            if dim not in DIMENSION_NAMES:
                raise ValueError(f"'{dim}' is not a valid MarketState dimension. Valid: {DIMENSION_NAMES}")
        if len(v) == 0:
            raise ValueError("feature_dims must not be empty")
        return v

    @field_validator("sequence_length", "hidden_size", "num_layers", "epochs", "batch_size", "mc_dropout_samples")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be a positive integer")
        return v

    @field_validator("dropout")
    @classmethod
    def _dropout_bounds(cls, v: float) -> float:
        if not (0.0 <= v < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        return v


class TransformerSequenceConfig(BaseModel):
    feature_dims: list[str] = Field(default=list(DIMENSION_NAMES))
    sequence_length: int = Field(default=20)
    d_model: int = Field(default=32, description="Transformer embedding dimensionality.")
    n_heads: int = Field(default=4, description="Number of self-attention heads. Must divide d_model evenly.")
    num_layers: int = Field(default=2, description="Number of stacked TransformerEncoderLayers.")
    dim_feedforward: int = Field(default=64)
    dropout: float = Field(default=0.2)
    learning_rate: float = Field(default=1e-3)
    weight_decay: float = Field(default=1e-5)
    epochs: int = Field(default=30)
    batch_size: int = Field(default=32)
    mc_dropout_samples: int = Field(default=20)
    random_seed: int = 42
    device: str = Field(default="cpu")

    @field_validator("feature_dims")
    @classmethod
    def _dims_must_exist(cls, v: list[str]) -> list[str]:
        for dim in v:
            if dim not in DIMENSION_NAMES:
                raise ValueError(f"'{dim}' is not a valid MarketState dimension. Valid: {DIMENSION_NAMES}")
        if len(v) == 0:
            raise ValueError("feature_dims must not be empty")
        return v

    @field_validator("sequence_length", "d_model", "n_heads", "num_layers", "dim_feedforward", "epochs", "batch_size", "mc_dropout_samples")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be a positive integer")
        return v

    @field_validator("dropout")
    @classmethod
    def _dropout_bounds(cls, v: float) -> float:
        if not (0.0 <= v < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        return v

    @field_validator("d_model")
    @classmethod
    def _d_model_divisible_placeholder(cls, v: int) -> int:
        return v  # cross-field check (n_heads divides d_model) done below

    @model_validator(mode="after")
    def _heads_divide_d_model(self) -> "TransformerSequenceConfig":
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be evenly divisible by n_heads ({self.n_heads})"
            )
        return self


class SequenceModelsConfig(BaseModel):
    lstm: LSTMSequenceConfig = LSTMSequenceConfig()
    transformer: TransformerSequenceConfig = TransformerSequenceConfig()

    def model_post_init(self, __context) -> None:  # pydantic v2 hook
        if self.transformer.d_model % self.transformer.n_heads != 0:
            raise ValueError(
                f"transformer.d_model ({self.transformer.d_model}) must be evenly divisible by "
                f"transformer.n_heads ({self.transformer.n_heads})"
            )
