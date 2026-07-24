"""
Transformer Encoder Sequence Classifier — Level 2 candidate model
(sequence family).

Architecture: a small stack of `nn.TransformerEncoderLayer`s with
learned positional embeddings (a window of `sequence_length` MarketState
vectors has no natural notion of "position" the way word-order does for
text, but the relative recency of each timestep is still informative,
so a learned positional embedding — rather than assuming none is
needed — is added to each timestep's projected input before the
encoder stack). Sequence-level representation is mean-pooled across the
time dimension (simpler and, for short sequences like this, no less
effective than a [CLS]-token scheme, which would need an extra learned
token this problem's short, fixed-length windows don't obviously
benefit from) before the classification head.

Same MC-dropout uncertainty method as the LSTM sibling — see
`sequence_models/lstm_model.py` and
`configs/sequence_models_schema.py`'s module docstrings.
"""

from __future__ import annotations

import logging

import numpy as np

from configs.sequence_models_schema import TransformerSequenceConfig
from probability.types import ProbabilityEstimate
from sequence_models._torch_optional import TORCH_AVAILABLE, nn, require_torch, torch
from sequence_models.dataset import build_sequences
from state_encoder.types import MarketState

logger = logging.getLogger("sequence_models.transformer")
NAN = float("nan")


if TORCH_AVAILABLE:

    class _TransformerNet(nn.Module):
        def __init__(
            self, n_features: int, sequence_length: int, d_model: int, n_heads: int,
            num_layers: int, dim_feedforward: int, dropout: float,
        ) -> None:
            super().__init__()
            self.input_projection = nn.Linear(n_features, d_model)
            self.positional_embedding = nn.Parameter(torch.zeros(1, sequence_length, d_model))
            nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            projected = self.input_projection(x) + self.positional_embedding
            encoded = self.encoder(projected)          # (batch, seq_len, d_model)
            pooled = encoded.mean(dim=1)                # mean-pool across time
            return self.head(self.dropout(pooled)).squeeze(-1)  # (batch,) logits


class TransformerSequenceEstimator:
    name = "transformer_sequence"

    def __init__(self, config: TransformerSequenceConfig) -> None:
        self._config = config
        self._model = None
        self._history_by_symbol: dict[str, list[MarketState]] = {}

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def _device(self) -> "torch.device":
        if self._config.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def fit(self, sequences: np.ndarray, labels: np.ndarray) -> "TransformerSequenceEstimator":
        """See `LSTMSequenceEstimator.fit` docstring — same contract:
        pre-built (n, sequence_length, n_features) tensors and (n,)
        binary labels, typically from
        `sequence_models.dataset.build_sequences_and_labels`."""
        require_torch()
        c = self._config
        if sequences.ndim != 3 or sequences.shape[1] != c.sequence_length or sequences.shape[2] != len(c.feature_dims):
            raise ValueError(
                f"sequences must have shape [n, {c.sequence_length}, {len(c.feature_dims)}], got {sequences.shape}"
            )
        if len(sequences) != len(labels):
            raise ValueError("sequences and labels must have the same length")
        if not np.all(np.isin(labels, [0, 1])):
            raise ValueError("labels must be binary (0/1)")
        if len(sequences) < 10:
            raise ValueError("Need at least 10 training sequences to fit a meaningful model")

        torch.manual_seed(c.random_seed)
        device = self._device()
        model = _TransformerNet(
            len(c.feature_dims), c.sequence_length, c.d_model, c.n_heads,
            c.num_layers, c.dim_feedforward, c.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        X = torch.tensor(sequences, dtype=torch.float32, device=device)
        y = torch.tensor(labels, dtype=torch.float32, device=device)
        n = len(X)

        model.train()
        rng = np.random.default_rng(c.random_seed)
        for epoch in range(c.epochs):
            permutation = rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, c.batch_size):
                batch_idx = permutation[start : start + c.batch_size]
                batch_idx_t = torch.tensor(batch_idx, dtype=torch.long, device=device)
                xb, yb = X[batch_idx_t], y[batch_idx_t]

                optimizer.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(batch_idx)
            if (epoch + 1) % max(1, c.epochs // 5) == 0:
                logger.info("Transformer epoch %d/%d — mean loss %.5f", epoch + 1, c.epochs, epoch_loss / n)

        self._model = model
        self._history_by_symbol = {}
        return self

    def observe(self, state: MarketState) -> None:
        buf = self._history_by_symbol.setdefault(state.symbol, [])
        buf.append(state)
        max_len = self._config.sequence_length * 3
        if len(buf) > max_len:
            del buf[: len(buf) - max_len]

    def predict(self, state: MarketState) -> ProbabilityEstimate:
        if self._model is None:
            raise RuntimeError("TransformerSequenceEstimator.predict() called before fit().")

        self.observe(state)
        history = self._history_by_symbol[state.symbol]

        if len(history) < self._config.sequence_length or not state.is_valid:
            return ProbabilityEstimate(
                symbol=state.symbol, epoch=state.epoch, model_name=self.name,
                prob_up=NAN, prob_down=NAN, uncertainty=NAN, expected_direction=0, confidence=NAN,
            )

        window = history[-self._config.sequence_length :]
        sequence = build_sequences(window, self._config.feature_dims, self._config.sequence_length)
        device = self._device()
        x = torch.tensor(sequence, dtype=torch.float32, device=device)

        self._model.train()  # keep dropout ACTIVE for MC-dropout — see module docstring
        probs = []
        with torch.no_grad():
            for _ in range(self._config.mc_dropout_samples):
                logits = self._model(x)
                probs.append(torch.sigmoid(logits).item())
        self._model.eval()

        probs_arr = np.array(probs)
        prob_up = float(np.mean(probs_arr))
        uncertainty = float(np.std(probs_arr))
        prob_down = 1.0 - prob_up
        expected_direction = 1 if prob_up > 0.5 else (-1 if prob_up < 0.5 else 0)
        confidence = max(prob_up, prob_down)

        return ProbabilityEstimate(
            symbol=state.symbol, epoch=state.epoch, model_name=self.name,
            prob_up=prob_up, prob_down=prob_down, uncertainty=uncertainty,
            expected_direction=expected_direction, confidence=confidence,
        )
