"""
LSTM Sequence Classifier — Level 2 candidate model (sequence family).

Architecture: a stacked LSTM consumes a (sequence_length, n_features)
window of MarketState vectors, and the final hidden state feeds a small
linear head producing a single logit for "next tick is up."

Uncertainty via MC-dropout (see configs/sequence_models_schema.py's
module docstring for the method): `predict()` runs
`config.mc_dropout_samples` stochastic forward passes with dropout left
ACTIVE (`model.train()`, not `model.eval()`), and uses the mean of the
resulting sigmoid outputs as `prob_up` and their std-dev as
`uncertainty` — genuinely epistemic (it reflects which dropout masks
the network could have used, standing in for "how much would this
prediction change under slightly different sampled sub-networks") in
the same spirit as the Bayesian Logistic model's posterior-variance
uncertainty and the Bagged GBM's cross-member disagreement, just
sourced from stochastic dropout instead of a closed-form posterior or
bootstrap ensemble.
"""

from __future__ import annotations

import logging

import numpy as np

from configs.sequence_models_schema import LSTMSequenceConfig
from probability.types import ProbabilityEstimate
from sequence_models._torch_optional import TORCH_AVAILABLE, nn, require_torch, torch
from sequence_models.dataset import build_sequences
from state_encoder.types import MarketState

logger = logging.getLogger("sequence_models.lstm")
NAN = float("nan")


if TORCH_AVAILABLE:

    class _LSTMNet(nn.Module):
        def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            _, (h_n, _) = self.lstm(x)
            last_hidden = h_n[-1]  # (batch, hidden_size) — final layer's hidden state
            return self.head(self.dropout(last_hidden)).squeeze(-1)  # (batch,) logits


class LSTMSequenceEstimator:
    name = "lstm_sequence"

    def __init__(self, config: LSTMSequenceConfig) -> None:
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

    def fit(self, sequences: np.ndarray, labels: np.ndarray) -> "LSTMSequenceEstimator":
        """
        `sequences` (n, sequence_length, n_features) and `labels` (n,
        binary) are expected to come from
        `sequence_models.dataset.build_sequences_and_labels` — this
        class trains on already-built sequence tensors rather than raw
        MarketState lists, mirroring how `BayesianLogisticRegression.fit`
        and `BaggedGBMEstimator.fit` both take an already-built feature
        matrix rather than doing their own feature extraction.
        """
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
        model = _LSTMNet(len(c.feature_dims), c.hidden_size, c.num_layers, c.dropout).to(device)
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
                logger.info("LSTM epoch %d/%d — mean loss %.5f", epoch + 1, c.epochs, epoch_loss / n)

        self._model = model
        self._history_by_symbol = {}
        return self

    def observe(self, state: MarketState) -> None:
        """Feed one new live MarketState into this symbol's rolling history
        buffer, so `predict()` can be called with just the latest state
        rather than requiring the caller to manage a window itself."""
        buf = self._history_by_symbol.setdefault(state.symbol, [])
        buf.append(state)
        max_len = self._config.sequence_length * 3  # keep some slack, not just the bare minimum
        if len(buf) > max_len:
            del buf[: len(buf) - max_len]

    def predict(self, state: MarketState) -> ProbabilityEstimate:
        if self._model is None:
            raise RuntimeError("LSTMSequenceEstimator.predict() called before fit().")

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
