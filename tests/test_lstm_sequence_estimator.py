import numpy as np
import pytest

from configs.sequence_models_schema import LSTMSequenceConfig
from sequence_models.dataset import build_sequences, build_sequences_and_labels
from state_encoder.types import MarketState

torch = pytest.importorskip("torch")
from sequence_models.lstm_model import LSTMSequenceEstimator  # noqa: E402


def make_state(symbol: str, epoch: int, trend: float) -> MarketState:
    return MarketState(
        symbol=symbol, epoch=epoch, trend=trend, momentum=0.0, acceleration=0.0, volatility=0.1,
        noise=0.0, persistence=0.0, compression_expansion=0.0, complexity=0.0, uncertainty=0.1,
        liquidity=0.0, market_phase=0.0,
    )


def make_config(**overrides) -> LSTMSequenceConfig:
    defaults = dict(
        feature_dims=["trend", "volatility"], sequence_length=5, hidden_size=8, num_layers=1,
        dropout=0.1, epochs=3, batch_size=8, mc_dropout_samples=5, random_seed=0,
    )
    defaults.update(overrides)
    return LSTMSequenceConfig(**defaults)


def _make_synthetic_series(n: int, seed: int) -> tuple[list[MarketState], np.ndarray]:
    rng = np.random.default_rng(seed)
    trends = rng.uniform(-1, 1, size=n)
    states = [make_state("TEST", i, trends[i]) for i in range(n)]
    # Closes trend upward whenever `trend` is positive — a learnable signal.
    closes = np.cumsum(np.where(trends > 0, 1.0, -1.0)) + 100.0
    return states, closes


def test_build_sequences_and_labels_shapes():
    states, closes = _make_synthetic_series(50, seed=1)
    sequences, labels = build_sequences_and_labels(states, closes, ["trend", "volatility"], sequence_length=5)
    assert sequences.shape == (50 - 5, 5, 2)
    assert labels.shape == (50 - 5,)
    assert set(np.unique(labels)).issubset({0, 1})


def test_lstm_fit_and_predict_smoke():
    states, closes = _make_synthetic_series(80, seed=2)
    config = make_config()
    sequences, labels = build_sequences_and_labels(states, closes, config.feature_dims, config.sequence_length)

    model = LSTMSequenceEstimator(config)
    model.fit(sequences, labels)
    assert model.is_fitted

    # Feed the same history through observe() so predict() has enough
    # buffered context, then predict on the final state.
    for s in states[:-1]:
        model.observe(s)
    result = model.predict(states[-1])
    assert result.is_valid
    assert 0.0 <= result.prob_up <= 1.0
    assert result.uncertainty >= 0.0
    assert result.expected_direction in (-1, 0, 1)


def test_lstm_predict_before_fit_raises():
    config = make_config()
    model = LSTMSequenceEstimator(config)
    with pytest.raises(RuntimeError):
        model.predict(make_state("TEST", 0, 0.1))


def test_lstm_predict_insufficient_history_returns_invalid():
    states, closes = _make_synthetic_series(80, seed=3)
    config = make_config()
    sequences, labels = build_sequences_and_labels(states, closes, config.feature_dims, config.sequence_length)
    model = LSTMSequenceEstimator(config)
    model.fit(sequences, labels)

    result = model.predict(make_state("NEWSYMBOL", 0, 0.1))
    assert not result.is_valid
