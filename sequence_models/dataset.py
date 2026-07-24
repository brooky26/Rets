"""
Sequence-building helpers, shared by both the LSTM and Transformer
models. Kept as pure functions (mirroring `features/math_utils.py`'s
pure-function-core pattern) so both models build their training
sequences identically and offline batch code can call this directly
without any model instance.
"""

from __future__ import annotations

import numpy as np

from state_encoder.types import MarketState


def build_sequences(
    states: list[MarketState], feature_dims: list[str], sequence_length: int
) -> np.ndarray:
    """
    Given a chronological list of MarketState, returns a
    (n_sequences, sequence_length, n_features) array where sequence i is
    states[i : i + sequence_length]. n_sequences = len(states) -
    sequence_length + 1. Raises if fewer than `sequence_length` states
    are given — there is no meaningful sequence to build otherwise.
    """
    if len(states) < sequence_length:
        raise ValueError(
            f"Need at least {sequence_length} states to build one sequence, got {len(states)}"
        )
    matrix = np.array(
        [[getattr(s, dim) for dim in feature_dims] for s in states], dtype=np.float64
    )
    n_sequences = len(states) - sequence_length + 1
    n_features = len(feature_dims)
    out = np.empty((n_sequences, sequence_length, n_features), dtype=np.float32)
    for i in range(n_sequences):
        out[i] = matrix[i : i + sequence_length]
    return out


def build_sequences_and_labels(
    states: list[MarketState], closes: np.ndarray, feature_dims: list[str], sequence_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Supervised framing: sequence i (states[i : i+sequence_length]) is
    labeled with whether the candle immediately AFTER the sequence's
    last state closed higher than the sequence's last close — i.e. "does
    the next tick go up, given the last `sequence_length` states."
    `closes` must be aligned 1:1 with `states` (same convention as
    `paper_trading.orchestrator.bootstrap` and `backtesting.walk_forward`).

    The final state has no "next candle" to label against, so it (and
    any sequence that would end on it) is excluded — this trims one
    fewer usable sequence than `build_sequences` would produce from the
    same input.
    """
    if len(states) != len(closes):
        raise ValueError("states and closes must be the same length")
    if len(states) < sequence_length + 1:
        raise ValueError(
            f"Need at least {sequence_length + 1} (state, close) pairs to build one labeled "
            f"sequence, got {len(states)}"
        )

    usable_states = states[:-1]
    labels_full = (np.diff(np.asarray(closes)) > 0).astype(np.int64)

    sequences = build_sequences(usable_states, feature_dims, sequence_length)
    # sequence i covers usable_states[i : i+sequence_length]; its label is
    # whether usable_states[i+sequence_length-1] -> the candle after it went up,
    # which is labels_full[i + sequence_length - 1].
    labels = labels_full[sequence_length - 1 :]
    return sequences, labels
