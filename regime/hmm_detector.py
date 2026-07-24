"""
Gaussian HMM Regime Detector.

Wraps the raw `GaussianHMM` (regime/hmm.py) with:
  1. Offline training (`fit`) on historical MarketState vectors.
  2. Hidden-state -> RegimeLabel mapping, derived automatically from each
     fitted state's mean vector (interpreting the same dimensions the
     RuleBasedRegimeDetector uses thresholds on) — so hidden state 2
     being "the state where trend is high and persistence is high" gets
     labeled STRONG_TREND, not left as an opaque integer.
  3. Live, causal classification via the incremental forward step —
     `classify()` never looks at future observations, and maintains one
     running forward-filter (`alpha_hat`) per symbol so each new
     MarketState is an O(n_states^2) update, not a full-sequence replay.

This detector is a challenger, not the default: it needs `fit()` called
with real historical data before it can be used (`classify()` raises if
called unfit). Until it's trained, validated, and proven statistically
superior to the rule-based baseline via a proper Champion-Challenger
comparison (a later stage), the rule-based detector remains the one
actually driving anything downstream.
"""

from __future__ import annotations

import numpy as np

from configs.regime_schema import GaussianHMMConfig
from regime.hmm import GaussianHMM
from regime.types import RegimeClassification, RegimeLabel
from state_encoder.types import MarketState

NAN = float("nan")


class HMMNotFittedError(Exception):
    pass


class GaussianHMMRegimeDetector:
    name = "gaussian_hmm"

    def __init__(self, config: GaussianHMMConfig) -> None:
        self._config = config
        self._hmm: GaussianHMM | None = None
        self._state_labels: dict[int, RegimeLabel] | None = None
        self._alpha_by_symbol: dict[str, np.ndarray] = {}

    @property
    def is_fitted(self) -> bool:
        return self._hmm is not None

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def fit(self, historical_states: list[MarketState]) -> "GaussianHMMRegimeDetector":
        valid_states = [s for s in historical_states if s.is_valid]
        if len(valid_states) < self._config.n_states * 2:
            raise ValueError(
                f"Need at least {self._config.n_states * 2} valid MarketState observations "
                f"to fit {self._config.n_states} hidden states, got {len(valid_states)}."
            )

        X = self._to_observation_matrix(valid_states)
        hmm = GaussianHMM(
            n_states=self._config.n_states,
            n_features=len(self._config.observation_dims),
            min_variance=self._config.min_variance,
            random_seed=self._config.random_seed,
        )
        hmm.fit(
            X,
            max_iterations=self._config.em_max_iterations,
            tolerance=self._config.em_tolerance,
        )
        self._hmm = hmm
        self._state_labels = self._label_states(hmm)
        self._alpha_by_symbol = {}
        return self

    def _to_observation_matrix(self, states: list[MarketState]) -> np.ndarray:
        dims = self._config.observation_dims
        return np.array(
            [[getattr(s, dim) for dim in dims] for s in states], dtype=np.float64
        )

    def _label_states(self, hmm: GaussianHMM) -> dict[int, RegimeLabel]:
        """
        Map each hidden state index to the RegimeLabel whose defining
        characteristic best matches that state's fitted mean vector.
        Uses the same dimension semantics as the rule-based detector,
        applied to the HMM's learned cluster centers rather than live
        readings.
        """
        dims = self._config.observation_dims
        dim_index = {d: i for i, d in enumerate(dims)}

        def get(mean_vec: np.ndarray, dim: str, default: float = 0.0) -> float:
            idx = dim_index.get(dim)
            return float(mean_vec[idx]) if idx is not None else default

        labels: dict[int, RegimeLabel] = {}
        for k in range(hmm.n_states):
            mean_vec = hmm.means[k]
            trend = get(mean_vec, "trend")
            vol = get(mean_vec, "volatility")
            persistence = get(mean_vec, "persistence")
            comp_exp = get(mean_vec, "compression_expansion")

            if comp_exp > 0.5 and abs(trend) > 0.5:
                labels[k] = RegimeLabel.BREAKOUT
            elif abs(trend) > 0.6:
                labels[k] = RegimeLabel.STRONG_TREND
            elif abs(trend) > 0.25:
                labels[k] = RegimeLabel.WEAK_TREND
            elif persistence < -0.3:
                labels[k] = RegimeLabel.MEAN_REVERSION
            elif comp_exp < -0.5:
                labels[k] = RegimeLabel.COMPRESSION
            elif comp_exp > 0.5:
                labels[k] = RegimeLabel.EXPANSION
            elif vol > 0.6:
                labels[k] = RegimeLabel.HIGH_VOLATILITY
            elif vol < -0.6:
                labels[k] = RegimeLabel.LOW_VOLATILITY
            elif abs(persistence) < 0.15 and abs(trend) < 0.25:
                labels[k] = RegimeLabel.RANDOM_WALK
            else:
                labels[k] = RegimeLabel.RANGE
        return labels

    # ------------------------------------------------------------------ #
    # Live classification (causal)
    # ------------------------------------------------------------------ #

    def classify(self, state: MarketState) -> RegimeClassification:
        if self._hmm is None or self._state_labels is None:
            raise HMMNotFittedError(
                "GaussianHMMRegimeDetector.classify() called before fit(). "
                "Use RuleBasedRegimeDetector until this has been trained on "
                "sufficient historical data."
            )

        if not state.is_valid:
            return RegimeClassification(
                symbol=state.symbol,
                epoch=state.epoch,
                detector_name=self.name,
                regime=RegimeLabel.RANGE,
                confidence=NAN,
                probabilities={},
            )

        x_t = np.array(
            [getattr(state, dim) for dim in self._config.observation_dims], dtype=np.float64
        )
        prev_alpha = self._alpha_by_symbol.get(state.symbol)
        new_alpha, _ = self._hmm.forward_step(prev_alpha, x_t)
        self._alpha_by_symbol[state.symbol] = new_alpha

        # Aggregate hidden-state probabilities into label probabilities
        # (multiple hidden states can map to the same semantic label).
        label_probs: dict[RegimeLabel, float] = {label: 0.0 for label in RegimeLabel}
        for k, prob in enumerate(new_alpha):
            label_probs[self._state_labels[k]] += float(prob)

        best_label = max(label_probs, key=label_probs.get)
        confidence = label_probs[best_label]

        return RegimeClassification(
            symbol=state.symbol,
            epoch=state.epoch,
            detector_name=self.name,
            regime=best_label,
            confidence=confidence,
            probabilities=label_probs,
        )

    def reset_symbol(self, symbol: str) -> None:
        """Clear the running forward-filter state for a symbol (e.g. after a data gap)."""
        self._alpha_by_symbol.pop(symbol, None)
