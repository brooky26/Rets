"""
Bagged Gradient Boosting Ensemble.

The spec's candidate model list includes XGBoost / LightGBM / CatBoost.
This uses scikit-learn's HistGradientBoostingClassifier as the tree
learner (lighter dependency footprint, same core algorithm family —
histogram-based gradient boosting) — the model is swappable behind the
same interface if a specific boosting library is preferred later; nothing
downstream depends on which one is used.

Uncertainty here comes from a genuinely different source than the
Bayesian model: `n_ensemble_members` independently bootstrap-resampled
models are trained, each seeing a different random subset of the
training data. Their predictions naturally disagree more in regions of
feature space that are sparsely covered or inherently noisy, and agree
more where the signal is strong and well-represented — that
cross-member disagreement (std of predicted probabilities across the
ensemble) *is* the uncertainty estimate, in the same spirit as the
spec's "Model agreement" component of the Confidence Engine, just
applied within a single model family here rather than across model
families.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from configs.probability_schema import BaggedGBMConfig
from probability.types import ProbabilityEstimate
from state_encoder.types import MarketState

NAN = float("nan")


class BaggedGBMEstimator:
    name = "bagged_gbm"

    def __init__(self, config: BaggedGBMConfig) -> None:
        self._config = config
        self._models: list[HistGradientBoostingClassifier] = []
        self._rng = np.random.default_rng(config.random_seed)

    @property
    def is_fitted(self) -> bool:
        return len(self._models) > 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaggedGBMEstimator":
        if X.ndim != 2 or X.shape[1] != len(self._config.feature_dims):
            raise ValueError(
                f"X must have shape [n, {len(self._config.feature_dims)}], got {X.shape}"
            )
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must be binary (0/1)")

        n = len(X)
        sample_size = max(10, int(n * self._config.bootstrap_fraction))
        if n < 20:
            raise ValueError("Need at least 20 observations to fit a bagged GBM ensemble")

        self._models = []
        for member_idx in range(self._config.n_ensemble_members):
            member_seed = int(self._rng.integers(0, 2**31 - 1))
            indices = self._rng.integers(0, n, size=sample_size)
            X_boot, y_boot = X[indices], y[indices]

            # A bootstrap resample can occasionally be single-class by
            # chance (especially with small n or imbalanced labels) — skip
            # such a member rather than let sklearn raise, since it would
            # never contribute a meaningful probability anyway.
            if len(np.unique(y_boot)) < 2:
                continue

            model = HistGradientBoostingClassifier(
                max_iter=self._config.max_boosting_iterations,
                learning_rate=self._config.learning_rate,
                max_depth=self._config.max_depth,
                random_state=member_seed,
            )
            model.fit(X_boot, y_boot)
            self._models.append(model)

        if len(self._models) < 2:
            raise ValueError(
                "Fewer than 2 ensemble members trained successfully (likely due to "
                "single-class bootstrap resamples) — cannot estimate cross-member "
                "disagreement. Provide more/better-balanced training data."
            )
        return self

    def predict(self, state: MarketState) -> ProbabilityEstimate:
        if not self.is_fitted:
            raise RuntimeError("BaggedGBMEstimator.predict() called before fit().")

        if not state.is_valid:
            return ProbabilityEstimate(
                symbol=state.symbol,
                epoch=state.epoch,
                model_name=self.name,
                prob_up=NAN,
                prob_down=NAN,
                uncertainty=NAN,
                expected_direction=0,
                confidence=NAN,
            )

        x = np.array(
            [[getattr(state, dim) for dim in self._config.feature_dims]], dtype=np.float64
        )
        member_probs = np.array(
            [model.predict_proba(x)[0, 1] for model in self._models]
        )

        prob_up = float(np.mean(member_probs))
        prob_down = 1.0 - prob_up
        uncertainty = float(np.std(member_probs))  # cross-member disagreement

        expected_direction = 1 if prob_up > 0.5 else (-1 if prob_up < 0.5 else 0)
        confidence = max(prob_up, prob_down)

        return ProbabilityEstimate(
            symbol=state.symbol,
            epoch=state.epoch,
            model_name=self.name,
            prob_up=prob_up,
            prob_down=prob_down,
            uncertainty=uncertainty,
            expected_direction=expected_direction,
            confidence=confidence,
        )
