"""
Bayesian Logistic Regression — Laplace approximation.

Model
-----
    p(y=1 | x, w) = sigmoid(w^T x)
    prior:  w ~ N(0, alpha^-1 I)          (Gaussian shrinkage prior)

MAP estimation via Newton-Raphson (IRLS)
------------------------------------------
Maximize the log posterior:
    L(w) = sum_i [y_i log p_i + (1-y_i) log(1-p_i)] - 0.5 * alpha * ||w||^2

Gradient:   g(w) = X^T (y - p) - alpha * w
Hessian:    H(w) = -X^T S X - alpha * I,   S = diag(p_i * (1 - p_i))
Newton step: w_new = w - H^-1 g = w + (X^T S X + alpha I)^-1 g

Iterate to convergence -> w_map.

Laplace approximation to the posterior
-----------------------------------------
Approximate p(w | D) ~ N(w_map, Sigma) where
    Sigma = (X^T S X + alpha I)^-1     evaluated at w_map

This Sigma is exactly the inverse of the negative Hessian at the mode —
the standard Laplace/Gaussian approximation to a log-concave posterior.

Predictive distribution (MacKay's probit approximation)
------------------------------------------------------------
For a new point x*, the "activation" a = w^T x* is approximately Gaussian
under the posterior:
    mu_a    = w_map^T x*
    sigma_a^2 = x*^T Sigma x*

The exact predictive probability requires integrating sigmoid(a) against
N(mu_a, sigma_a^2), which has no closed form. MacKay's approximation
(itself a well-known, standard result) uses the probit-sigmoid similarity:
    p(y=1 | x*) ~= sigmoid( mu_a / sqrt(1 + pi * sigma_a^2 / 8) )

This is what actually delivers the spec's "Prediction Uncertainty" output
as something distinct from p(1-p): sigma_a^2 is genuine epistemic
uncertainty from having a finite, specific training set — it grows for
points far from the training data in feature space (large x*^T Sigma x*),
even when the point estimate mu_a is confident.
"""

from __future__ import annotations

import logging

import numpy as np

from configs.probability_schema import BayesianLogisticConfig
from probability.types import ProbabilityEstimate
from state_encoder.types import MarketState

logger = logging.getLogger("probability.bayesian_logistic")
NAN = float("nan")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid.
    out = np.empty_like(z, dtype=np.float64)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


class BayesianLogisticRegression:
    name = "bayesian_logistic"

    def __init__(self, config: BayesianLogisticConfig) -> None:
        self._config = config
        self.w_map: np.ndarray | None = None
        self.Sigma: np.ndarray | None = None
        self.converged: bool = False
        self.n_iterations_run: int = 0

    @property
    def is_fitted(self) -> bool:
        return self.w_map is not None

    def _augment(self, X: np.ndarray) -> np.ndarray:
        if self._config.include_intercept:
            ones = np.ones((len(X), 1))
            return np.hstack([ones, X])
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianLogisticRegression":
        if X.ndim != 2 or X.shape[1] != len(self._config.feature_dims):
            raise ValueError(
                f"X must have shape [n, {len(self._config.feature_dims)}], got {X.shape}"
            )
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must be binary (0/1)")
        if len(X) < 5:
            raise ValueError("Need at least 5 observations to fit a meaningful model")

        Xa = self._augment(X)
        n, d = Xa.shape
        alpha = self._config.prior_precision
        w = np.zeros(d)

        self.converged = False
        for iteration in range(self._config.max_iterations):
            z = Xa @ w
            p = _sigmoid(z)
            p = np.clip(p, 1e-10, 1 - 1e-10)  # avoid exact 0/1 -> singular S
            S = p * (1 - p)

            grad = Xa.T @ (y - p) - alpha * w
            A = (Xa * S[:, None]).T @ Xa + alpha * np.eye(d)

            try:
                step = np.linalg.solve(A, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(A, grad, rcond=None)[0]

            w_new = w + step
            self.n_iterations_run = iteration + 1

            if np.linalg.norm(step) < self._config.tolerance:
                w = w_new
                self.converged = True
                break
            w = w_new

        # Final posterior covariance at the converged (or last) w.
        z = Xa @ w
        p = np.clip(_sigmoid(z), 1e-10, 1 - 1e-10)
        S = p * (1 - p)
        A = (Xa * S[:, None]).T @ Xa + alpha * np.eye(d)
        self.Sigma = np.linalg.inv(A)
        self.w_map = w
        return self

    def predict(self, state: MarketState) -> ProbabilityEstimate:
        if self.w_map is None or self.Sigma is None:
            raise RuntimeError(
                "BayesianLogisticRegression.predict() called before fit()."
            )

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

        x = np.array([getattr(state, dim) for dim in self._config.feature_dims], dtype=np.float64)
        xa = np.concatenate([[1.0], x]) if self._config.include_intercept else x

        mu_a = float(xa @ self.w_map)
        sigma_a_sq = float(xa @ self.Sigma @ xa)
        sigma_a_sq = max(sigma_a_sq, 0.0)  # guard tiny negative from float error

        # MacKay's probit approximation.
        kappa = 1.0 / np.sqrt(1.0 + np.pi * sigma_a_sq / 8.0)
        raw_prob_up = float(_sigmoid(np.array([mu_a * kappa]))[0])
        # `fit()` clips p away from exact 0/1 during training (to avoid a
        # singular Hessian) but that guard doesn't cover predict()'s own
        # final sigmoid call. If a live feature value sits far outside the
        # training distribution, mu_a*kappa can be large enough that the
        # sigmoid underflows to exactly 0.0 or 1.0 in float64 (exp(-z)
        # underflows once |z| exceeds ~37) — a confidence of exactly 1.00
        # is not a genuine calibrated signal, it's a numerical artifact,
        # and treating it as real would let one out-of-distribution point
        # dominate every downstream EV/risk/opportunity calculation. Clamp
        # it the same way fit() does, and log when clamping actually fired
        # so a spike in this warning is visible rather than silently eaten.
        prob_up = float(np.clip(raw_prob_up, 1e-6, 1 - 1e-6))
        if prob_up != raw_prob_up:
            logger.warning(
                "%s: predict() sigmoid saturated (mu_a=%.3f, kappa=%.3f, raw_prob_up=%.10f) "
                "— clamped to %.6f. Recurring warnings for this symbol indicate the model "
                "is extrapolating far outside its training distribution and may need "
                "refitting on more recent data.",
                state.symbol, mu_a, kappa, raw_prob_up, prob_up,
            )
        prob_down = 1.0 - prob_up

        # Normalize sigma_a (an unbounded std) into a roughly [0,1) uncertainty score.
        sigma_a = np.sqrt(sigma_a_sq)
        uncertainty = float(sigma_a / (1.0 + sigma_a))

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
