"""
Platt Scaling — probability calibration.

Raw outputs from any classifier (the Bayesian model's mu_a, the GBM
ensemble's mean probability, whatever) are not necessarily *calibrated*
— a model saying "70% confident" doesn't automatically mean it's right
70% of the time historically. Platt scaling fits a 1D logistic
regression mapping raw scores to calibrated probabilities:

    p_calibrated(s) = sigmoid(A * s + B)

fit by maximizing the log-likelihood of (s_i, y_i) pairs via Newton's
method (a 2-parameter special case of the same IRLS machinery used in
BayesianLogisticRegression, but kept as its own small, dependency-free
implementation since it operates on a different kind of input — raw
model scores, not MarketState features — and is meant to wrap *any*
upstream estimator, not just the Bayesian one).

This is deliberately a post-hoc wrapper: it doesn't change what a model
predicts, only how its raw output is mapped onto a probability that
should now match observed frequencies more closely, particularly useful
for models like the GBM ensemble whose raw averaged probabilities aren't
guaranteed to be well-calibrated the way a proper Bayesian posterior is.
"""

from __future__ import annotations

import numpy as np

from configs.probability_schema import CalibrationConfig


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


class PlattCalibrator:
    def __init__(self, config: CalibrationConfig) -> None:
        self._config = config
        self.A: float | None = None
        self.B: float | None = None
        self.converged: bool = False

    @property
    def is_fitted(self) -> bool:
        return self.A is not None

    def fit(self, raw_scores: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        if len(raw_scores) != len(y):
            raise ValueError("raw_scores and y must have the same length")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must be binary (0/1)")
        if len(raw_scores) < 5:
            raise ValueError("Need at least 5 observations to fit Platt scaling")

        s = np.asarray(raw_scores, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        A, B = 0.0, 0.0
        self.converged = False
        for _ in range(self._config.max_iterations):
            z = A * s + B
            p = np.clip(_sigmoid(z), 1e-10, 1 - 1e-10)

            grad_A = np.sum((y - p) * s)
            grad_B = np.sum(y - p)

            w = p * (1 - p)
            H_AA = -np.sum(w * s * s)
            H_AB = -np.sum(w * s)
            H_BB = -np.sum(w)
            H = np.array([[H_AA, H_AB], [H_AB, H_BB]])
            g = np.array([grad_A, grad_B])

            try:
                step = np.linalg.solve(H, g)  # Newton step: [A,B] -= H^-1 g, H negative-definite
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, g, rcond=None)[0]

            A_new, B_new = A - step[0], B - step[1]
            if abs(A_new - A) + abs(B_new - B) < self._config.tolerance:
                A, B = A_new, B_new
                self.converged = True
                break
            A, B = A_new, B_new

        self.A, self.B = float(A), float(B)
        return self

    def transform(self, raw_score: float) -> float:
        if self.A is None or self.B is None:
            raise RuntimeError("PlattCalibrator.transform() called before fit().")
        z = np.array([self.A * raw_score + self.B])
        return float(_sigmoid(z)[0])
