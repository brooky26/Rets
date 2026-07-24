"""
Gaussian Hidden Markov Model — regime detection core.

Diagonal-covariance Gaussian HMM: each hidden state k has a per-dimension
mean mu_k and variance sigma_k^2, with dimensions assumed conditionally
independent given the state (standard simplification — full covariance
needs far more data to estimate reliably than a live trading system
accumulates quickly).

Model
-----
  pi        : initial state distribution, pi_i = P(state_1 = i)
  A         : transition matrix, A[i,j] = P(state_{t+1}=j | state_t=i)
  mu, var   : emission Gaussian parameters per state, per observation dim
  b_i(x)    : emission probability = prod_d N(x_d; mu[i,d], var[i,d])

Training — Baum-Welch (EM), with Rabiner's scaling for numerical stability
---------------------------------------------------------------------------
Forward pass (scaled):
    alpha_1(i)      = pi_i * b_i(x_1)
    c_1             = 1 / sum_i alpha_1(i)
    alpha_hat_1(i)  = c_1 * alpha_1(i)
    alpha_t(i)      = b_i(x_t) * sum_j alpha_hat_{t-1}(j) * A(j,i)      for t > 1
    c_t             = 1 / sum_i alpha_t(i)
    alpha_hat_t(i)  = c_t * alpha_t(i)
    log-likelihood  = -sum_t log(c_t)

Backward pass (scaled with the same c_t sequence):
    beta_hat_T(i)     = 1
    beta_t(i)         = sum_j A(i,j) * b_j(x_{t+1}) * beta_hat_{t+1}(j)
    beta_hat_t(i)     = c_{t+1} * beta_t(i)

E-step:
    gamma_t(i)   = alpha_hat_t(i) * beta_hat_t(i) / sum_i(...)          (renormalized)
    xi_t(i,j)    = alpha_hat_t(i) * A(i,j) * b_j(x_{t+1}) * beta_hat_{t+1}(j) / sum(...)

M-step:
    pi_i         = gamma_1(i)
    A(i,j)       = sum_t xi_t(i,j) / sum_t gamma_t(i)          (t = 1..T-1)
    mu_k         = sum_t gamma_t(k) x_t / sum_t gamma_t(k)
    sigma_k^2    = sum_t gamma_t(k) (x_t - mu_k)^2 / sum_t gamma_t(k),  floored at min_variance

Live / causal filtering
------------------------
The forward recursion above is inherently incremental: alpha_hat_t only
depends on alpha_hat_{t-1} and the new observation x_t. `forward_step()`
exposes exactly this one-step update, which is what a live system uses —
it never looks at future observations, so filtered probabilities computed
live are identical to filtered probabilities computed by replaying the
same sequence in a batch forward pass (verified in tests).

Smoothed posteriors (gamma, using the backward pass) use future
observations and are therefore for offline analysis / regime labeling
only — never for live decisions. This distinction is enforced by keeping
`forward_filter` (causal) and `smoothed_posteriors` (needs full sequence,
non-causal) as clearly separate methods.
"""

from __future__ import annotations

import numpy as np


class GaussianHMM:
    def __init__(
        self,
        n_states: int,
        n_features: int,
        min_variance: float = 1e-3,
        random_seed: int = 42,
    ) -> None:
        if n_states < 2:
            raise ValueError("n_states must be >= 2")
        if n_features < 1:
            raise ValueError("n_features must be >= 1")

        self.n_states = n_states
        self.n_features = n_features
        self.min_variance = min_variance
        self._rng = np.random.default_rng(random_seed)

        # Uniform initial guesses; fit() will overwrite these.
        self.pi = np.full(n_states, 1.0 / n_states)
        self.A = np.full((n_states, n_states), 1.0 / n_states)
        self.means = self._rng.normal(0, 0.5, size=(n_states, n_features))
        self.variances = np.full((n_states, n_features), 1.0)

        self.log_likelihood_history: list[float] = []
        self.converged: bool = False

    # ------------------------------------------------------------------ #
    # Emission model
    # ------------------------------------------------------------------ #

    def emission_probs(self, x: np.ndarray) -> np.ndarray:
        """b_i(x) for every state i, for a single observation x (shape [n_features])."""
        probs, _ = self._emission_probs_and_log_scale(x)
        return probs

    def _emission_probs_and_log_scale(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Returns (rescaled emission probs, log_scale) where log_scale is the
        max log-pdf subtracted before exponentiating (for numerical
        stability — without it, diagonal Gaussians on bounded [-1,1]
        features with small variances can underflow exp() to exactly 0
        for every state).

        The rescaling multiplies every state's b_i(x) by the same
        constant exp(-log_scale), which cancels out of every *relative*
        computation (filtering, Viterbi argmax) but must be added back
        via `log_scale` when accumulating a true log-likelihood — see
        `_forward_backward`.
        """
        var = np.maximum(self.variances, self.min_variance)
        log_norm = -0.5 * np.sum(np.log(2 * np.pi * var), axis=1)
        log_exp = -0.5 * np.sum(((x - self.means) ** 2) / var, axis=1)
        log_probs = log_norm + log_exp
        log_scale = float(np.max(log_probs))
        probs = np.exp(log_probs - log_scale)
        return probs, log_scale

    # ------------------------------------------------------------------ #
    # Batch scaled forward-backward
    # ------------------------------------------------------------------ #

    def _forward_backward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        T = len(X)
        n = self.n_states
        alpha_hat = np.zeros((T, n))
        c = np.zeros(T)
        log_scale_sum = 0.0

        b0, scale0 = self._emission_probs_and_log_scale(X[0])
        log_scale_sum += scale0
        alpha0 = self.pi * b0
        c[0] = 1.0 / max(np.sum(alpha0), 1e-300)
        alpha_hat[0] = alpha0 * c[0]

        emissions = [b0]
        for t in range(1, T):
            b_t, scale_t = self._emission_probs_and_log_scale(X[t])
            log_scale_sum += scale_t
            emissions.append(b_t)
            alpha_t = b_t * (alpha_hat[t - 1] @ self.A)
            c[t] = 1.0 / max(np.sum(alpha_t), 1e-300)
            alpha_hat[t] = alpha_t * c[t]

        beta_hat = np.zeros((T, n))
        beta_hat[T - 1] = 1.0
        for t in range(T - 2, -1, -1):
            beta_t = self.A @ (emissions[t + 1] * beta_hat[t + 1])
            beta_hat[t] = beta_t * c[t + 1]

        # See _emission_probs_and_log_scale: each timestep's b was rescaled
        # by exp(-log_scale_t), so -sum(log(c_t)) understates the true
        # log-likelihood by exactly sum(log_scale_t). Add it back.
        log_likelihood = -np.sum(np.log(c)) + log_scale_sum
        return alpha_hat, beta_hat, c, log_likelihood

    def smoothed_posteriors(self, X: np.ndarray) -> np.ndarray:
        """
        gamma_t(i) for every t — uses the full sequence (forward AND
        backward passes), so this is NOT causal. Offline use only
        (regime labeling for historical analysis / backtest annotation).
        """
        alpha_hat, beta_hat, _, _ = self._forward_backward(X)
        gamma = alpha_hat * beta_hat
        gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
        return gamma

    def forward_filter(self, X: np.ndarray) -> np.ndarray:
        """
        Filtered state probabilities P(state_t | x_1..x_t) for every t —
        causal, forward-only. This is what a live system would see
        replaying the sequence one step at a time via forward_step().
        """
        alpha_hat, _, _, _ = self._forward_backward(X)
        return alpha_hat  # already normalized to sum to 1 at each t by construction

    def forward_step(
        self, prev_alpha_hat: np.ndarray | None, x_t: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        One incremental forward step — the live/streaming primitive.
        `prev_alpha_hat=None` means this is the first observation.
        Returns (new_alpha_hat, scaling_factor_c_t).
        """
        b_t = self.emission_probs(x_t)
        if prev_alpha_hat is None:
            alpha_t = self.pi * b_t
        else:
            alpha_t = b_t * (prev_alpha_hat @ self.A)
        total = max(np.sum(alpha_t), 1e-300)
        c_t = 1.0 / total
        return alpha_t * c_t, c_t

    def log_likelihood(self, X: np.ndarray) -> float:
        _, _, _, ll = self._forward_backward(X)
        return float(ll)

    # ------------------------------------------------------------------ #
    # Viterbi (most likely state sequence) — offline use
    # ------------------------------------------------------------------ #

    def viterbi(self, X: np.ndarray) -> np.ndarray:
        T = len(X)
        n = self.n_states
        log_A = np.log(np.maximum(self.A, 1e-300))
        log_pi = np.log(np.maximum(self.pi, 1e-300))

        delta = np.zeros((T, n))
        psi = np.zeros((T, n), dtype=int)

        b0 = self.emission_probs(X[0])
        delta[0] = log_pi + np.log(np.maximum(b0, 1e-300))

        for t in range(1, T):
            b_t = self.emission_probs(X[t])
            log_b_t = np.log(np.maximum(b_t, 1e-300))
            scores = delta[t - 1][:, None] + log_A  # [from, to]
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = np.max(scores, axis=0) + log_b_t

        path = np.zeros(T, dtype=int)
        path[T - 1] = int(np.argmax(delta[T - 1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    # ------------------------------------------------------------------ #
    # Baum-Welch (EM) training
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray, max_iterations: int = 100, tolerance: float = 1e-4) -> "GaussianHMM":
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"X must have shape [T, {self.n_features}], got {X.shape}")
        if len(X) < self.n_states * 2:
            raise ValueError(
                f"Need at least {self.n_states * 2} observations to fit {self.n_states} states, got {len(X)}"
            )

        self._kmeans_pp_init(X)

        prev_ll = -np.inf
        self.log_likelihood_history = []
        self.converged = False

        for iteration in range(max_iterations):
            alpha_hat, beta_hat, c, ll = self._forward_backward(X)
            self.log_likelihood_history.append(ll)

            gamma = alpha_hat * beta_hat
            gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

            T = len(X)
            emissions = np.array([self.emission_probs(X[t]) for t in range(T)])
            xi_sum = np.zeros((self.n_states, self.n_states))
            for t in range(T - 1):
                numer = (
                    alpha_hat[t][:, None]
                    * self.A
                    * emissions[t + 1][None, :]
                    * beta_hat[t + 1][None, :]
                )
                numer *= c[t + 1]
                xi_sum += numer / max(numer.sum(), 1e-300)

            # M-step
            self.pi = gamma[0].copy()
            gamma_sum_excl_last = gamma[:-1].sum(axis=0)
            self.A = xi_sum / np.maximum(gamma_sum_excl_last[:, None], 1e-300)
            self.A /= np.maximum(self.A.sum(axis=1, keepdims=True), 1e-300)  # renormalize rows

            gamma_sum = gamma.sum(axis=0)
            self.means = (gamma.T @ X) / np.maximum(gamma_sum[:, None], 1e-300)

            diff_sq = (X[:, None, :] - self.means[None, :, :]) ** 2  # [T, n_states, n_features]
            weighted = gamma[:, :, None] * diff_sq
            self.variances = weighted.sum(axis=0) / np.maximum(gamma_sum[:, None], 1e-300)
            self.variances = np.maximum(self.variances, self.min_variance)

            if abs(ll - prev_ll) < tolerance:
                self.converged = True
                break
            prev_ll = ll

        return self

    def _kmeans_pp_init(self, X: np.ndarray) -> None:
        """k-means++ style seeding for the emission means — much better
        starting point for EM than pure random init, reducing the chance
        of converging to a degenerate local optimum."""
        n = self.n_states
        T = len(X)
        first_idx = self._rng.integers(0, T)
        chosen = [X[first_idx]]
        for _ in range(1, n):
            dists = np.min(
                [np.sum((X - c) ** 2, axis=1) for c in chosen], axis=0
            )
            probs = dists / max(dists.sum(), 1e-300)
            next_idx = self._rng.choice(T, p=probs)
            chosen.append(X[next_idx])
        self.means = np.array(chosen)
        self.variances = np.full((n, self.n_features), max(np.var(X), self.min_variance))
        self.pi = np.full(n, 1.0 / n)
        self.A = np.full((n, n), 1.0 / n)
