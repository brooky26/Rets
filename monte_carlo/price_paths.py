"""
Monte Carlo Price-Path Simulator — forward GBM simulation.

Model
-----
Geometric Brownian Motion in log-space (the standard, closed-form-exact
way to simulate GBM without Euler discretization bias):

    S_t = S_0 * exp( (mu - 0.5*sigma^2) * t + sigma * W_t )

where `mu` and `sigma` are the PER-TICK drift and volatility (not
annualized — Deriv synthetic-index tick contracts have no natural
trading calendar, matching the same no-annualization stance
`post_trade/analyzer.py` documents for performance ratios), and
`W_t = sqrt(t) * Z`, `Z ~ N(0, 1)`.

Simulating the full path (not just the terminal price) is necessary
here specifically because MFE/MAE and "expected favorable duration" are
properties of the WHOLE path, not the terminal point — vectorized via
cumulative sums of per-tick log-returns, so `n_paths` full paths of
`horizon_ticks` ticks each cost one `(n_paths, horizon_ticks)` array of
i.i.d. standard normals plus a cumulative sum, not a Python loop over
ticks.

Estimating mu/sigma from a MarketState
-----------------------------------------
This module is deliberately given already-estimated `mu_per_tick` /
`sigma_per_tick` rather than a `MarketState` directly — the caller
(opportunity scorer, continuous learning orchestrator, ensemble fusion)
decides how to derive them (e.g. `mu_per_tick` from recent realized
mean log-return, `sigma_per_tick` from recent realized log-return
std-dev, or a features-informed estimate blending `state.trend` and
`state.volatility`). Keeping this module a pure function of
(current_price, mu, sigma, horizon, direction) — the same
pure-core/stateful-wrapper split used throughout the platform
(`features/math_utils.py` vs `features/pipeline.py`) — means it is
identically usable from live scoring, offline backtesting, and the
continuous learning pipeline without any of them needing to agree on
how a MarketState maps to GBM parameters.
"""

from __future__ import annotations

import numpy as np

from configs.monte_carlo_schema import MonteCarloPricePathConfig
from monte_carlo.types import PricePathSimulationResult

NAN = float("nan")


def estimate_gbm_parameters_from_returns(log_returns: np.ndarray) -> tuple[float, float]:
    """
    Simple, honest MLE-style estimate of per-tick GBM drift and
    volatility from a window of historical log-returns:
        mu_per_tick    = mean(log_returns)
        sigma_per_tick = std(log_returns, ddof=1)

    Returns (NAN, NAN) if there isn't enough history (mirrors the
    NaN-on-insufficient-history convention used throughout
    `features/math_utils.py`) rather than raising, so callers can
    decide whether to skip simulation for this symbol/epoch.
    """
    if len(log_returns) < 2:
        return NAN, NAN
    mu = float(np.mean(log_returns))
    sigma = float(np.std(log_returns, ddof=1))
    return mu, sigma


def simulate_gbm_paths(
    current_price: float,
    mu_per_tick: float,
    sigma_per_tick: float,
    horizon_ticks: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Vectorized exact-GBM path simulation. Returns an array of shape
    (n_paths, horizon_ticks + 1) — column 0 is `current_price` for every
    path (the shared starting point), columns 1..horizon_ticks are the
    simulated forward prices.
    """
    if horizon_ticks < 1:
        raise ValueError("horizon_ticks must be >= 1")
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")
    if current_price <= 0:
        raise ValueError("current_price must be positive")

    z = rng.standard_normal(size=(n_paths, horizon_ticks))
    drift_term = (mu_per_tick - 0.5 * sigma_per_tick**2)
    log_increments = drift_term + sigma_per_tick * z
    cumulative_log_returns = np.cumsum(log_increments, axis=1)

    paths = np.empty((n_paths, horizon_ticks + 1))
    paths[:, 0] = current_price
    paths[:, 1:] = current_price * np.exp(cumulative_log_returns)
    return paths


class MonteCarloPricePathSimulator:
    """Stateless (config-only) wrapper — see module docstring for why
    `mu_per_tick`/`sigma_per_tick` are caller-supplied rather than derived
    internally from a MarketState."""

    def __init__(self, config: MonteCarloPricePathConfig) -> None:
        self._config = config
        self._rng = np.random.default_rng(config.random_seed)

    def simulate(
        self,
        symbol: str,
        epoch: int,
        current_price: float,
        mu_per_tick: float,
        sigma_per_tick: float,
        direction: int,
        horizon_ticks: int | None = None,
        n_paths: int | None = None,
    ) -> PricePathSimulationResult:
        """
        Run the simulation and summarize it against `direction`. Returns
        an all-NaN-scored (but structurally valid) result if
        `mu_per_tick`/`sigma_per_tick` are NaN (insufficient history to
        estimate them) or `direction` is 0 (no directional hypothesis to
        evaluate against) — mirrors the NaN-propagation convention used
        by `ProbabilityEstimate`/`EVEstimate`/etc. throughout the
        platform, rather than raising.
        """
        horizon = horizon_ticks or self._config.horizon_ticks
        paths_n = n_paths or self._config.n_paths

        if direction not in (1, -1):
            return self._invalid_result(symbol, epoch, direction, paths_n, horizon, current_price, mu_per_tick, sigma_per_tick)
        if mu_per_tick != mu_per_tick or sigma_per_tick != sigma_per_tick:  # NaN check
            return self._invalid_result(symbol, epoch, direction, paths_n, horizon, current_price, mu_per_tick, sigma_per_tick)
        if sigma_per_tick < 0:
            raise ValueError("sigma_per_tick cannot be negative")

        paths = simulate_gbm_paths(current_price, mu_per_tick, sigma_per_tick, horizon, paths_n, self._rng)

        # Direction-signed returns at every tick along every path, incl. t=0 (always 0).
        signed_returns = direction * (paths - current_price) / current_price

        terminal_returns = signed_returns[:, -1]
        prob_favorable = float(np.mean(terminal_returns > 0))
        terminal_return_mean = float(np.mean(terminal_returns))
        terminal_return_std = float(np.std(terminal_returns, ddof=1)) if paths_n > 1 else 0.0

        running_max = np.maximum.accumulate(signed_returns, axis=1)
        running_min = np.minimum.accumulate(signed_returns, axis=1)
        mfe_per_path = running_max[:, -1]  # best favorable excursion reached by horizon, per path
        mae_per_path = running_min[:, -1]  # worst adverse excursion reached by horizon, per path (<= 0)

        mfe_mean = float(np.mean(mfe_per_path))
        mfe_p95 = float(np.percentile(mfe_per_path, 95))
        mae_mean = float(np.mean(mae_per_path))
        mae_p95 = float(np.percentile(np.abs(mae_per_path), 95))

        expected_favorable_duration = self._mean_first_favorable_crossing_tick(signed_returns)

        return PricePathSimulationResult(
            symbol=symbol, epoch=epoch, direction=direction, n_paths=paths_n, horizon_ticks=horizon,
            current_price=current_price, mu_per_tick=mu_per_tick, sigma_per_tick=sigma_per_tick,
            prob_favorable=prob_favorable,
            expected_favorable_duration_ticks=expected_favorable_duration,
            terminal_return_mean=terminal_return_mean, terminal_return_std=terminal_return_std,
            mfe_mean=mfe_mean, mfe_p95=mfe_p95, mae_mean=mae_mean, mae_p95=mae_p95,
        )

    @staticmethod
    def _mean_first_favorable_crossing_tick(signed_returns: np.ndarray) -> float:
        """
        Mean, across paths that ever cross into favorable territory
        (signed_return > 0) at all, of the first tick index at which
        that crossing happens. Paths that never cross are excluded from
        the mean (NaN if none ever cross) — this deliberately answers
        "how fast does it become favorable, GIVEN that it does," not
        "how fast on average including paths that stay unfavorable the
        whole horizon," since the latter would conflate two different
        questions (speed vs likelihood, the latter already captured by
        prob_favorable).
        """
        favorable_mask = signed_returns > 0
        any_favorable = favorable_mask.any(axis=1)
        if not np.any(any_favorable):
            return NAN
        first_crossing = np.argmax(favorable_mask[any_favorable], axis=1)
        return float(np.mean(first_crossing))

    @staticmethod
    def _invalid_result(
        symbol: str, epoch: int, direction: int, n_paths: int, horizon: int,
        current_price: float, mu_per_tick: float, sigma_per_tick: float,
    ) -> PricePathSimulationResult:
        return PricePathSimulationResult(
            symbol=symbol, epoch=epoch, direction=direction, n_paths=n_paths, horizon_ticks=horizon,
            current_price=current_price, mu_per_tick=mu_per_tick, sigma_per_tick=sigma_per_tick,
            prob_favorable=NAN, expected_favorable_duration_ticks=NAN,
            terminal_return_mean=NAN, terminal_return_std=NAN,
            mfe_mean=NAN, mfe_p95=NAN, mae_mean=NAN, mae_p95=NAN,
        )


def refine_ev_with_monte_carlo(ev, mc_result: PricePathSimulationResult, blend_weight: float):
    """
    Blend a Level-3 EVEstimate's model-based `probability_used` with the
    Monte-Carlo-implied probability of a favorable terminal outcome,
    then RECOMPUTE every downstream EV field from the blended
    probability — never just overwrite `probability_used` in place,
    since every other field (`expected_value`, `risk_adjusted_score`,
    etc.) is derived from it and would otherwise silently go stale/
    inconsistent with the field it was computed from.

    `blend_weight` in [0, 1] is the weight given to the Monte Carlo
    probability; `(1 - blend_weight)` goes to the original model
    probability. blend_weight=0 recovers the original ev unchanged
    (structurally, not just numerically — the same fields are simply
    recomputed from the same probability).

    Returns a NEW `EVEstimate` (these are frozen dataclasses) built via
    the same formulas `ExpectedValueEngine.evaluate` uses — duplicated
    here rather than imported, since the engine's method also needs a
    `ProbabilityEstimate` and a config-driven gate, neither of which is
    the operation this helper performs (blending two SCALAR
    probabilities, not re-running gating policy).
    """
    from dataclasses import replace
    import math

    if not (0.0 <= blend_weight <= 1.0):
        raise ValueError("blend_weight must be in [0, 1]")
    if not ev.is_valid or not mc_result.is_valid or ev.direction == 0:
        return ev
    if mc_result.direction != ev.direction:
        raise ValueError(
            f"mc_result.direction ({mc_result.direction}) must match ev.direction ({ev.direction}) "
            "— blending probabilities computed against different trade directions is meaningless."
        )

    p_model = ev.probability_used
    p_mc = mc_result.prob_favorable
    p_blended = (1.0 - blend_weight) * p_model + blend_weight * p_mc

    stake = ev.stake
    payout = ev.payout
    profit_if_win = payout - stake
    loss_if_lose = -stake

    expected_value = p_blended * payout - stake
    expected_value_pct = expected_value / stake
    win_component = p_blended * profit_if_win
    loss_component = (1 - p_blended) * loss_if_lose
    variance = p_blended * (1 - p_blended) * (profit_if_win - loss_if_lose) ** 2
    outcome_std = math.sqrt(max(variance, 0.0))
    risk_adjusted_score = expected_value / outcome_std if outcome_std > 0 else 0.0

    return replace(
        ev,
        probability_used=p_blended,
        expected_value=expected_value,
        expected_value_pct=expected_value_pct,
        win_component=win_component,
        loss_component=loss_component,
        outcome_std=outcome_std,
        risk_adjusted_score=risk_adjusted_score,
    )
