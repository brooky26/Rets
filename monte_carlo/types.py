"""Monte Carlo Price-Path Simulation — shared types.

Distinct from `backtesting.monte_carlo.MonteCarloStressTester`: that
module resamples ALREADY-REALIZED trade P&L sequences (via circular
block bootstrap) to stress-test an equity curve. This module instead
forward-simulates the UNDERLYING PRICE, forward in time from *now*,
under a Geometric Brownian Motion assumption — a genuinely different
question ("given the current estimated drift/volatility regime, what
does the distribution of price paths over the next N ticks look like,
and how often does it end up favorable to a candidate direction")
rather than "how much of what already happened was luck."
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PricePathSimulationResult:
    """
    Summary statistics over `n_paths` simulated GBM price paths, all
    starting from the same current price and simulated `horizon_ticks`
    ticks forward, evaluated against a candidate `direction` (+1 betting
    the price ends higher, -1 betting it ends lower).

    `prob_favorable` is the fraction of simulated paths whose TERMINAL
    price move is favorable to `direction` — this is a genuinely
    different probability estimate than any of the statistical
    probability models (Level 2), since it comes from explicitly
    forward-simulating price dynamics rather than fitting a
    classifier to historical (state, outcome) pairs. Combining both
    kinds of evidence (statistical pattern-matching AND explicit
    stochastic-process simulation) is exactly the kind of independent
    evidence the Ensemble Fusion Engine is built to combine.

    MFE/MAE (Maximum Favorable/Adverse Excursion) are computed as the
    best and worst signed-to-direction returns reached at ANY point
    along each path (not just at the terminal tick) — useful for the
    RL Trade Management agent's reward shaping and for a human/dashboard
    sense of "how much upside/downside risk does this trade path
    typically pass through en route to its terminal outcome."
    """

    symbol: str
    epoch: int
    direction: int  # +1 or -1; the candidate trade direction being evaluated
    n_paths: int
    horizon_ticks: int
    current_price: float
    mu_per_tick: float
    sigma_per_tick: float

    prob_favorable: float               # P(terminal move is favorable to direction), in [0, 1]
    expected_favorable_duration_ticks: float  # among favorable paths, mean tick-index of first crossing into favorable territory
    terminal_return_mean: float         # mean of (direction * (S_T - S_0) / S_0) across all paths
    terminal_return_std: float
    mfe_mean: float                     # mean best-seen favorable excursion (direction-signed return) across all paths
    mfe_p95: float
    mae_mean: float                     # mean worst-seen adverse excursion (direction-signed return, negative) across all paths
    mae_p95: float                      # 95th percentile of |adverse excursion| — a tail-risk figure, always >= 0

    @property
    def is_valid(self) -> bool:
        return self.prob_favorable == self.prob_favorable  # False only for NaN

    @property
    def implied_direction_confidence(self) -> float:
        """Symmetric transform of prob_favorable onto the same [0.5, 1.0]
        'confidence' scale ProbabilityEstimate.confidence uses, so this
        can be compared/blended with model-based confidence directly."""
        return max(self.prob_favorable, 1.0 - self.prob_favorable)
