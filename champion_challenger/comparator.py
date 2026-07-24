"""
Champion-Challenger Comparator.

The spec's rule for the Continuous Learning Pipeline is explicit:
"Deploy only when statistically superior." This module is what makes
that concrete rather than aspirational — a challenger doesn't get
promoted just because its point-estimate win rate or Sharpe looks
slightly better; it has to clear a bootstrap-based significance test.

Method: independent bootstrap on each model's returns
----------------------------------------------------------
Given the champion's and challenger's realized per-trade returns (two
independent samples), resample each independently `n_bootstrap_resamples`
times (an ordinary i.i.d. bootstrap — reuses
`backtesting.monte_carlo.circular_block_bootstrap` at `block_size=1`,
which is exactly an i.i.d. bootstrap, rather than duplicating the
resampling logic). For each pair of resamples:

    improvement_i = mean(challenger_resample_i) - mean(champion_resample_i)

This gives a distribution of plausible values for the true improvement.
The lower bound of that distribution at `confidence_level` (e.g. the 5th
percentile for a 95% one-sided test) is what's checked:

    promote  <=>  bootstrap_lower_bound > min_improvement_threshold

Requiring the LOWER bound (not the mean improvement) to clear the bar is
what makes this a genuine significance test rather than a point-estimate
comparison.
"""

from __future__ import annotations

import numpy as np

from backtesting.monte_carlo import circular_block_bootstrap
from champion_challenger.types import PromotionDecision
from configs.champion_challenger_schema import ChampionChallengerConfig


class ChampionChallengerComparator:
    def __init__(self, config: ChampionChallengerConfig) -> None:
        self._config = config
        self._rng = np.random.default_rng(config.random_seed)

    def compare(
        self,
        champion_id: str,
        challenger_id: str,
        champion_returns: list[float],
        challenger_returns: list[float],
    ) -> PromotionDecision:
        c = self._config
        n_champion = len(champion_returns)
        n_challenger = len(challenger_returns)

        champion_mean = float(np.mean(champion_returns)) if n_champion > 0 else float("nan")
        challenger_mean = float(np.mean(challenger_returns)) if n_challenger > 0 else float("nan")
        mean_improvement = challenger_mean - champion_mean if n_champion > 0 and n_challenger > 0 else float("nan")

        if n_champion < c.min_trades_required or n_challenger < c.min_trades_required:
            return PromotionDecision(
                champion_id=champion_id, challenger_id=challenger_id, promote=False,
                reason=f"Insufficient sample size: champion has {n_champion}, challenger has "
                f"{n_challenger} trades, both need at least {c.min_trades_required}.",
                n_champion_trades=n_champion, n_challenger_trades=n_challenger,
                champion_mean_return=champion_mean, challenger_mean_return=challenger_mean,
                mean_improvement=mean_improvement, bootstrap_lower_bound=float("nan"),
                confidence_level=c.confidence_level,
            )

        champion_arr = np.array(champion_returns)
        challenger_arr = np.array(challenger_returns)

        champion_resamples = circular_block_bootstrap(champion_arr, 1, c.n_bootstrap_resamples, self._rng)
        challenger_resamples = circular_block_bootstrap(challenger_arr, 1, c.n_bootstrap_resamples, self._rng)

        champion_resample_means = champion_resamples.mean(axis=1)
        challenger_resample_means = challenger_resamples.mean(axis=1)
        improvements = challenger_resample_means - champion_resample_means

        lower_bound_percentile = (1.0 - c.confidence_level) * 100.0
        bootstrap_lower_bound = float(np.percentile(improvements, lower_bound_percentile))

        promote = bootstrap_lower_bound > c.min_improvement_threshold
        if promote:
            reason = (
                f"Challenger's improvement is statistically significant: the "
                f"{c.confidence_level:.0%} lower confidence bound on improvement "
                f"({bootstrap_lower_bound:.4f}) exceeds the minimum threshold "
                f"({c.min_improvement_threshold:.4f})."
            )
        else:
            reason = (
                f"Not statistically significant: the {c.confidence_level:.0%} lower confidence "
                f"bound on improvement ({bootstrap_lower_bound:.4f}) does not exceed the minimum "
                f"threshold ({c.min_improvement_threshold:.4f}) — the observed improvement could "
                f"plausibly be due to sampling noise rather than a genuine edge."
            )

        return PromotionDecision(
            champion_id=champion_id, challenger_id=challenger_id, promote=promote, reason=reason,
            n_champion_trades=n_champion, n_challenger_trades=n_challenger,
            champion_mean_return=champion_mean, challenger_mean_return=challenger_mean,
            mean_improvement=mean_improvement, bootstrap_lower_bound=bootstrap_lower_bound,
            confidence_level=c.confidence_level,
        )
