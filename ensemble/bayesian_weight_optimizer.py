"""
Bayesian Weight Optimizer — contextual, multi-objective ensemble weight
search via Optuna.

What "weights" means here, and how the constraint is enforced
--------------------------------------------------------------------
A weight vector `w` over the ensemble's member models (one entry per
name in `model_names`), constrained `w_i >= 0` and `sum(w_i) == 1`.
Rather than sampling n-1 free values and solving for the last (which
needs an explicit feasibility check/rejection whenever the implied
last weight would be negative), each trial samples `n` independent
values in `[1e-6, 1.0]` — one per model — and L1-normalizes them:
`w_i = raw_i / sum(raw)`. This satisfies the sum-to-one constraint BY
CONSTRUCTION for every trial (no infeasible region, nothing to reject
or prune on constraint grounds), while still letting Optuna's sampler
(TPE by default) learn the useful relative-magnitude structure between
models.

Objective: how a candidate weight vector's "return series" is built
-------------------------------------------------------------------------
Re-running the full pipeline (fusion -> EV gate -> risk sizing ->
execution) for every one of `n_trials` candidate weight vectors would
be prohibitively expensive and would entangle this module with all of
those. Instead — and this simplification is the key modeling choice
this module makes, documented plainly rather than presented as more
rigorous than it is — each historical `WeightOptimizationRecord`
already carries the REALIZED return_pct of a trade that was actually
taken (in some direction, under whatever weights/sizing were live at
the time). For a candidate weight vector, this module asks: "had THIS
weight vector been used to fuse the same member probabilities, how
confidently would it have agreed with the direction that was actually
traded?" —

    fused_prob_up   = weighted average of model_probabilities (renormalized
                       over whichever models this record actually has)
    aligned_conf    = fused_prob_up if direction == 1 else (1 - fused_prob_up)
    confidence_scalar = clip(2 * aligned_conf - 1, 0.0, 1.0)   # 0 at 50/50, 1 at fully confident
    simulated_return  = confidence_scalar * realized_return_pct

`confidence_scalar` is 0 whenever the candidate weights would have put
this record's fused probability at or below 50% agreement with the
direction actually traded — modeled as "this weight vector would not
have sized this trade at all," contributing a flat 0 return, rather
than inverting the trade (there is no historical data on what would
have happened trading the OTHER direction, so this module does not
invent an answer for that). Weight vectors that consistently agree
strongly with winning trades and weakly/not at all with losing ones
produce a `simulated_return` series with a higher realized Sharpe/Calmar
— which is exactly the ranking signal Bayesian optimization needs, even
though `simulated_return` is a proxy for (not identical to) what would
have literally happened with different weights live in production.

Sharpe/Calmar formulas are copied from `post_trade/analyzer.py`'s
`PerformanceMetrics` computation (not imported — this module needs them
applied to a `simulated_return` series that is a proxy for actual
trade returns, not a re-run of the actual post-trade pipeline, so
importing would misleadingly suggest a tighter coupling than exists).
"""

from __future__ import annotations

import logging

import numpy as np
import optuna

from configs.ensemble_schema import BayesianWeightOptimizerConfig
from ensemble.types import WeightOptimizationRecord
from meta_learning.types import EnsembleWeights
from regime.types import RegimeLabel

logger = logging.getLogger("ensemble.bayesian_weight_optimizer")
optuna.logging.set_verbosity(optuna.logging.WARNING)  # match this project's low-noise logging convention


def _fused_prob_up(weights: dict[str, float], record: WeightOptimizationRecord) -> float | None:
    available = {name: w for name, w in weights.items() if name in record.model_probabilities}
    total = sum(available.values())
    if total <= 0:
        return None
    return sum((w / total) * record.model_probabilities[name] for name, w in available.items())


def simulate_returns_for_weights(weights: dict[str, float], records: list[WeightOptimizationRecord]) -> np.ndarray:
    """Pure function — see module docstring for the confidence-scaling
    methodology. Records for which none of the candidate weights' models
    are present are skipped entirely (not even a 0 contributed) since
    there is no evidence at all to evaluate this weight vector against
    for that record."""
    returns = []
    for record in records:
        fused = _fused_prob_up(weights, record)
        if fused is None:
            continue
        aligned_confidence = fused if record.direction == 1 else (1.0 - fused)
        confidence_scalar = float(np.clip(2 * aligned_confidence - 1.0, 0.0, 1.0))
        returns.append(confidence_scalar * record.realized_return_pct)
    return np.array(returns, dtype=np.float64)


def compute_sharpe_and_calmar(returns: np.ndarray) -> tuple[float, float, float]:
    """Returns (sharpe_ratio, calmar_ratio, max_drawdown_pct) — same
    formulas as `post_trade.analyzer.PostTradeAnalyzer.compute_metrics`.
    `calmar_ratio` is capped at 100.0 (rather than left as `inf`) when
    `max_drawdown_pct == 0`, since Optuna's samplers handle finite
    objective values far better than `inf`; 100.0 is far above any
    achievable finite Calmar ratio in practice, so it still always ranks
    above every non-degenerate trial without poisoning the sampler's
    internal statistics with an actual infinity."""
    n = len(returns)
    if n == 0:
        return 0.0, 0.0, 0.0

    average_return = float(np.mean(returns))
    return_std = float(np.std(returns, ddof=1)) if n > 1 else 0.0
    sharpe_ratio = average_return / return_std if return_std > 0 else 0.0

    equity_curve = np.cumprod(1.0 + returns)
    equity_curve_with_start = np.concatenate([[1.0], equity_curve])
    running_peak = np.maximum.accumulate(equity_curve_with_start)
    drawdown_series = (running_peak - equity_curve_with_start) / running_peak
    max_drawdown_pct = float(np.max(drawdown_series))
    total_return_pct = float(equity_curve[-1] - 1.0)

    if max_drawdown_pct > 0:
        calmar_ratio = min(total_return_pct / max_drawdown_pct, 100.0)
    else:
        calmar_ratio = 100.0 if total_return_pct > 0 else 0.0

    return sharpe_ratio, calmar_ratio, max_drawdown_pct


class BayesianWeightOptimizer:
    def __init__(self, config: BayesianWeightOptimizerConfig, model_names: list[str]) -> None:
        self._config = config
        self._model_names = list(model_names)

    def _sample_weights(self, trial: "optuna.Trial") -> dict[str, float]:
        raw = {name: trial.suggest_float(f"w_{name}", 1e-6, 1.0) for name in self._model_names}
        total = sum(raw.values())
        return {name: v / total for name, v in raw.items()}

    def _objective_multi(self, trial: "optuna.Trial", records: list[WeightOptimizationRecord]) -> tuple[float, float]:
        weights = self._sample_weights(trial)
        returns = simulate_returns_for_weights(weights, records)
        sharpe, calmar, _ = compute_sharpe_and_calmar(returns)
        return sharpe, calmar

    def _objective_custom(self, trial: "optuna.Trial", records: list[WeightOptimizationRecord]) -> float:
        weights = self._sample_weights(trial)
        returns = simulate_returns_for_weights(weights, records)
        if len(returns) == 0:
            return -1e9
        expectancy = float(np.mean(returns))
        _, _, max_drawdown_pct = compute_sharpe_and_calmar(returns)
        # quality_score_proxy: mean confidence_scalar isn't directly available here without
        # recomputing, so use |2*mean(returns>0 alignment)| via a second lightweight pass —
        # simplest honest proxy is just expectancy sign-agnostic magnitude of engagement:
        n_engaged = np.count_nonzero(returns)
        engagement_ratio = n_engaged / len(records) if records else 0.0
        quality_score_proxy = engagement_ratio
        return quality_score_proxy * expectancy - self._config.lambda_drawdown * max_drawdown_pct

    def _optimize_one(self, records: list[WeightOptimizationRecord], source_label: str) -> EnsembleWeights:
        sampler = optuna.samplers.TPESampler(seed=self._config.sampler_seed)

        if self._config.objective_mode == "multi_objective":
            study = optuna.create_study(directions=["maximize", "maximize"], sampler=sampler)
            study.optimize(lambda t: self._objective_multi(t, records), n_trials=self._config.n_trials, show_progress_bar=False)

            pareto_trials = study.best_trials
            if not pareto_trials:
                logger.warning("Bayesian weight optimization for '%s' produced no Pareto-optimal trials; using equal weights.", source_label)
                return EnsembleWeights(weights={n: 1.0 / len(self._model_names) for n in self._model_names}, source=source_label, n_trials=0)

            sharpe_vals = np.array([t.values[0] for t in pareto_trials])
            calmar_vals = np.array([t.values[1] for t in pareto_trials])
            sharpe_range = max(sharpe_vals.max() - sharpe_vals.min(), 1e-9)
            calmar_range = max(calmar_vals.max() - calmar_vals.min(), 1e-9)
            norm_sharpe = (sharpe_vals - sharpe_vals.min()) / sharpe_range
            norm_calmar = (calmar_vals - calmar_vals.min()) / calmar_range
            scalarized = norm_sharpe + norm_calmar  # equal-weighted scalarization across the Pareto front
            best_idx = int(np.argmax(scalarized))
            best_trial = pareto_trials[best_idx]
        else:
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(lambda t: self._objective_custom(t, records), n_trials=self._config.n_trials, show_progress_bar=False)
            best_trial = study.best_trial

        raw = {name: best_trial.params[f"w_{name}"] for name in self._model_names}
        total = sum(raw.values())
        weights = {name: v / total for name, v in raw.items()}
        return EnsembleWeights(weights=weights, source=source_label, n_trials=self._config.n_trials)

    def optimize(
        self, records: list[WeightOptimizationRecord]
    ) -> tuple[EnsembleWeights, dict[RegimeLabel, EnsembleWeights]]:
        """
        Runs the global optimization (over all records), and — if
        `config.per_regime` — one additional optimization per regime
        with at least `config.min_trades_per_regime` records. Returns
        `(global_weights, {regime: regime_weights, ...})`.
        """
        if len(records) == 0:
            raise ValueError("Cannot optimize weights with zero historical records.")

        global_weights = self._optimize_one(records, source_label="bayesian_optimizer_global")

        regime_weights: dict[RegimeLabel, EnsembleWeights] = {}
        if self._config.per_regime:
            by_regime: dict[RegimeLabel, list[WeightOptimizationRecord]] = {}
            for record in records:
                if record.regime is not None:
                    by_regime.setdefault(record.regime, []).append(record)

            for regime, regime_records in by_regime.items():
                if len(regime_records) < self._config.min_trades_per_regime:
                    logger.info(
                        "Skipping regime-specific optimization for %s — only %d records "
                        "(need >= %d).", regime.value, len(regime_records), self._config.min_trades_per_regime,
                    )
                    continue
                regime_weights[regime] = self._optimize_one(regime_records, source_label=f"bayesian_optimizer_{regime.value}")

        return global_weights, regime_weights
