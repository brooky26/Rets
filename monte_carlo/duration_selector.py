"""
Duration selection: picks which candidate `duration_ticks` to trade at,
for THIS symbol/candle, informed by two signals in a fixed priority
order — replacing a single fixed `duration_ticks` config value with a
per-candle computed choice.

1. PRIMARY — Monte Carlo price-path simulation (monte_carlo/price_paths.py):
   for each candidate duration, simulate forward GBM paths from the
   current (mu, sigma) estimate and read off `prob_favorable` (the
   fraction of paths ending favorable to the fused model's currently
   favored direction). MC's answer is trusted for a given candidate only
   when it's actually confident: the standard error of a proportion
   estimated from `n_paths`, sqrt(p*(1-p)/n_paths), must be at or below
   `mc_max_standard_error`, AND `n_paths` itself must be at least
   `mc_min_paths` (a large `n_paths` with a degenerate p near 0 or 1 can
   report a small SE that isn't really trustworthy under GBM's
   thin-tailed assumption; this check is deliberately simple — an
   approximation, not a rigorous fat-tail-adjusted interval).

2. FALLBACK — Hurst-exponent-based analytical scaling, used whenever MC
   isn't confident for a given candidate (including when mu/sigma are
   NaN — not enough recent closes yet — so a fallback is ALWAYS
   available as long as the fused probability estimate itself is valid).
   The Hurst exponent H (recovered from `MarketState.persistence` via
   the state encoder's own affine mapping: persistence = (H - 0.5) / 0.5,
   so H = 0.5 + 0.5 * persistence) describes how a process's cumulative
   deviation scales with time — classically, rescaled-range analysis
   gives R/S ~ n^H (Hurst, 1951). Using this AS AN APPROXIMATION (not a
   rigorous derivation — no claim of formal correctness beyond "a
   reasonable, documented heuristic") to rescale the fused model's
   current directional confidence across candidate durations:

       analytical_confidence(d) = 0.5 + (p_now - 0.5) * (d / d_ref) ** (H - 0.5)

   where `p_now` is the fused estimate's `confidence` (already in
   [0.5, 1.0]) and `d_ref` is `hurst_reference_duration_ticks` — the
   duration the fused probability is implicitly "at" (roughly the
   tick-to-tick gap its training labels were defined over). For H > 0.5
   (persistent/trending) the edge decays SLOWER than a plain random
   walk as duration grows; for H < 0.5 (mean-reverting) it decays
   FASTER. This reuses math the platform already computes for every
   `MarketState` rather than inventing new machinery.

Whichever signal is used for a given candidate, EV is computed the same
way (`ExpectedValueEngine`, at that candidate's own `ContractSpec`), and
the duration MAXIMIZING risk-adjusted EV among EV-positive candidates is
selected. If every candidate's EV is non-positive, no duration is
selected — the caller treats this exactly like any other reason a
candle's evaluation doesn't produce a trade this cycle: skipped, not
forced. Every candidate's full evaluation is kept (see
`DurationSelectionResult.all_candidates`) for logging/explainability,
mirroring the cross-symbol opportunity ranking table already logged
elsewhere.
"""

from __future__ import annotations

import math

from configs.duration_selection_schema import DurationSelectionConfig
from configs.ev_schema import ExpectedValueConfig
from ensemble.fusion_engine import monte_carlo_result_to_probability_estimate
from expected_value.engine import ExpectedValueEngine
from expected_value.types import ContractSpec, ContractType
from monte_carlo.duration_selection_types import DurationCandidateEvaluation, DurationSelectionResult
from monte_carlo.price_paths import MonteCarloPricePathSimulator
from probability.types import ProbabilityEstimate

NAN = float("nan")


def _hurst_from_persistence(persistence: float) -> float:
    """Inverse of state_encoder's affine mapping: persistence = (H - 0.5) / 0.5."""
    return 0.5 + 0.5 * persistence


def _clip_to_confidence_range(p: float) -> float:
    """Confidence is defined on [0.5, 1.0] throughout the platform (see
    ProbabilityEstimate docstring) — the Hurst rescaling can in principle
    push slightly outside that range for extreme H/duration-ratio
    combinations, so clip rather than let an out-of-convention value leak
    downstream."""
    if p != p:  # NaN
        return p
    return min(1.0, max(0.5, p))


def _hurst_fallback_probability(
    symbol: str, epoch: int, fused: ProbabilityEstimate, duration_ticks: int, config: DurationSelectionConfig,
    hurst_exponent: float,
) -> ProbabilityEstimate:
    d_ref = config.hurst_reference_duration_ticks
    if hurst_exponent != hurst_exponent:  # NaN persistence — cannot rescale, pass fused through unchanged
        scaled_confidence = fused.confidence
    else:
        ratio = (duration_ticks / d_ref) ** (hurst_exponent - 0.5)
        scaled_confidence = _clip_to_confidence_range(0.5 + (fused.confidence - 0.5) * ratio)

    if fused.expected_direction == 1:
        prob_up, prob_down = scaled_confidence, 1.0 - scaled_confidence
    elif fused.expected_direction == -1:
        prob_up, prob_down = 1.0 - scaled_confidence, scaled_confidence
    else:
        prob_up = prob_down = 0.5

    return ProbabilityEstimate(
        symbol=symbol, epoch=epoch, model_name="hurst_fallback",
        prob_up=prob_up, prob_down=prob_down,
        uncertainty=fused.uncertainty,  # not re-estimated — see module docstring
        expected_direction=fused.expected_direction, confidence=scaled_confidence,
    )


class DurationSelector:
    def __init__(
        self,
        config: DurationSelectionConfig,
        ev_config: ExpectedValueConfig,
        mc_simulator: MonteCarloPricePathSimulator | None,
    ) -> None:
        self._config = config
        self._ev_engine = ExpectedValueEngine(ev_config)
        self._mc_simulator = mc_simulator

    def select(
        self,
        symbol: str,
        epoch: int,
        current_price: float,
        fused_probability: ProbabilityEstimate,
        mu_per_tick: float,
        sigma_per_tick: float,
        hurst_persistence: float,
        stake: float,
        assumed_payout_ratio: float,
    ) -> DurationSelectionResult:
        if not fused_probability.is_valid or fused_probability.expected_direction == 0:
            return DurationSelectionResult(symbol=symbol, epoch=epoch, chosen=None, all_candidates=())

        direction = fused_probability.expected_direction
        hurst_exponent = _hurst_from_persistence(hurst_persistence)
        candidates: list[DurationCandidateEvaluation] = []

        for duration_ticks in self._config.candidate_durations_ticks:
            probability_estimate, method = self._evaluate_candidate_probability(
                symbol, epoch, current_price, fused_probability, mu_per_tick, sigma_per_tick,
                direction, duration_ticks, hurst_exponent,
            )
            contract = ContractSpec(
                contract_type=ContractType.RISE_FALL, stake=stake,
                payout=stake * assumed_payout_ratio, duration_ticks=duration_ticks,
            )
            ev_estimate = self._ev_engine.evaluate(probability_estimate, contract)
            candidates.append(
                DurationCandidateEvaluation(
                    duration_ticks=duration_ticks, method=method,
                    probability_estimate=probability_estimate, ev_estimate=ev_estimate,
                )
            )

        viable = [c for c in candidates if c.ev_estimate.is_valid and c.ev_estimate.is_positive_ev]
        chosen = max(viable, key=lambda c: c.ev_estimate.risk_adjusted_score) if viable else None

        return DurationSelectionResult(
            symbol=symbol, epoch=epoch, chosen=chosen, all_candidates=tuple(candidates),
        )

    def _evaluate_candidate_probability(
        self, symbol: str, epoch: int, current_price: float, fused: ProbabilityEstimate,
        mu_per_tick: float, sigma_per_tick: float, direction: int, duration_ticks: int,
        hurst_exponent: float,
    ) -> tuple[ProbabilityEstimate, str]:
        if self._mc_simulator is not None and mu_per_tick == mu_per_tick and sigma_per_tick == sigma_per_tick:
            mc_result = self._mc_simulator.simulate(
                symbol=symbol, epoch=epoch, current_price=current_price,
                mu_per_tick=mu_per_tick, sigma_per_tick=sigma_per_tick,
                direction=direction, horizon_ticks=duration_ticks,
            )
            if mc_result.is_valid and mc_result.n_paths >= self._config.mc_min_paths:
                p = mc_result.prob_favorable
                standard_error = math.sqrt(max(p * (1.0 - p), 0.0) / mc_result.n_paths)
                if standard_error <= self._config.mc_max_standard_error:
                    return monte_carlo_result_to_probability_estimate(mc_result), "monte_carlo"

        return (
            _hurst_fallback_probability(symbol, epoch, fused, duration_ticks, self._config, hurst_exponent),
            "hurst_fallback",
        )
