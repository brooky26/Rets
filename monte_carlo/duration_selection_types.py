"""Duration selection: shared types. See duration_selector.py for the design."""

from __future__ import annotations

from dataclasses import dataclass

from expected_value.types import ContractSpec, EVEstimate
from probability.types import ProbabilityEstimate


@dataclass(frozen=True, slots=True)
class DurationCandidateEvaluation:
    """One candidate duration's full evaluation — kept for every candidate
    (not just the chosen one) so the ranked table is fully explainable,
    same discipline as the cross-symbol opportunity rankings."""

    duration_ticks: int
    method: str  # "monte_carlo" or "hurst_fallback"
    probability_estimate: ProbabilityEstimate
    ev_estimate: EVEstimate


@dataclass(frozen=True, slots=True)
class DurationSelectionResult:
    """Output of one duration-selection pass for one symbol/candle.

    `chosen` is None when no candidate durations are risk-adjusted-EV
    positive — meaning no duration is worth trading right now, not that
    selection itself failed. `all_candidates` is always populated
    (even when `chosen` is None) for logging/explainability."""

    symbol: str
    epoch: int
    chosen: DurationCandidateEvaluation | None
    all_candidates: tuple[DurationCandidateEvaluation, ...]

    @property
    def contract(self) -> ContractSpec | None:
        if self.chosen is None:
            return None
        return _contract_from_candidate(self.chosen)


def _contract_from_candidate(candidate: DurationCandidateEvaluation) -> ContractSpec:
    from expected_value.types import ContractType

    ev = candidate.ev_estimate
    return ContractSpec(
        contract_type=ContractType.RISE_FALL,
        stake=ev.stake,
        payout=ev.payout,
        duration_ticks=candidate.duration_ticks,
    )
