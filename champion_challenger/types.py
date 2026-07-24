"""Champion-Challenger — shared types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    champion_id: str
    challenger_id: str
    promote: bool
    reason: str

    n_champion_trades: int
    n_challenger_trades: int
    champion_mean_return: float
    challenger_mean_return: float
    mean_improvement: float
    bootstrap_lower_bound: float
    confidence_level: float
