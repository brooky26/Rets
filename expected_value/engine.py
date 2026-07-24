"""
Expected Value Engine.

Math
----
For a binary win/lose contract with probability p of winning:
    profit_if_win = payout - stake
    loss_if_lose  = -stake

    EV = p * profit_if_win + (1-p) * loss_if_lose
       = p*(payout - stake) - (1-p)*stake
       = p*payout - stake                              <- simplifies cleanly

    EV_pct         = EV / stake
    reward_to_risk = profit_if_win / stake               (the "b" in Kelly-criterion notation)

    win_component  = p * profit_if_win
    loss_component = (1-p) * loss_if_lose                (negative)
    EV = win_component + loss_component                  (sanity identity, checked in tests)

Outcome variance (single Bernoulli bet, two-point distribution):
    Var[outcome] = p(1-p) * (profit_if_win - loss_if_lose)^2
    outcome_std  = sqrt(Var[outcome])

    risk_adjusted_score = EV / outcome_std               (Sharpe-style ratio for one bet —
                                                            NOT the same as the portfolio-level
                                                            Risk Assessment Level 4 will do;
                                                            this is a single-trade signal-to-noise
                                                            measure, nothing more)

Gating
------
`min_ev_threshold` (default 0.0) is the spec's hard rule made concrete:
"never execute negative EV trades." `min_reward_to_risk` and
`min_probability_confidence` are additional, optional filters — set them
to 0.0 / 0.5 respectively to disable and rely on the EV threshold alone.
Every rejection carries a human-readable `rejection_reason`, matching the
spec's explainability requirement that rejected trades explain why.
"""

from __future__ import annotations

import math

from configs.ev_schema import ExpectedValueConfig
from expected_value.types import ContractSpec, EVEstimate
from probability.types import ProbabilityEstimate

NAN = float("nan")


class ExpectedValueEngine:
    def __init__(self, config: ExpectedValueConfig) -> None:
        self._config = config

    def evaluate(
        self, probability: ProbabilityEstimate, contract: ContractSpec
    ) -> EVEstimate:
        if not probability.is_valid:
            return EVEstimate(
                symbol=probability.symbol,
                epoch=probability.epoch,
                direction=0,
                probability_used=NAN,
                stake=contract.stake,
                payout=contract.payout,
                expected_value=NAN,
                expected_value_pct=NAN,
                reward_to_risk=NAN,
                win_component=NAN,
                loss_component=NAN,
                outcome_std=NAN,
                risk_adjusted_score=NAN,
                is_positive_ev=False,
                rejection_reason="Input ProbabilityEstimate is invalid (NaN present).",
            )

        if probability.expected_direction == 0:
            return EVEstimate(
                symbol=probability.symbol,
                epoch=probability.epoch,
                direction=0,
                probability_used=0.5,
                stake=contract.stake,
                payout=contract.payout,
                expected_value=NAN,
                expected_value_pct=NAN,
                reward_to_risk=NAN,
                win_component=NAN,
                loss_component=NAN,
                outcome_std=NAN,
                risk_adjusted_score=NAN,
                is_positive_ev=False,
                rejection_reason="No directional edge (expected_direction == 0) — nothing to evaluate.",
            )

        direction = probability.expected_direction
        p = probability.prob_up if direction == 1 else probability.prob_down

        stake = contract.stake
        payout = contract.payout
        profit_if_win = contract.profit_if_win
        loss_if_lose = contract.loss_if_lose

        expected_value = p * payout - stake
        expected_value_pct = expected_value / stake
        reward_to_risk = profit_if_win / stake

        win_component = p * profit_if_win
        loss_component = (1 - p) * loss_if_lose

        variance = p * (1 - p) * (profit_if_win - loss_if_lose) ** 2
        outcome_std = math.sqrt(max(variance, 0.0))
        risk_adjusted_score = expected_value / outcome_std if outcome_std > 0 else 0.0

        is_positive_ev = expected_value >= self._config.min_ev_threshold
        rejection_reason = None

        if not is_positive_ev:
            rejection_reason = (
                f"Expected value {expected_value:.4f} is below the minimum "
                f"threshold {self._config.min_ev_threshold:.4f}."
            )
        elif reward_to_risk < self._config.min_reward_to_risk:
            is_positive_ev = False
            rejection_reason = (
                f"Reward-to-risk {reward_to_risk:.4f} is below the minimum "
                f"{self._config.min_reward_to_risk:.4f}."
            )
        elif p < self._config.min_probability_confidence:
            is_positive_ev = False
            rejection_reason = (
                f"Probability {p:.4f} is below the minimum confidence "
                f"{self._config.min_probability_confidence:.4f}."
            )

        return EVEstimate(
            symbol=probability.symbol,
            epoch=probability.epoch,
            direction=direction,
            probability_used=p,
            stake=stake,
            payout=payout,
            expected_value=expected_value,
            expected_value_pct=expected_value_pct,
            reward_to_risk=reward_to_risk,
            win_component=win_component,
            loss_component=loss_component,
            outcome_std=outcome_std,
            risk_adjusted_score=risk_adjusted_score,
            is_positive_ev=is_positive_ev,
            rejection_reason=rejection_reason,
        )
