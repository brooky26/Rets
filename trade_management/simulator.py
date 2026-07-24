"""
Contract Episode Simulator — training/testing support only.

IMPORTANT: this is NOT a model of Deriv's real contract pricing. Deriv's
actual `current_bid_price` for an open contract reflects their own
pricing engine (proper time-decay, volatility surface, spread) that
isn't available to this platform outside a live `proposal_open_contract`
subscription. This simulator exists purely to generate synthetic
(state, action, reward, next_state) transitions so the Q-learning agent
has something to learn from in tests and offline training demos, before
it's ever pointed at real Deriv contract-lifetime data.

The pricing proxy used here is a simplified digital-option intuition:
    bid_price ≈ P(finish in the money) * payout
where P(finish ITM) is approximated via a logistic function of a
standardized "distance from the money" — favorable price movement so
far, scaled by remaining time and volatility. This captures the right
*qualitative* behavior (favorable moves and less remaining uncertainty
both increase the price) without pretending to be a real pricing model.
"""

from __future__ import annotations

import numpy as np

from trade_management.types import OpenContractState


def simulate_contract_episode(
    rng: np.random.Generator,
    symbol: str,
    direction: int,
    ticks_total: int,
    stake: float,
    payout: float,
    tick_volatility: float = 0.01,
    drift_per_tick: float = 0.0,
    price_sensitivity: float = 3.0,
    entry_spot: float = 100.0,
    start_epoch: int = 0,
) -> tuple[list[OpenContractState], bool, float]:
    """
    Returns (states, settled_in_the_money, final_payout). `states[i]` is
    the state observed at tick i, *before* that tick's price move is
    applied — i.e. what an agent would actually see when deciding.
    """
    spot = entry_spot
    states: list[OpenContractState] = []

    for t in range(ticks_total):
        ticks_remaining = ticks_total - t
        favorable_move_pct = (spot - entry_spot) / entry_spot * direction
        z = favorable_move_pct / (tick_volatility * np.sqrt(max(ticks_remaining, 1)))
        p_itm = 1.0 / (1.0 + np.exp(-price_sensitivity * z))
        bid_price = p_itm * payout

        states.append(
            OpenContractState(
                symbol=symbol,
                epoch=start_epoch + t,
                ticks_remaining=ticks_remaining,
                ticks_total=ticks_total,
                stake=stake,
                current_bid_price=bid_price,
                entry_spot=entry_spot,
                current_spot=spot,
                direction=direction,
            )
        )

        spot *= 1.0 + rng.normal(drift_per_tick, tick_volatility)

    final_favorable_move = (spot - entry_spot) / entry_spot * direction
    settled_itm = final_favorable_move > 0
    final_payout = payout if settled_itm else 0.0
    return states, settled_itm, final_payout
