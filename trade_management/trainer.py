"""
Episode runner — drives one simulated contract lifetime through the
agent, handling the terminal-transition logic correctly (SELL ends the
episode immediately; HOLD at the final tick forces settlement; HOLD
before the final tick bootstraps to the next state). This logic is
shared between training (explore=True, updates the Q-table) and
evaluation (explore=False, pure greedy, no updates) so there's exactly
one place that can get the terminal-vs-bootstrap distinction wrong.
"""

from __future__ import annotations

import numpy as np

from trade_management.q_learning_agent import QLearningAgent
from trade_management.simulator import simulate_contract_episode
from trade_management.types import TradeManagementAction


def run_episode(
    agent: QLearningAgent,
    rng: np.random.Generator,
    symbol: str,
    direction: int,
    ticks_total: int,
    stake: float,
    payout: float,
    tick_volatility: float = 0.01,
    drift_per_tick: float = 0.0,
    learn: bool = True,
) -> float:
    """Returns the realized reward (profit/loss in currency units) for this episode."""
    states, settled_itm, final_payout = simulate_contract_episode(
        rng=rng, symbol=symbol, direction=direction, ticks_total=ticks_total,
        stake=stake, payout=payout, tick_volatility=tick_volatility, drift_per_tick=drift_per_tick,
    )

    for i, state in enumerate(states):
        decision = agent.act(state, explore=learn)
        is_last_tick = i == len(states) - 1

        if decision.action == TradeManagementAction.SELL:
            reward = state.current_bid_price - state.stake
            if learn:
                agent.update(state, TradeManagementAction.SELL, reward, None, done=True)
                agent.end_episode()
            return reward

        if is_last_tick:
            reward = final_payout - state.stake
            if learn:
                agent.update(state, TradeManagementAction.HOLD, reward, None, done=True)
                agent.end_episode()
            return reward
        else:
            next_state = states[i + 1]
            if learn:
                agent.update(state, TradeManagementAction.HOLD, 0.0, next_state, done=False)

    raise RuntimeError("run_episode fell through without a terminal transition.")
