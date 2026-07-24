"""
Tabular Q-Learning Agent — Level 7 trade management.

Why tabular Q-learning rather than PPO/SAC/DQN
------------------------------------------------
The spec names PPO, SAC, and (Rainbow/Double) DQN as candidate
algorithms — all designed for large or continuous state/action spaces
learned via neural function approximation. This problem doesn't have
that shape: the action space is 2 discrete actions (HOLD, SELL), the
decision horizon is a handful of ticks per contract, and the informative
state (time remaining, unrealized return, price trend since entry)
compresses naturally into a small number of bins. Tabular Q-learning is
the textbook-correct tool for exactly this regime — it converges to the
optimal Q-function under standard conditions, needs no neural network
training infrastructure, and every learned value is directly inspectable
(literally a table), which matters for the platform's explainability
requirement more than a neural net's opacity would help here. If the
state space grows later (more features, finer bins), the same interface
supports swapping in function approximation without touching callers.

The update rule
-----------------
Standard one-step Q-learning (Watkins, 1989):

    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

For terminal transitions (contract sold, or expired), there is no s' to
bootstrap from — the target is just `r`, not `r + gamma * max Q(s',*)`.
Getting this terminal case right is the single most common tabular
Q-learning bug (bootstrapping past the end of an episode silently
corrupts the learned values), so it's handled as an explicit branch,
not inferred from a sentinel state.

Reward structure (defined by the caller building training episodes, not
this agent — the agent is reward-agnostic by design):
    SELL at any point: reward = current_bid_price - stake  (realized, terminal)
    HOLD before expiry: reward = 0 (no immediate reward — the outcome
        is only realized at expiry or upon selling)
    HOLD at the final tick (forced settlement): reward = payout - stake
        or -stake, terminal, whichever the contract actually settled as
"""

from __future__ import annotations

import numpy as np

from configs.trade_management_schema import QLearningConfig
from trade_management.discretizer import StateKey, discretize_state
from trade_management.types import OpenContractState, TradeManagementAction, TradeManagementDecision

_ACTIONS = [TradeManagementAction.HOLD, TradeManagementAction.SELL]
_ACTION_INDEX = {a: i for i, a in enumerate(_ACTIONS)}


class QLearningAgent:
    def __init__(self, config: QLearningConfig) -> None:
        self._config = config
        self._rng = np.random.default_rng(config.random_seed)
        self._q_table: dict[StateKey, np.ndarray] = {}
        self._episodes_seen = 0

    def _q_row(self, key: StateKey) -> np.ndarray:
        if key not in self._q_table:
            self._q_table[key] = np.zeros(len(_ACTIONS))
        return self._q_table[key]

    def q_values(self, state: OpenContractState) -> tuple[float, float]:
        key = discretize_state(state, self._config)
        row = self._q_table.get(key, np.zeros(len(_ACTIONS)))
        return float(row[_ACTION_INDEX[TradeManagementAction.HOLD]]), float(
            row[_ACTION_INDEX[TradeManagementAction.SELL]]
        )

    @property
    def epsilon(self) -> float:
        c = self._config
        if c.epsilon_decay_episodes <= 0:
            return c.epsilon_end
        progress = min(1.0, self._episodes_seen / c.epsilon_decay_episodes)
        return c.epsilon_start + progress * (c.epsilon_end - c.epsilon_start)

    def end_episode(self) -> None:
        self._episodes_seen += 1

    def act(self, state: OpenContractState, explore: bool = True) -> TradeManagementDecision:
        key = discretize_state(state, self._config)
        row = self._q_row(key)
        q_hold, q_sell = float(row[0]), float(row[1])

        eps = self.epsilon if explore else 0.0
        explored = False
        if explore and self._rng.random() < eps:
            action_idx = int(self._rng.integers(0, len(_ACTIONS)))
            explored = True
        else:
            action_idx = int(np.argmax(row)) if row[1] > row[0] else 0

        return TradeManagementDecision(
            symbol=state.symbol,
            epoch=state.epoch,
            action=_ACTIONS[action_idx],
            q_hold=q_hold,
            q_sell=q_sell,
            epsilon_used=eps,
            explored=explored,
        )

    def update(
        self,
        state: OpenContractState,
        action: TradeManagementAction,
        reward: float,
        next_state: OpenContractState | None,
        done: bool,
    ) -> None:
        key = discretize_state(state, self._config)
        row = self._q_row(key)
        a_idx = _ACTION_INDEX[action]

        if done or next_state is None:
            target = reward
        else:
            next_key = discretize_state(next_state, self._config)
            next_row = self._q_row(next_key)
            target = reward + self._config.discount_factor * float(np.max(next_row))

        td_error = target - row[a_idx]
        row[a_idx] += self._config.learning_rate * td_error

    @property
    def n_states_visited(self) -> int:
        return len(self._q_table)

    @property
    def episodes_seen(self) -> int:
        return self._episodes_seen
