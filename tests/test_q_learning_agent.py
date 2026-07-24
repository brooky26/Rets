import pytest

from configs.trade_management_schema import QLearningConfig
from trade_management.discretizer import discretize_state
from trade_management.q_learning_agent import QLearningAgent
from trade_management.types import OpenContractState, TradeManagementAction


def make_state(**overrides) -> OpenContractState:
    defaults = dict(
        symbol="STPRNG100", epoch=1000, ticks_remaining=3, ticks_total=5,
        stake=10.0, current_bid_price=12.0, entry_spot=100.0, current_spot=101.0, direction=1,
    )
    defaults.update(overrides)
    return OpenContractState(**defaults)


def make_config(**overrides) -> QLearningConfig:
    defaults = dict(learning_rate=0.1, discount_factor=0.9, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_episodes=100)
    defaults.update(overrides)
    return QLearningConfig(**defaults)


def test_initial_q_values_are_zero():
    agent = QLearningAgent(make_config())
    state = make_state()
    q_hold, q_sell = agent.q_values(state)
    assert q_hold == 0.0
    assert q_sell == 0.0


def test_terminal_update_matches_hand_computed_td_target():
    agent = QLearningAgent(make_config(learning_rate=0.1))
    state = make_state()
    agent.update(state, TradeManagementAction.SELL, reward=5.0, next_state=None, done=True)
    _, q_sell = agent.q_values(state)
    assert q_sell == pytest.approx(0.5)


def test_bootstrap_update_matches_hand_computed_td_target():
    agent = QLearningAgent(make_config(learning_rate=0.1, discount_factor=0.9))
    # ticks_remaining=4/5=0.8 and 1/5=0.2 land in clearly different time bins
    # (unlike e.g. 3/5 vs 2/5, which — with 5 equal-width bins — can share a bin
    # depending on exact digitize boundary behavior; picking well-separated
    # fractions avoids that ambiguity entirely).
    state = make_state(ticks_remaining=4, ticks_total=5)
    next_state = make_state(ticks_remaining=1, ticks_total=5, epoch=1001)
    assert discretize_state(state, agent._config) != discretize_state(next_state, agent._config)

    agent.update(next_state, TradeManagementAction.HOLD, reward=2.0, next_state=None, done=True)
    q_hold_next, _ = agent.q_values(next_state)
    assert q_hold_next == pytest.approx(0.2)

    agent.update(state, TradeManagementAction.HOLD, reward=0.0, next_state=next_state, done=False)
    q_hold, _ = agent.q_values(state)
    assert q_hold == pytest.approx(0.018, abs=1e-9)


def test_done_flag_overrides_next_state_even_if_provided():
    agent = QLearningAgent(make_config(learning_rate=0.1, discount_factor=0.9))
    next_state = make_state(epoch=2000)
    agent.update(next_state, TradeManagementAction.HOLD, reward=100.0, next_state=None, done=True)

    state = make_state()
    agent.update(state, TradeManagementAction.SELL, reward=3.0, next_state=next_state, done=True)
    _, q_sell = agent.q_values(state)
    assert q_sell == pytest.approx(0.3)


def test_repeated_updates_converge_toward_target_for_deterministic_reward():
    agent = QLearningAgent(make_config(learning_rate=0.3))
    state = make_state()
    for _ in range(200):
        agent.update(state, TradeManagementAction.SELL, reward=7.0, next_state=None, done=True)
    _, q_sell = agent.q_values(state)
    assert q_sell == pytest.approx(7.0, abs=1e-3)


def test_epsilon_starts_at_configured_start_value():
    agent = QLearningAgent(make_config(epsilon_start=0.8, epsilon_decay_episodes=100))
    assert agent.epsilon == pytest.approx(0.8)


def test_epsilon_decays_linearly_to_end_value():
    config = make_config(epsilon_start=1.0, epsilon_end=0.0, epsilon_decay_episodes=10)
    agent = QLearningAgent(config)
    for _ in range(5):
        agent.end_episode()
    assert agent.epsilon == pytest.approx(0.5, abs=1e-6)
    for _ in range(5):
        agent.end_episode()
    assert agent.epsilon == pytest.approx(0.0, abs=1e-6)


def test_epsilon_floors_at_end_value_after_decay_period():
    config = make_config(epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_episodes=5)
    agent = QLearningAgent(config)
    for _ in range(50):
        agent.end_episode()
    assert agent.epsilon == pytest.approx(0.1)


def test_act_explore_false_is_deterministic_greedy():
    agent = QLearningAgent(make_config())
    state = make_state()
    agent.update(state, TradeManagementAction.SELL, reward=10.0, next_state=None, done=True)
    decision = agent.act(state, explore=False)
    assert decision.action == TradeManagementAction.SELL
    assert decision.explored is False
    assert decision.epsilon_used == 0.0


def test_act_ties_break_toward_hold():
    agent = QLearningAgent(make_config())
    state = make_state()
    decision = agent.act(state, explore=False)
    assert decision.action == TradeManagementAction.HOLD


def test_act_explore_true_with_epsilon_one_always_random():
    config = make_config(epsilon_start=1.0, epsilon_end=1.0, epsilon_decay_episodes=1)
    agent = QLearningAgent(config)
    state = make_state()
    agent.update(state, TradeManagementAction.SELL, reward=10.0, next_state=None, done=True)

    actions_seen = set()
    for _ in range(50):
        decision = agent.act(state, explore=True)
        actions_seen.add(decision.action)
    assert TradeManagementAction.HOLD in actions_seen
    assert TradeManagementAction.SELL in actions_seen


def test_n_states_visited_tracks_distinct_states():
    agent = QLearningAgent(make_config())
    state_a = make_state(ticks_remaining=3)
    state_b = make_state(ticks_remaining=1)
    agent.update(state_a, TradeManagementAction.HOLD, 0.0, None, done=True)
    agent.update(state_b, TradeManagementAction.SELL, 1.0, None, done=True)
    assert agent.n_states_visited == 2


def test_episodes_seen_increments_on_end_episode():
    agent = QLearningAgent(make_config())
    assert agent.episodes_seen == 0
    agent.end_episode()
    agent.end_episode()
    assert agent.episodes_seen == 2
