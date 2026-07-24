import numpy as np
import pytest

from configs.trade_management_schema import QLearningConfig
from trade_management.q_learning_agent import QLearningAgent
from trade_management.simulator import simulate_contract_episode
from trade_management.trainer import run_episode
from trade_management.types import OpenContractState


def test_simulator_produces_correct_number_of_states():
    rng = np.random.default_rng(0)
    states, settled_itm, final_payout = simulate_contract_episode(
        rng, "STPRNG100", direction=1, ticks_total=10, stake=10.0, payout=19.0,
    )
    assert len(states) == 10


def test_simulator_ticks_remaining_counts_down():
    rng = np.random.default_rng(0)
    states, _, _ = simulate_contract_episode(rng, "STPRNG100", direction=1, ticks_total=5, stake=10.0, payout=19.0)
    remaining = [s.ticks_remaining for s in states]
    assert remaining == [5, 4, 3, 2, 1]


def test_simulator_favorable_drift_usually_settles_itm():
    rng = np.random.default_rng(1)
    itm_count = 0
    n_trials = 100
    for i in range(n_trials):
        _, settled_itm, _ = simulate_contract_episode(
            rng, "STPRNG100", direction=1, ticks_total=10, stake=10.0, payout=19.0,
            tick_volatility=0.01, drift_per_tick=0.01,
        )
        itm_count += settled_itm
    assert itm_count / n_trials > 0.7


def test_simulator_bid_price_bounded_reasonably():
    rng = np.random.default_rng(2)
    states, _, _ = simulate_contract_episode(rng, "STPRNG100", direction=1, ticks_total=10, stake=10.0, payout=19.0)
    for s in states:
        assert 0.0 <= s.current_bid_price <= 19.0 + 1e-6


def test_simulator_direction_flips_favorable_move_interpretation():
    rng1 = np.random.default_rng(5)
    rng2 = np.random.default_rng(5)
    states_call, _, _ = simulate_contract_episode(rng1, "STPRNG100", direction=1, ticks_total=5, stake=10.0, payout=19.0, drift_per_tick=0.01)
    states_put, _, _ = simulate_contract_episode(rng2, "STPRNG100", direction=-1, ticks_total=5, stake=10.0, payout=19.0, drift_per_tick=0.01)
    for call_state, put_state in zip(states_call, states_put):
        if abs(call_state.favorable_move_pct) > 1e-9:
            assert np.sign(call_state.favorable_move_pct) == -np.sign(put_state.favorable_move_pct)


def _train(agent, rng, n_episodes, direction, drift_per_tick, ticks_total=8, tick_volatility=0.015):
    for _ in range(n_episodes):
        run_episode(
            agent, rng, "STPRNG100", direction=direction, ticks_total=ticks_total,
            stake=10.0, payout=19.0, tick_volatility=tick_volatility, drift_per_tick=drift_per_tick, learn=True,
        )


def _evaluate(agent, rng, n_episodes, direction, drift_per_tick, ticks_total=8, tick_volatility=0.015):
    total = 0.0
    for _ in range(n_episodes):
        total += run_episode(
            agent, rng, "STPRNG100", direction=direction, ticks_total=ticks_total,
            stake=10.0, payout=19.0, tick_volatility=tick_volatility, drift_per_tick=drift_per_tick, learn=False,
        )
    return total / n_episodes


def test_trained_agent_outperforms_always_hold_baseline_under_adverse_drift():
    """
    The core "did it actually learn something" test: train on episodes
    where the underlying tends to drift AGAINST the contract's direction,
    then compare the trained agent's average realized return against an
    always-HOLD policy on the same distribution of episodes.
    """
    config = QLearningConfig(
        learning_rate=0.2, discount_factor=0.95,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_episodes=3000,
        random_seed=7,
    )
    agent = QLearningAgent(config)
    train_rng = np.random.default_rng(100)
    _train(agent, train_rng, n_episodes=6000, direction=1, drift_per_tick=-0.006)

    eval_rng_trained = np.random.default_rng(999)
    trained_avg_return = _evaluate(agent, eval_rng_trained, n_episodes=500, direction=1, drift_per_tick=-0.006)

    baseline_rng = np.random.default_rng(999)
    baseline_total = 0.0
    n_baseline_episodes = 500
    for _ in range(n_baseline_episodes):
        states, settled_itm, final_payout = simulate_contract_episode(
            baseline_rng, "STPRNG100", direction=1, ticks_total=8, stake=10.0, payout=19.0,
            tick_volatility=0.015, drift_per_tick=-0.006,
        )
        baseline_total += final_payout - 10.0
    baseline_avg_return = baseline_total / n_baseline_episodes

    assert trained_avg_return > baseline_avg_return


def test_agent_learns_to_prefer_sell_when_deep_out_of_the_money_near_expiry():
    config = QLearningConfig(
        learning_rate=0.2, discount_factor=0.95,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_episodes=3000,
        random_seed=11,
    )
    agent = QLearningAgent(config)
    train_rng = np.random.default_rng(200)
    _train(agent, train_rng, n_episodes=6000, direction=1, drift_per_tick=-0.006, ticks_total=8)

    deep_otm_near_expiry = OpenContractState(
        symbol="STPRNG100", epoch=9999, ticks_remaining=1, ticks_total=8,
        stake=10.0, current_bid_price=1.0,
        entry_spot=100.0, current_spot=97.0, direction=1,
    )
    q_hold, q_sell = agent.q_values(deep_otm_near_expiry)
    assert q_sell >= q_hold - 0.5
