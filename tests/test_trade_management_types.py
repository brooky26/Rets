import pytest

from configs.trade_management_schema import QLearningConfig
from trade_management.discretizer import discretize_return, discretize_state, discretize_time_remaining, discretize_trend
from trade_management.types import OpenContractState


def make_state(**overrides) -> OpenContractState:
    defaults = dict(
        symbol="STPRNG100", epoch=1000, ticks_remaining=3, ticks_total=5,
        stake=10.0, current_bid_price=12.0, entry_spot=100.0, current_spot=101.0, direction=1,
    )
    defaults.update(overrides)
    return OpenContractState(**defaults)


def test_time_remaining_fraction():
    state = make_state(ticks_remaining=2, ticks_total=4)
    assert state.time_remaining_fraction == pytest.approx(0.5)


def test_time_remaining_fraction_zero_total_is_safe():
    state = make_state(ticks_remaining=0, ticks_total=0)
    assert state.time_remaining_fraction == 0.0


def test_unrealized_return_positive_when_bid_above_stake():
    state = make_state(stake=10.0, current_bid_price=15.0)
    assert state.unrealized_return == pytest.approx(0.5)


def test_unrealized_return_negative_when_bid_below_stake():
    state = make_state(stake=10.0, current_bid_price=4.0)
    assert state.unrealized_return == pytest.approx(-0.6)


def test_favorable_move_pct_positive_for_call_when_price_rises():
    state = make_state(direction=1, entry_spot=100.0, current_spot=102.0)
    assert state.favorable_move_pct == pytest.approx(0.02)


def test_favorable_move_pct_positive_for_put_when_price_falls():
    state = make_state(direction=-1, entry_spot=100.0, current_spot=98.0)
    assert state.favorable_move_pct == pytest.approx(0.02)


def test_favorable_move_pct_negative_for_call_when_price_falls():
    state = make_state(direction=1, entry_spot=100.0, current_spot=98.0)
    assert state.favorable_move_pct == pytest.approx(-0.02)


def test_discretize_time_remaining_boundaries():
    assert discretize_time_remaining(0.0, 5) == 0
    assert discretize_time_remaining(0.1, 5) == 0
    assert discretize_time_remaining(0.25, 5) == 1
    assert discretize_time_remaining(0.5, 5) == 2
    assert discretize_time_remaining(0.75, 5) == 3
    assert discretize_time_remaining(1.0, 5) == 4


def test_discretize_return_uses_provided_edges():
    edges = [-0.4, -0.1, 0.1, 0.4]
    assert discretize_return(-0.9, edges) == 0
    assert discretize_return(-0.2, edges) == 1
    assert discretize_return(0.0, edges) == 2
    assert discretize_return(0.2, edges) == 3
    assert discretize_return(0.9, edges) == 4


def test_discretize_trend_uses_provided_edges():
    edges = [-0.02, -0.005, 0.005, 0.02]
    assert discretize_trend(-0.03, edges) == 0
    assert discretize_trend(0.0, edges) == 2
    assert discretize_trend(0.03, edges) == 4


def test_discretize_state_returns_three_tuple():
    config = QLearningConfig()
    state = make_state()
    key = discretize_state(state, config)
    assert len(key) == 3
    assert all(isinstance(x, int) for x in key)


def test_discretize_state_consistent_for_identical_states():
    config = QLearningConfig()
    state_a = make_state()
    state_b = make_state()
    assert discretize_state(state_a, config) == discretize_state(state_b, config)
