import math

import pytest

from risk.ruin import risk_of_ruin


def test_certain_ruin_for_nonpositive_edge():
    # p=0.5, b=1.0 -> edge = 0.5*1 - 0.5 = 0 -> non-positive drift -> certain ruin
    ror = risk_of_ruin(win_probability=0.5, reward_to_risk=1.0, capital_in_stake_units=100)
    assert ror == 1.0


def test_certain_ruin_for_negative_edge():
    ror = risk_of_ruin(win_probability=0.3, reward_to_risk=0.9, capital_in_stake_units=100)
    assert ror == 1.0


def test_certain_ruin_for_zero_capital():
    ror = risk_of_ruin(win_probability=0.9, reward_to_risk=2.0, capital_in_stake_units=0)
    assert ror == 1.0


def test_positive_edge_gives_ruin_probability_below_one():
    ror = risk_of_ruin(win_probability=0.6, reward_to_risk=0.9, capital_in_stake_units=50)
    assert 0.0 < ror < 1.0


def test_ruin_probability_decreases_with_more_capital():
    small_capital = risk_of_ruin(win_probability=0.55, reward_to_risk=1.0, capital_in_stake_units=10)
    large_capital = risk_of_ruin(win_probability=0.55, reward_to_risk=1.0, capital_in_stake_units=200)
    assert large_capital < small_capital


def test_ruin_probability_decreases_with_larger_edge():
    weak_edge = risk_of_ruin(win_probability=0.51, reward_to_risk=1.0, capital_in_stake_units=50)
    strong_edge = risk_of_ruin(win_probability=0.7, reward_to_risk=1.0, capital_in_stake_units=50)
    assert strong_edge < weak_edge


def test_adjustment_coefficient_satisfies_its_own_equation():
    """
    Directly verify the defining equation of the adjustment coefficient:
    p*r^w + q*r^(-1) = 1 at the r implied by the returned ruin probability
    (since ruin_prob = r^C, we can back out r = ruin_prob^(1/C) and check
    it actually solves the characteristic equation).
    """
    p, w, C = 0.6, 0.9, 80.0
    q = 1 - p
    ror = risk_of_ruin(win_probability=p, reward_to_risk=w, capital_in_stake_units=C)
    r = ror ** (1.0 / C)

    lhs = p * (r ** w) + q * (r ** (-1))
    assert lhs == pytest.approx(1.0, abs=1e-6)


def test_consistency_with_ev_positive_condition():
    """
    EV > 0 for a contract translates (in these w=reward_to_risk, l=1 units)
    to exactly the positive-drift condition p*w > q. Confirm the boundary
    behaves as expected: just above the EV=0 line gives ror<1, just below
    gives ror==1 (certain ruin).
    """
    # p*w = q at the boundary: for w=1.0, boundary is p=0.5
    just_above = risk_of_ruin(win_probability=0.51, reward_to_risk=1.0, capital_in_stake_units=100)
    just_below = risk_of_ruin(win_probability=0.49, reward_to_risk=1.0, capital_in_stake_units=100)
    assert just_above < 1.0
    assert just_below == 1.0


def test_ruin_probability_bounded_zero_one():
    for p in [0.51, 0.6, 0.7, 0.9, 0.99]:
        for w in [0.5, 1.0, 2.0]:
            for C in [5, 50, 500]:
                ror = risk_of_ruin(win_probability=p, reward_to_risk=w, capital_in_stake_units=C)
                assert 0.0 <= ror <= 1.0


def test_very_strong_edge_and_large_capital_gives_near_zero_ruin():
    ror = risk_of_ruin(win_probability=0.9, reward_to_risk=2.0, capital_in_stake_units=1000)
    assert ror < 1e-6
