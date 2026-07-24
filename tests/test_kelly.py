import pytest

from risk.kelly import kelly_fraction


def test_kelly_matches_hand_computed_value():
    # p=0.6, b=0.9 (from earlier EV example) -> f* = (0.9*0.6 - 0.4)/0.9 = (0.54-0.4)/0.9 = 0.1556
    f = kelly_fraction(win_probability=0.6, reward_to_risk=0.9)
    assert f == pytest.approx(0.1556, abs=1e-3)


def test_kelly_zero_for_negative_edge():
    # p=0.4, b=0.9 -> f* = (0.36-0.6)/0.9 = negative -> clipped to 0
    f = kelly_fraction(win_probability=0.4, reward_to_risk=0.9)
    assert f == 0.0


def test_kelly_zero_for_zero_reward_to_risk():
    f = kelly_fraction(win_probability=0.9, reward_to_risk=0.0)
    assert f == 0.0


def test_kelly_zero_for_negative_reward_to_risk():
    f = kelly_fraction(win_probability=0.9, reward_to_risk=-0.5)
    assert f == 0.0


def test_kelly_increases_with_probability():
    low = kelly_fraction(win_probability=0.55, reward_to_risk=0.9)
    high = kelly_fraction(win_probability=0.75, reward_to_risk=0.9)
    assert high > low


def test_kelly_bounded_zero_to_one():
    for p in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
        for b in [0.1, 0.5, 1.0, 2.0, 5.0]:
            f = kelly_fraction(p, b)
            assert 0.0 <= f <= 1.0


def test_kelly_extreme_case_near_certain_win_high_payout():
    # p close to 1, generous payout -> f* should be close to 1 (bet big)
    f = kelly_fraction(win_probability=0.99, reward_to_risk=5.0)
    assert f > 0.9
