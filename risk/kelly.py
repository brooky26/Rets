"""
Kelly Criterion — optimal position sizing for a binary bet.

For a bet with win probability p, reward-to-risk ratio b (profit per
unit staked if you win — i.e. `reward_to_risk` from the EV engine), and
loss of the full stake if you lose, the fraction of capital f* that
maximizes long-run geometric growth rate is:

    f* = (b*p - q) / b,   where q = 1 - p

This is the standard closed-form result from maximizing E[log(1 + f*X)]
over f, where X = +b with prob p, -1 with prob q.

f* is negative whenever the bet has negative edge (b*p < q) — clipped to
0 in that case (never bet on a negative-edge outcome, which the upstream
EV gate should have already filtered out, but this is defense in depth).

Full Kelly is provably growth-optimal but has severe variance: it's
common for a full-Kelly bettor to see 50%+ drawdowns even on a genuinely
profitable edge, purely from bad luck. This is why RiskConfig applies
`kelly_fraction_multiplier` (fractional Kelly, e.g. 0.25 = quarter-Kelly)
on top of f* — trading some long-run growth rate for a much smoother
equity curve, which matters enormously when the "edge" itself is a
statistical estimate with its own uncertainty, not a certainty.
"""

from __future__ import annotations


def kelly_fraction(win_probability: float, reward_to_risk: float) -> float:
    """f* = (b*p - q) / b, clipped to [0, 1]."""
    if reward_to_risk <= 0:
        return 0.0
    p = win_probability
    q = 1.0 - p
    f_star = (reward_to_risk * p - q) / reward_to_risk
    return max(0.0, min(1.0, f_star))
