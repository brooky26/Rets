"""
Risk of Ruin — via the Cramér-Lundberg adjustment coefficient.

Setup
-----
Model account equity as a random walk in units of one stake: each bet
changes capital by +w (= reward_to_risk, profit per unit staked) with
probability p, or -1 (lose the stake) with probability q = 1-p. Starting
capital is C stake-units (= account_equity / stake). "Ruin" = the walk
hits 0.

This is exactly the setup classical ruin theory (Cramér-Lundberg, from
actuarial science / insurance mathematics) analyzes, applied here to a
sequence of trades instead of an insurer's claims process.

The adjustment coefficient
----------------------------
r^{S_n} is a martingale for any r solving E[r^X] = 1, where X is one
step's outcome. Substituting the two-point distribution:

    p * r^w + q * r^{-1} = 1

r = 1 is always a trivial solution. Reparametrizing r = e^{-theta}
(theta > 0 <=> r in (0,1)) makes the other side of the equation:

    h(theta) = p * e^{-theta*w} + q * e^{theta} - 1

h is convex in theta (sum of exponentials), h(0) = 0, and
h'(0) = -p*w + q = -(p*w - q) = -E[X]. So whenever the bet has positive
expected value (p*w > q, which is exactly the EV>0 condition from the EV
engine, restated in these units), h'(0) < 0: h dips negative just past
theta=0, then — since h(theta) -> infinity as theta -> infinity (the
q*e^{theta} term dominates) — must cross back to 0 exactly once more, at
some theta* > 0. That gives the nontrivial root r* = e^{-theta*} in (0,1).

By the optional stopping theorem applied to the martingale r*^{S_n}
(over an infinite time horizon, no upper barrier — i.e. the trader never
"cashes out" and stops), the ruin probability starting from capital C is:

    P(ruin | C) = (r*)^C

This is the textbook Cramér-Lundberg / Wald martingale result. It's an
approximation of the real situation in the usual ways ruin theory is:
infinite time horizon, no compounding of stake size as equity changes
(stake-in-units stays fixed), and iid bet outcomes — but it's a
principled, standard piece of applied probability, not an ad hoc
heuristic, and it degrades gracefully (returns 1.0, certain ruin) when
the edge is non-positive.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq


def risk_of_ruin(
    win_probability: float,
    reward_to_risk: float,
    capital_in_stake_units: float,
    theta_search_upper_bound: float = 50.0,
) -> float:
    """
    Returns P(eventual ruin) in [0, 1].

    `capital_in_stake_units` = account_equity / stake — how many stakes
    the account could survive losing in a row, dimensionally consistent
    with the w=reward_to_risk, l=1 (loss of one stake) parametrization.
    """
    p = win_probability
    q = 1.0 - p
    w = reward_to_risk

    if capital_in_stake_units <= 0:
        return 1.0
    if w <= 0 or p <= 0:
        return 1.0  # no possible win, or no payout — ruin is certain given infinite time

    edge = p * w - q  # positive-EV condition in these units
    if edge <= 0:
        return 1.0  # non-positive drift -> certain ruin over infinite horizon

    def h(theta: float) -> float:
        return p * math.exp(-theta * w) + q * math.exp(theta) - 1.0

    theta_lo = 1e-9
    theta_hi = theta_search_upper_bound
    if h(theta_lo) >= 0:
        # Numerically degenerate (edge extremely close to 0) — treat as ruin-certain.
        return 1.0

    expansions = 0
    while h(theta_hi) <= 0 and expansions < 20:
        theta_hi *= 2
        expansions += 1

    theta_star = brentq(h, theta_lo, theta_hi, xtol=1e-12, rtol=1e-12)
    r_star = math.exp(-theta_star)

    ruin_prob = r_star ** capital_in_stake_units
    return float(max(0.0, min(1.0, ruin_prob)))
