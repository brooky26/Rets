"""
Risk Engine — Level 4.

Independent from prediction, with absolute veto authority over
execution, per the spec. This is the one component in the whole pipeline
whose job is to say "no" — a positive-EV trade from Level 3 is necessary
but never sufficient; this engine can and does override it.

State tracked across calls (per account, not per symbol — risk is a
portfolio-level, not a per-symbol, concept):
    - equity curve (for drawdown-from-peak)
    - today's starting equity (for daily loss limit; resets on day rollover)
    - consecutive loss streak (resets on any win)
    - full trade P&L history (for empirical expected shortfall)

Checks applied (all are evaluated — `veto_reasons` can contain more than
one simultaneously, not just the first failure):
    1. Upstream EV gate must have passed (defense in depth — this engine
       doesn't re-derive EV, but refuses to approve anything the EV
       engine already rejected).
    2. Daily loss circuit breaker.
    3. Drawdown circuit breaker.
    4. Consecutive-loss circuit breaker.
    5. Risk of ruin, at the position size Kelly/exposure sizing arrives at.
    6. Expected shortfall (once enough trade history exists to estimate it).

Position sizing: fractional-Kelly (kelly_fraction_multiplier * f*),
additionally hard-capped by max_exposure_pct of current equity and
floored at min_stake. If the floor exceeds the cap (tiny account, or an
edge too thin to justify even the minimum stake), the trade is rejected
outright rather than silently forcing a stake outside the intended range.
"""

from __future__ import annotations

import math

from configs.risk_schema import RiskConfig
from expected_value.types import EVEstimate
from risk.kelly import kelly_fraction
from risk.ruin import risk_of_ruin
from risk.types import RiskAssessment, TradeOutcome

NAN = float("nan")
SECONDS_PER_DAY = 86400


class RiskEngine:
    def __init__(self, config: RiskConfig, starting_equity: float) -> None:
        if starting_equity <= 0:
            raise ValueError("starting_equity must be positive")
        self._config = config
        self._equity = starting_equity
        self._peak_equity = starting_equity
        self._current_day: int | None = None
        self._day_start_equity = starting_equity
        self._consecutive_losses = 0
        self._trade_pnls: list[float] = []
        self._trade_pnl_pcts: list[float] = []  # pnl as fraction of equity at the time
        self._consecutive_loss_cooldown_remaining = 0

    @property
    def equity(self) -> float:
        return self._equity

    # ------------------------------------------------------------------ #
    # State updates
    # ------------------------------------------------------------------ #

    def record_trade_result(self, outcome: TradeOutcome) -> None:
        day = outcome.epoch // SECONDS_PER_DAY
        if self._current_day is None:
            self._current_day = day
            self._day_start_equity = self._equity
        elif day != self._current_day:
            self._current_day = day
            self._day_start_equity = self._equity  # roll over to a fresh daily budget

        pnl_pct = outcome.pnl / self._equity if self._equity > 0 else 0.0

        self._equity = outcome.equity_after
        self._peak_equity = max(self._peak_equity, self._equity)

        if outcome.pnl > 0:
            self._consecutive_losses = 0
            self._consecutive_loss_cooldown_remaining = 0  # a win also clears any active cooldown
        else:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self._config.max_consecutive_losses:
                self._consecutive_loss_cooldown_remaining = self._config.consecutive_loss_cooldown_evaluations

        self._trade_pnls.append(outcome.pnl)
        self._trade_pnl_pcts.append(pnl_pct)

    def reset(self, starting_equity: float) -> None:
        """Full state reset (e.g. new trading session, or after a manual account reset)."""
        self.__init__(self._config, starting_equity)

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #

    @property
    def current_drawdown_pct(self) -> float:
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, (self._peak_equity - self._equity) / self._peak_equity)

    @property
    def daily_loss_pct(self) -> float:
        if self._day_start_equity <= 0:
            return 0.0
        loss = self._day_start_equity - self._equity
        return max(0.0, loss / self._day_start_equity)

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    @property
    def consecutive_loss_cooldown_remaining(self) -> int:
        return self._consecutive_loss_cooldown_remaining

    def expected_shortfall_pct(self) -> float:
        """
        Empirical CVaR at `expected_shortfall_confidence`: the mean loss
        (as a fraction of equity) among the worst (1 - confidence)
        fraction of recorded trades. NaN if there isn't enough history
        yet (min_trades_for_expected_shortfall).
        """
        n = len(self._trade_pnl_pcts)
        if n < self._config.min_trades_for_expected_shortfall:
            return NAN

        sorted_pcts = sorted(self._trade_pnl_pcts)  # ascending: worst first
        tail_fraction = 1.0 - self._config.expected_shortfall_confidence
        tail_count = max(1, int(math.ceil(n * tail_fraction)))
        worst = sorted_pcts[:tail_count]
        # Expected shortfall conventionally reported as a positive loss magnitude.
        return float(-sum(worst) / len(worst))

    # ------------------------------------------------------------------ #
    # Assessment
    # ------------------------------------------------------------------ #

    def assess(self, ev_estimate: EVEstimate) -> RiskAssessment:
        veto_reasons: list[str] = []

        # Tick the consecutive-loss cooldown, if one is active. This is what
        # actually prevents the self-locking failure mode: the streak
        # counter only resets on a WIN via record_trade_result, but a win
        # can never happen while this breaker is blocking every trade — so
        # an explicit, time-based cooldown is what lets trading resume.
        if self._consecutive_loss_cooldown_remaining > 0:
            self._consecutive_loss_cooldown_remaining -= 1
            if self._consecutive_loss_cooldown_remaining == 0:
                self._consecutive_losses = 0

        if not ev_estimate.is_valid or not ev_estimate.is_positive_ev:
            veto_reasons.append(
                "Upstream EV gate did not approve this trade"
                + (f": {ev_estimate.rejection_reason}" if ev_estimate.rejection_reason else ".")
            )

        if self.daily_loss_pct >= self._config.max_daily_loss_pct:
            veto_reasons.append(
                f"Daily loss circuit breaker: {self.daily_loss_pct:.2%} >= "
                f"{self._config.max_daily_loss_pct:.2%} limit."
            )

        if self.current_drawdown_pct >= self._config.max_drawdown_pct:
            veto_reasons.append(
                f"Drawdown circuit breaker: {self.current_drawdown_pct:.2%} >= "
                f"{self._config.max_drawdown_pct:.2%} limit."
            )

        if self._consecutive_losses >= self._config.max_consecutive_losses:
            veto_reasons.append(
                f"Consecutive-loss circuit breaker: {self._consecutive_losses} >= "
                f"{self._config.max_consecutive_losses} limit."
            )

        es_pct = self.expected_shortfall_pct()
        if es_pct == es_pct and es_pct > self._config.expected_shortfall_max_pct:  # not NaN
            veto_reasons.append(
                f"Expected shortfall {es_pct:.2%} exceeds the {self._config.expected_shortfall_max_pct:.2%} limit."
            )

        # Position sizing (computed regardless of vetoes above, so the
        # assessment is informative even when rejected — e.g. "here's
        # what we WOULD have staked").
        p = ev_estimate.probability_used if ev_estimate.is_valid else NAN
        b = ev_estimate.reward_to_risk if ev_estimate.is_valid else NAN

        f_raw = kelly_fraction(p, b) if p == p and b == b else 0.0
        f_applied = f_raw * self._config.kelly_fraction_multiplier

        exposure_cap_stake = self._equity * self._config.max_exposure_pct
        kelly_stake = f_applied * self._equity
        candidate_stake = min(kelly_stake, exposure_cap_stake)

        if candidate_stake < self._config.min_stake:
            veto_reasons.append(
                f"Sized stake {candidate_stake:.2f} falls below the minimum stake "
                f"{self._config.min_stake:.2f} — edge too thin relative to risk limits to trade at all."
            )
            recommended_stake = 0.0
        else:
            recommended_stake = candidate_stake

        # Risk of ruin at the position size actually being considered.
        if recommended_stake > 0 and p == p and b == b:
            capital_units = self._equity / recommended_stake
            ror = risk_of_ruin(p, b, capital_units)
        else:
            ror = 1.0 if (p == p and b == b) else NAN

        if ror == ror and ror > self._config.risk_of_ruin_threshold:
            veto_reasons.append(
                f"Risk of ruin {ror:.4%} exceeds the {self._config.risk_of_ruin_threshold:.4%} limit."
            )

        approved = len(veto_reasons) == 0
        final_stake = recommended_stake if approved else 0.0

        return RiskAssessment(
            symbol=ev_estimate.symbol,
            epoch=ev_estimate.epoch,
            approved=approved,
            recommended_stake=final_stake,
            kelly_fraction_raw=f_raw,
            kelly_fraction_applied=f_applied,
            risk_of_ruin=ror,
            current_drawdown_pct=self.current_drawdown_pct,
            daily_loss_pct=self.daily_loss_pct,
            consecutive_losses=self._consecutive_losses,
            expected_shortfall_pct=es_pct,
            veto_reasons=veto_reasons,
        )
