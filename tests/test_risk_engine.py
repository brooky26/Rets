import pytest

from configs.risk_schema import RiskConfig
from expected_value.types import EVEstimate
from risk.engine import RiskEngine
from risk.types import TradeOutcome


def make_ev(
    symbol="STPRNG100", epoch=1000, is_positive_ev=True, probability_used=0.6,
    reward_to_risk=0.9, expected_value=1.4, rejection_reason=None,
) -> EVEstimate:
    return EVEstimate(
        symbol=symbol, epoch=epoch, direction=1, probability_used=probability_used,
        stake=10.0, payout=19.0, expected_value=expected_value,
        expected_value_pct=expected_value / 10.0, reward_to_risk=reward_to_risk,
        win_component=0.0, loss_component=0.0, outcome_std=1.0,
        risk_adjusted_score=1.0, is_positive_ev=is_positive_ev,
        rejection_reason=rejection_reason,
    )


def make_config(**overrides) -> RiskConfig:
    defaults = dict(
        max_daily_loss_pct=0.05, max_drawdown_pct=0.20, max_consecutive_losses=6,
        kelly_fraction_multiplier=0.25, max_exposure_pct=0.10, min_stake=1.0,
        risk_of_ruin_threshold=0.05, expected_shortfall_confidence=0.95,
        expected_shortfall_max_pct=0.10, min_trades_for_expected_shortfall=10,
    )
    defaults.update(overrides)
    return RiskConfig(**defaults)


def test_positive_ev_trade_with_clean_state_is_approved():
    engine = RiskEngine(make_config(), starting_equity=10000.0)
    ev = make_ev()
    result = engine.assess(ev)
    assert result.approved is True
    assert result.recommended_stake > 0
    assert result.veto_reasons == []


def test_upstream_negative_ev_is_vetoed():
    engine = RiskEngine(make_config(), starting_equity=10000.0)
    ev = make_ev(is_positive_ev=False, rejection_reason="EV below threshold")
    result = engine.assess(ev)
    assert result.approved is False
    assert any("EV" in r for r in result.veto_reasons)
    assert result.recommended_stake == 0.0


def test_daily_loss_circuit_breaker_triggers():
    config = make_config(max_daily_loss_pct=0.05)
    engine = RiskEngine(config, starting_equity=10000.0)
    engine.record_trade_result(TradeOutcome(epoch=1000, pnl=-600.0, equity_after=9400.0))
    result = engine.assess(make_ev())
    assert result.approved is False
    assert any("Daily loss" in r for r in result.veto_reasons)


def test_drawdown_circuit_breaker_triggers():
    config = make_config(max_drawdown_pct=0.15)
    engine = RiskEngine(config, starting_equity=10000.0)
    engine.record_trade_result(TradeOutcome(epoch=1000, pnl=2000.0, equity_after=12000.0))
    engine.record_trade_result(TradeOutcome(epoch=86400 + 1000, pnl=-2000.0, equity_after=10000.0))
    result = engine.assess(make_ev(epoch=86400 + 2000))
    assert result.current_drawdown_pct == pytest.approx((12000 - 10000) / 12000)
    assert result.approved is False
    assert any("Drawdown" in r for r in result.veto_reasons)


def test_consecutive_loss_circuit_breaker_triggers():
    config = make_config(max_consecutive_losses=3)
    engine = RiskEngine(config, starting_equity=10000.0)
    for i in range(3):
        engine.record_trade_result(TradeOutcome(epoch=1000 + i, pnl=-50.0, equity_after=10000 - 50 * (i + 1)))
    result = engine.assess(make_ev(epoch=2000))
    assert result.consecutive_losses == 3
    assert result.approved is False
    assert any("Consecutive-loss" in r for r in result.veto_reasons)


def test_consecutive_losses_reset_on_win():
    config = make_config(max_consecutive_losses=3)
    engine = RiskEngine(config, starting_equity=10000.0)
    engine.record_trade_result(TradeOutcome(epoch=1000, pnl=-50.0, equity_after=9950.0))
    engine.record_trade_result(TradeOutcome(epoch=1001, pnl=-50.0, equity_after=9900.0))
    engine.record_trade_result(TradeOutcome(epoch=1002, pnl=100.0, equity_after=10000.0))
    assert engine.consecutive_losses == 0


def test_kelly_sizing_scales_with_equity():
    engine_small = RiskEngine(make_config(), starting_equity=1000.0)
    engine_large = RiskEngine(make_config(), starting_equity=100000.0)
    result_small = engine_small.assess(make_ev())
    result_large = engine_large.assess(make_ev())
    assert result_large.recommended_stake > result_small.recommended_stake
    assert result_small.kelly_fraction_applied == pytest.approx(result_large.kelly_fraction_applied)


def test_max_exposure_caps_stake_even_when_kelly_wants_more():
    config = make_config(kelly_fraction_multiplier=1.0, max_exposure_pct=0.01)
    engine = RiskEngine(config, starting_equity=10000.0)
    ev = make_ev(probability_used=0.95, reward_to_risk=2.0)
    result = engine.assess(ev)
    if result.approved:
        assert result.recommended_stake <= 10000.0 * 0.01 + 1e-6


def test_min_stake_floor_rejects_thin_edge():
    config = make_config(min_stake=50.0, kelly_fraction_multiplier=0.25, max_exposure_pct=0.10)
    engine = RiskEngine(config, starting_equity=1000.0)
    ev = make_ev(probability_used=0.501, reward_to_risk=1.0)
    result = engine.assess(ev)
    assert result.approved is False
    assert any("minimum stake" in r for r in result.veto_reasons)


def test_expected_shortfall_not_enforced_before_min_trades():
    config = make_config(min_trades_for_expected_shortfall=100)
    engine = RiskEngine(config, starting_equity=10000.0)
    engine.record_trade_result(TradeOutcome(epoch=1000, pnl=-50.0, equity_after=9950.0))
    result = engine.assess(make_ev(epoch=2000))
    assert result.expected_shortfall_pct != result.expected_shortfall_pct
    assert not any("Expected shortfall" in r for r in result.veto_reasons)


def test_expected_shortfall_enforced_once_enough_history():
    config = make_config(
        min_trades_for_expected_shortfall=10, expected_shortfall_confidence=0.9,
        expected_shortfall_max_pct=0.01,
    )
    engine = RiskEngine(config, starting_equity=10000.0)
    equity = 10000.0
    for i in range(20):
        pnl = -500.0 if i % 2 == 0 else 100.0
        equity += pnl
        engine.record_trade_result(TradeOutcome(epoch=1000 + i, pnl=pnl, equity_after=equity))
    result = engine.assess(make_ev(epoch=5000))
    assert result.expected_shortfall_pct == result.expected_shortfall_pct
    assert result.expected_shortfall_pct > 0
    assert result.approved is False
    assert any("Expected shortfall" in r for r in result.veto_reasons)


def test_risk_of_ruin_included_in_assessment():
    engine = RiskEngine(make_config(risk_of_ruin_threshold=0.5), starting_equity=10000.0)
    result = engine.assess(make_ev())
    assert 0.0 <= result.risk_of_ruin <= 1.0


def test_multiple_simultaneous_veto_reasons_all_reported():
    config = make_config(max_daily_loss_pct=0.01, max_consecutive_losses=1)
    engine = RiskEngine(config, starting_equity=10000.0)
    engine.record_trade_result(TradeOutcome(epoch=1000, pnl=-200.0, equity_after=9800.0))
    result = engine.assess(make_ev(epoch=2000))
    assert result.approved is False
    assert len(result.veto_reasons) >= 2


def test_starting_equity_must_be_positive():
    with pytest.raises(ValueError):
        RiskEngine(make_config(), starting_equity=0.0)
    with pytest.raises(ValueError):
        RiskEngine(make_config(), starting_equity=-100.0)


def test_reset_clears_all_state():
    engine = RiskEngine(make_config(), starting_equity=10000.0)
    engine.record_trade_result(TradeOutcome(epoch=1000, pnl=-500.0, equity_after=9500.0))
    engine.record_trade_result(TradeOutcome(epoch=1001, pnl=-500.0, equity_after=9000.0))
    assert engine.consecutive_losses == 2

    engine.reset(starting_equity=5000.0)
    assert engine.equity == 5000.0
    assert engine.consecutive_losses == 0
    assert engine.current_drawdown_pct == 0.0


def test_daily_loss_resets_on_new_day():
    config = make_config(max_daily_loss_pct=0.05)
    engine = RiskEngine(config, starting_equity=10000.0)
    engine.record_trade_result(TradeOutcome(epoch=1000, pnl=-600.0, equity_after=9400.0))
    result_day1 = engine.assess(make_ev(epoch=2000))
    assert result_day1.approved is False

    next_day_epoch = 1000 + 86400
    engine.record_trade_result(TradeOutcome(epoch=next_day_epoch, pnl=10.0, equity_after=9410.0))
    result_day2 = engine.assess(make_ev(epoch=next_day_epoch + 1000))
    assert not any("Daily loss" in r for r in result_day2.veto_reasons)


def test_consecutive_loss_breaker_recovers_after_cooldown_elapses():
    """
    The core fix: without an explicit cooldown, the consecutive-loss
    streak can only reset on a WIN — but a win can never happen while the
    breaker itself blocks every trade, so trading would lock out
    permanently. Confirm the cooldown actually clears the streak and
    lets a subsequent otherwise-clean trade through.
    """
    config = make_config(max_consecutive_losses=3, consecutive_loss_cooldown_evaluations=5)
    engine = RiskEngine(config, starting_equity=10000.0)
    for i in range(3):
        engine.record_trade_result(TradeOutcome(epoch=1000 + i, pnl=-50.0, equity_after=10000 - 50 * (i + 1)))

    # Breaker is tripped immediately after the 3rd loss.
    assert engine.consecutive_loss_cooldown_remaining == 5

    # Each assess() call ticks the cooldown down by one. It takes exactly
    # 5 calls to reach zero; the tick-down and the veto check happen in
    # the SAME call, so the call that brings it to zero already sees a
    # cleared streak and is approved — the first 4 calls stay blocked,
    # the 5th is not.
    for _ in range(4):
        result = engine.assess(make_ev(epoch=2000))
        assert result.approved is False
        assert any("Consecutive-loss" in r for r in result.veto_reasons)

    assert engine.consecutive_loss_cooldown_remaining == 1
    result = engine.assess(make_ev(epoch=2010))
    assert engine.consecutive_loss_cooldown_remaining == 0
    assert engine.consecutive_losses == 0
    assert not any("Consecutive-loss" in r for r in result.veto_reasons)
    assert result.approved is True


def test_consecutive_loss_cooldown_cleared_immediately_by_a_win():
    """A win occurring for any reason (e.g. a different symbol's trade settling)
    should clear the cooldown immediately, not wait for it to tick down."""
    config = make_config(max_consecutive_losses=2, consecutive_loss_cooldown_evaluations=100)
    engine = RiskEngine(config, starting_equity=10000.0)
    engine.record_trade_result(TradeOutcome(epoch=1000, pnl=-50.0, equity_after=9950.0))
    engine.record_trade_result(TradeOutcome(epoch=1001, pnl=-50.0, equity_after=9900.0))
    assert engine.consecutive_loss_cooldown_remaining == 100

    engine.record_trade_result(TradeOutcome(epoch=1002, pnl=25.0, equity_after=9925.0))  # a win
    assert engine.consecutive_loss_cooldown_remaining == 0
    assert engine.consecutive_losses == 0


def test_drawdown_breaker_does_not_auto_recover_without_manual_reset():
    """
    Unlike the consecutive-loss breaker, the drawdown breaker is
    intentionally a hard stop — matching how institutional risk desks
    typically treat a drawdown breach (pending review) rather than a
    scheduled pause. Confirm it stays tripped indefinitely across many
    assess() calls with no trades, until an explicit reset().
    """
    config = make_config(max_drawdown_pct=0.10)
    engine = RiskEngine(config, starting_equity=10000.0)
    engine.record_trade_result(TradeOutcome(epoch=1000, pnl=-1500.0, equity_after=8500.0))

    for i in range(50):
        result = engine.assess(make_ev(epoch=2000 + i))
        assert result.approved is False
        assert any("Drawdown" in r for r in result.veto_reasons)

    engine.reset(starting_equity=10000.0)
    result = engine.assess(make_ev(epoch=99999))
    assert not any("Drawdown" in r for r in result.veto_reasons)
