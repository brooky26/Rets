import numpy as np
import pytest

from configs.monte_carlo_schema import MonteCarloPricePathConfig
from expected_value.types import EVEstimate
from monte_carlo.price_paths import (
    MonteCarloPricePathSimulator,
    estimate_gbm_parameters_from_returns,
    refine_ev_with_monte_carlo,
    simulate_gbm_paths,
)


def make_config(**overrides) -> MonteCarloPricePathConfig:
    defaults = dict(n_paths=2000, horizon_ticks=10, random_seed=42)
    defaults.update(overrides)
    return MonteCarloPricePathConfig(**defaults)


def test_simulate_gbm_paths_shape_and_start():
    rng = np.random.default_rng(1)
    paths = simulate_gbm_paths(100.0, 0.0001, 0.01, horizon_ticks=20, n_paths=500, rng=rng)
    assert paths.shape == (500, 21)
    assert np.all(paths[:, 0] == 100.0)
    assert np.all(paths > 0)  # GBM never goes negative


def test_positive_drift_favors_up_direction():
    sim = MonteCarloPricePathSimulator(make_config())
    result = sim.simulate(
        symbol="TEST", epoch=1, current_price=100.0, mu_per_tick=0.005, sigma_per_tick=0.01, direction=1,
    )
    assert result.prob_favorable > 0.5


def test_negative_drift_favors_down_direction():
    sim = MonteCarloPricePathSimulator(make_config())
    result = sim.simulate(
        symbol="TEST", epoch=1, current_price=100.0, mu_per_tick=-0.005, sigma_per_tick=0.01, direction=-1,
    )
    assert result.prob_favorable > 0.5


def test_zero_drift_near_half_probability():
    sim = MonteCarloPricePathSimulator(make_config(n_paths=5000))
    result = sim.simulate(
        symbol="TEST", epoch=1, current_price=100.0, mu_per_tick=0.0, sigma_per_tick=0.01, direction=1,
    )
    assert abs(result.prob_favorable - 0.5) < 0.05


def test_invalid_direction_returns_nan_result():
    sim = MonteCarloPricePathSimulator(make_config())
    result = sim.simulate(
        symbol="TEST", epoch=1, current_price=100.0, mu_per_tick=0.0, sigma_per_tick=0.01, direction=0,
    )
    assert not result.is_valid


def test_nan_sigma_returns_nan_result():
    sim = MonteCarloPricePathSimulator(make_config())
    result = sim.simulate(
        symbol="TEST", epoch=1, current_price=100.0, mu_per_tick=float("nan"), sigma_per_tick=0.01, direction=1,
    )
    assert not result.is_valid


def test_mfe_mae_ordering():
    sim = MonteCarloPricePathSimulator(make_config())
    result = sim.simulate(
        symbol="TEST", epoch=1, current_price=100.0, mu_per_tick=0.0, sigma_per_tick=0.02, direction=1,
    )
    assert result.mfe_mean >= 0
    assert result.mae_mean <= 0
    assert result.mae_p95 >= 0


def test_estimate_gbm_parameters_from_returns():
    returns = np.array([0.01, -0.005, 0.02, -0.01, 0.005])
    mu, sigma = estimate_gbm_parameters_from_returns(returns)
    assert mu == pytest.approx(np.mean(returns))
    assert sigma == pytest.approx(np.std(returns, ddof=1))


def test_estimate_gbm_parameters_insufficient_history():
    mu, sigma = estimate_gbm_parameters_from_returns(np.array([0.01]))
    assert mu != mu and sigma != sigma  # NaN


def make_ev(direction=1, probability_used=0.55) -> EVEstimate:
    stake, payout = 10.0, 19.0
    profit_if_win = payout - stake
    ev_val = probability_used * payout - stake
    return EVEstimate(
        symbol="TEST", epoch=1, direction=direction, probability_used=probability_used,
        stake=stake, payout=payout, expected_value=ev_val, expected_value_pct=ev_val / stake,
        reward_to_risk=profit_if_win / stake, win_component=probability_used * profit_if_win,
        loss_component=(1 - probability_used) * -stake, outcome_std=1.0, risk_adjusted_score=0.5,
        is_positive_ev=True, rejection_reason=None,
    )


def test_refine_ev_with_zero_blend_weight_is_unchanged():
    sim = MonteCarloPricePathSimulator(make_config())
    ev = make_ev()
    mc = sim.simulate(symbol="TEST", epoch=1, current_price=100.0, mu_per_tick=0.01, sigma_per_tick=0.01, direction=1)
    refined = refine_ev_with_monte_carlo(ev, mc, blend_weight=0.0)
    assert refined.probability_used == pytest.approx(ev.probability_used)
    assert refined.expected_value == pytest.approx(ev.expected_value)


def test_refine_ev_with_full_blend_weight_uses_mc_probability():
    sim = MonteCarloPricePathSimulator(make_config())
    ev = make_ev(probability_used=0.55)
    mc = sim.simulate(symbol="TEST", epoch=1, current_price=100.0, mu_per_tick=0.01, sigma_per_tick=0.005, direction=1)
    refined = refine_ev_with_monte_carlo(ev, mc, blend_weight=1.0)
    assert refined.probability_used == pytest.approx(mc.prob_favorable)


def test_refine_ev_direction_mismatch_raises():
    sim = MonteCarloPricePathSimulator(make_config())
    ev = make_ev(direction=1)
    mc = sim.simulate(symbol="TEST", epoch=1, current_price=100.0, mu_per_tick=0.0, sigma_per_tick=0.01, direction=-1)
    with pytest.raises(ValueError):
        refine_ev_with_monte_carlo(ev, mc, blend_weight=0.5)
