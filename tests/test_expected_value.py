import math

import pytest

from configs.ev_schema import ExpectedValueConfig
from expected_value.engine import ExpectedValueEngine
from expected_value.types import ContractSpec, ContractType
from probability.types import ProbabilityEstimate


def make_probability(
    symbol="STPRNG100", epoch=1000, prob_up=0.6, expected_direction=1, uncertainty=0.1
) -> ProbabilityEstimate:
    return ProbabilityEstimate(
        symbol=symbol,
        epoch=epoch,
        model_name="test_model",
        prob_up=prob_up,
        prob_down=1.0 - prob_up,
        uncertainty=uncertainty,
        expected_direction=expected_direction,
        confidence=max(prob_up, 1 - prob_up),
    )


def make_contract(stake=10.0, payout=19.0, contract_type=ContractType.RISE_FALL) -> ContractSpec:
    return ContractSpec(contract_type=contract_type, stake=stake, payout=payout, duration_ticks=5)


@pytest.fixture
def engine() -> ExpectedValueEngine:
    return ExpectedValueEngine(ExpectedValueConfig())


# --------------------------------------------------------------------- #
# Formula correctness (hand-computed values)
# --------------------------------------------------------------------- #


def test_ev_matches_hand_computed_value(engine):
    # p=0.6, stake=10, payout=19 -> EV = 0.6*19 - 10 = 11.4 - 10 = 1.4
    prob = make_probability(prob_up=0.6, expected_direction=1)
    contract = make_contract(stake=10.0, payout=19.0)
    result = engine.evaluate(prob, contract)
    assert result.expected_value == pytest.approx(1.4)
    assert result.expected_value_pct == pytest.approx(0.14)


def test_reward_to_risk_matches_hand_computed_value(engine):
    # profit_if_win = payout - stake = 19 - 10 = 9; reward_to_risk = 9/10 = 0.9
    prob = make_probability(prob_up=0.6, expected_direction=1)
    contract = make_contract(stake=10.0, payout=19.0)
    result = engine.evaluate(prob, contract)
    assert result.reward_to_risk == pytest.approx(0.9)


def test_win_and_loss_components_sum_to_ev(engine):
    prob = make_probability(prob_up=0.7, expected_direction=1)
    contract = make_contract(stake=25.0, payout=45.0)
    result = engine.evaluate(prob, contract)
    assert result.win_component + result.loss_component == pytest.approx(result.expected_value)


def test_negative_ev_detected(engine):
    # p=0.4, stake=10, payout=19 -> EV = 0.4*19 - 10 = 7.6-10 = -2.4
    prob = make_probability(prob_up=0.6, expected_direction=-1)  # betting DOWN with prob_down=0.4
    contract = make_contract(stake=10.0, payout=19.0)
    result = engine.evaluate(prob, contract)
    assert result.expected_value == pytest.approx(-2.4)
    assert result.is_positive_ev is False
    assert result.rejection_reason is not None


def test_uses_prob_down_when_direction_is_negative(engine):
    prob = make_probability(prob_up=0.3, expected_direction=-1)
    contract = make_contract()
    result = engine.evaluate(prob, contract)
    assert result.probability_used == pytest.approx(0.7)  # prob_down = 1 - 0.3


def test_uses_prob_up_when_direction_is_positive(engine):
    prob = make_probability(prob_up=0.65, expected_direction=1)
    contract = make_contract()
    result = engine.evaluate(prob, contract)
    assert result.probability_used == pytest.approx(0.65)


def test_outcome_std_matches_hand_computed_bernoulli_formula(engine):
    p = 0.6
    stake, payout = 10.0, 19.0
    profit_if_win = payout - stake  # 9
    loss_if_lose = -stake  # -10
    expected_std = math.sqrt(p * (1 - p) * (profit_if_win - loss_if_lose) ** 2)

    prob = make_probability(prob_up=p, expected_direction=1)
    contract = make_contract(stake=stake, payout=payout)
    result = engine.evaluate(prob, contract)
    assert result.outcome_std == pytest.approx(expected_std)


def test_risk_adjusted_score_is_ev_over_std(engine):
    prob = make_probability(prob_up=0.6, expected_direction=1)
    contract = make_contract(stake=10.0, payout=19.0)
    result = engine.evaluate(prob, contract)
    assert result.risk_adjusted_score == pytest.approx(
        result.expected_value / result.outcome_std
    )


# --------------------------------------------------------------------- #
# Gating logic
# --------------------------------------------------------------------- #


def test_positive_ev_passes_default_gate(engine):
    prob = make_probability(prob_up=0.6, expected_direction=1)
    contract = make_contract(stake=10.0, payout=19.0)  # EV = +1.4
    result = engine.evaluate(prob, contract)
    assert result.is_positive_ev is True
    assert result.rejection_reason is None


def test_exactly_zero_ev_passes_default_threshold():
    # min_ev_threshold defaults to 0.0, gate is >=, so EV==0 should pass.
    engine = ExpectedValueEngine(ExpectedValueConfig(min_ev_threshold=0.0))
    # Solve for p such that p*payout - stake == 0: p = stake/payout = 10/20 = 0.5
    prob = make_probability(prob_up=0.5, expected_direction=1)
    contract = make_contract(stake=10.0, payout=20.0)
    result = engine.evaluate(prob, contract)
    assert result.expected_value == pytest.approx(0.0, abs=1e-9)
    assert result.is_positive_ev is True


def test_min_ev_threshold_above_zero_rejects_small_positive_ev():
    engine = ExpectedValueEngine(ExpectedValueConfig(min_ev_threshold=2.0))
    prob = make_probability(prob_up=0.6, expected_direction=1)
    contract = make_contract(stake=10.0, payout=19.0)  # EV = +1.4, below threshold of 2.0
    result = engine.evaluate(prob, contract)
    assert result.is_positive_ev is False
    assert "threshold" in result.rejection_reason


def test_min_reward_to_risk_rejects_low_payout_contract():
    engine = ExpectedValueEngine(ExpectedValueConfig(min_ev_threshold=0.0, min_reward_to_risk=1.0))
    # Very high probability but low payout multiple -> positive EV but low reward:risk.
    # Need p*payout > stake for positive EV: p > 10/10.5 ≈ 0.952
    prob = make_probability(prob_up=0.97, expected_direction=1)
    contract = make_contract(stake=10.0, payout=10.5)  # profit_if_win=0.5, r:r=0.05
    result = engine.evaluate(prob, contract)
    assert result.expected_value > 0  # EV itself is positive
    assert result.is_positive_ev is False  # but gated out by reward:risk
    assert "Reward-to-risk" in result.rejection_reason


def test_min_probability_confidence_rejects_weak_edge():
    engine = ExpectedValueEngine(ExpectedValueConfig(min_probability_confidence=0.55))
    prob = make_probability(prob_up=0.52, expected_direction=1)
    contract = make_contract(stake=10.0, payout=20.0)  # slightly positive EV
    result = engine.evaluate(prob, contract)
    assert result.is_positive_ev is False
    assert "confidence" in result.rejection_reason


# --------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------- #


def test_invalid_probability_returns_nan_and_rejected(engine):
    prob = make_probability(prob_up=float("nan"), expected_direction=1)
    contract = make_contract()
    result = engine.evaluate(prob, contract)
    assert not result.is_valid
    assert result.is_positive_ev is False
    assert result.rejection_reason is not None


def test_zero_direction_returns_no_edge_result(engine):
    prob = make_probability(prob_up=0.5, expected_direction=0)
    contract = make_contract()
    result = engine.evaluate(prob, contract)
    assert result.direction == 0
    assert result.is_positive_ev is False
    assert "No directional edge" in result.rejection_reason


def test_contract_spec_rejects_nonpositive_stake():
    with pytest.raises(ValueError, match="stake"):
        ContractSpec(contract_type=ContractType.RISE_FALL, stake=0.0, payout=10.0, duration_ticks=5)


def test_contract_spec_rejects_negative_payout():
    with pytest.raises(ValueError, match="payout"):
        ContractSpec(contract_type=ContractType.RISE_FALL, stake=10.0, payout=-1.0, duration_ticks=5)


def test_contract_spec_rejects_nonpositive_duration():
    with pytest.raises(ValueError, match="duration"):
        ContractSpec(contract_type=ContractType.RISE_FALL, stake=10.0, payout=15.0, duration_ticks=0)


def test_profit_and_loss_properties():
    contract = make_contract(stake=10.0, payout=19.0)
    assert contract.profit_if_win == pytest.approx(9.0)
    assert contract.loss_if_lose == pytest.approx(-10.0)
