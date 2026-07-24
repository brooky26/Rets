import pytest

from configs.execution_schema import ExecutionConfig
from execution.engine import ExecutionConfigurationError, ExecutionEngine
from execution.types import ExecutionMode
from expected_value.types import ContractSpec, ContractType, EVEstimate
from opportunity.types import QualityScoreComponents, TradeOpportunity
from regime.types import RegimeLabel
from risk.types import RiskAssessment


class FakeBrokerClient:
    def __init__(self, proposal: dict | None = None, buy_result: dict | None = None, raise_on_proposal=False, raise_on_buy=False):
        self._proposal = proposal or {"id": "prop123", "ask_price": 10.0, "payout": 19.0}
        self._buy_result = buy_result or {"contract_id": "c789", "buy_price": 10.0, "payout": 19.0}
        self._raise_on_proposal = raise_on_proposal
        self._raise_on_buy = raise_on_buy
        self.proposal_calls = []
        self.buy_calls = []

    async def fetch_proposal(self, symbol, contract_type_code, stake, duration_ticks, currency):
        self.proposal_calls.append((symbol, contract_type_code, stake, duration_ticks, currency))
        if self._raise_on_proposal:
            raise RuntimeError("simulated network failure")
        return self._proposal

    async def buy(self, proposal_id, price):
        self.buy_calls.append((proposal_id, price))
        if self._raise_on_buy:
            raise RuntimeError("simulated buy failure")
        return self._buy_result


def make_opportunity(approved=True, veto_reasons=None) -> TradeOpportunity:
    return TradeOpportunity(
        symbol="STPRNG100", epoch=1000, regime=RegimeLabel.STRONG_TREND,
        quality_score=0.7,
        components=QualityScoreComponents(0.6, 0.6, 0.8, 0.7, 0.8),
        threshold_applied=0.55, approved=approved, veto_reasons=veto_reasons or [],
    )


def make_ev(direction=1, reward_to_risk=0.9) -> EVEstimate:
    return EVEstimate(
        symbol="STPRNG100", epoch=1000, direction=direction, probability_used=0.6,
        stake=10.0, payout=19.0, expected_value=1.4, expected_value_pct=0.14,
        reward_to_risk=reward_to_risk, win_component=0.0, loss_component=0.0,
        outcome_std=1.0, risk_adjusted_score=1.0, is_positive_ev=True, rejection_reason=None,
    )


def make_risk(recommended_stake=10.0, approved=True) -> RiskAssessment:
    return RiskAssessment(
        symbol="STPRNG100", epoch=1000, approved=approved, recommended_stake=recommended_stake,
        kelly_fraction_raw=0.1, kelly_fraction_applied=0.025, risk_of_ruin=0.001,
        current_drawdown_pct=0.0, daily_loss_pct=0.0, consecutive_losses=0,
        expected_shortfall_pct=float("nan"), veto_reasons=[],
    )


def make_contract(contract_type=ContractType.RISE_FALL) -> ContractSpec:
    return ContractSpec(contract_type=contract_type, stake=10.0, payout=19.0, duration_ticks=5)


def test_paper_mode_constructs_without_broker_client():
    engine = ExecutionEngine(ExecutionConfig(mode="paper"), platform_environment="development")
    assert engine.mode == ExecutionMode.PAPER


def test_live_mode_requires_matching_platform_environment():
    with pytest.raises(ExecutionConfigurationError, match="must agree"):
        ExecutionEngine(
            ExecutionConfig(mode="live"), platform_environment="development",
            broker_client=FakeBrokerClient(),
        )


def test_live_mode_requires_broker_client():
    with pytest.raises(ExecutionConfigurationError, match="no broker_client"):
        ExecutionEngine(ExecutionConfig(mode="live"), platform_environment="live", broker_client=None)


def test_live_mode_constructs_when_both_rails_agree():
    engine = ExecutionEngine(
        ExecutionConfig(mode="live"), platform_environment="live", broker_client=FakeBrokerClient()
    )
    assert engine.mode == ExecutionMode.LIVE


@pytest.mark.asyncio
async def test_paper_buy_simulates_without_calling_broker():
    engine = ExecutionEngine(ExecutionConfig(mode="paper"), platform_environment="development")
    decision = await engine.execute(make_opportunity(), make_ev(), make_risk(), make_contract())
    assert decision.action == "buy"
    assert decision.contract_id is None
    assert decision.mode == ExecutionMode.PAPER


@pytest.mark.asyncio
async def test_paper_buy_payout_scales_with_reward_to_risk():
    engine = ExecutionEngine(ExecutionConfig(mode="paper"), platform_environment="development")
    decision = await engine.execute(make_opportunity(), make_ev(reward_to_risk=0.9), make_risk(recommended_stake=20.0), make_contract())
    assert decision.payout == pytest.approx(20.0 * 1.9)


@pytest.mark.asyncio
async def test_unapproved_opportunity_is_skipped():
    engine = ExecutionEngine(ExecutionConfig(mode="paper"), platform_environment="development")
    opp = make_opportunity(approved=False, veto_reasons=["quality score too low"])
    decision = await engine.execute(opp, make_ev(), make_risk(), make_contract())
    assert decision.action == "skip"
    assert "quality score too low" in decision.reason


@pytest.mark.asyncio
async def test_zero_recommended_stake_is_skipped():
    engine = ExecutionEngine(ExecutionConfig(mode="paper"), platform_environment="development")
    decision = await engine.execute(make_opportunity(), make_ev(), make_risk(recommended_stake=0.0), make_contract())
    assert decision.action == "skip"


@pytest.mark.asyncio
async def test_unsupported_contract_type_returns_error_not_silent_mismap():
    engine = ExecutionEngine(ExecutionConfig(mode="paper"), platform_environment="development")
    decision = await engine.execute(make_opportunity(), make_ev(), make_risk(), make_contract(contract_type=ContractType.TOUCH_NO_TOUCH))
    assert decision.action == "error"
    assert "does not yet support" in decision.reason


@pytest.mark.asyncio
async def test_live_buy_success_returns_real_contract_id():
    broker = FakeBrokerClient(
        proposal={"id": "prop123", "ask_price": 10.0, "payout": 19.0},
        buy_result={"contract_id": "c789", "buy_price": 10.0, "payout": 19.0},
    )
    engine = ExecutionEngine(ExecutionConfig(mode="live"), platform_environment="live", broker_client=broker)
    decision = await engine.execute(make_opportunity(), make_ev(reward_to_risk=0.9), make_risk(), make_contract())
    assert decision.action == "buy"
    assert decision.contract_id == "c789"
    assert decision.mode == ExecutionMode.LIVE
    assert len(broker.proposal_calls) == 1
    assert len(broker.buy_calls) == 1


@pytest.mark.asyncio
async def test_live_buy_uses_correct_contract_type_code_for_direction():
    broker = FakeBrokerClient()
    engine = ExecutionEngine(ExecutionConfig(mode="live"), platform_environment="live", broker_client=broker)
    await engine.execute(make_opportunity(), make_ev(direction=1), make_risk(), make_contract())
    assert broker.proposal_calls[0][1] == "CALL"

    broker2 = FakeBrokerClient()
    engine2 = ExecutionEngine(ExecutionConfig(mode="live"), platform_environment="live", broker_client=broker2)
    await engine2.execute(make_opportunity(), make_ev(direction=-1), make_risk(), make_contract())
    assert broker2.proposal_calls[0][1] == "PUT"


@pytest.mark.asyncio
async def test_live_buy_aborts_on_excessive_payout_drift():
    broker = FakeBrokerClient(proposal={"id": "prop123", "ask_price": 10.0, "payout": 10.5})
    config = ExecutionConfig(mode="live", max_payout_drift_pct=0.15)
    engine = ExecutionEngine(config, platform_environment="live", broker_client=broker)
    decision = await engine.execute(make_opportunity(), make_ev(reward_to_risk=0.9), make_risk(), make_contract())
    assert decision.action == "skip"
    assert "drifted" in decision.reason
    assert len(broker.buy_calls) == 0


@pytest.mark.asyncio
async def test_live_buy_within_drift_tolerance_proceeds():
    broker = FakeBrokerClient(proposal={"id": "prop123", "ask_price": 10.0, "payout": 18.5})
    config = ExecutionConfig(mode="live", max_payout_drift_pct=0.15)
    engine = ExecutionEngine(config, platform_environment="live", broker_client=broker)
    decision = await engine.execute(make_opportunity(), make_ev(reward_to_risk=0.9), make_risk(), make_contract())
    assert decision.action == "buy"


@pytest.mark.asyncio
async def test_live_buy_aborts_on_excessive_slippage():
    broker = FakeBrokerClient(proposal={"id": "prop123", "ask_price": 15.0, "payout": 28.5})
    config = ExecutionConfig(mode="live", max_payout_drift_pct=0.50, price_slippage_tolerance_pct=0.0)
    engine = ExecutionEngine(config, platform_environment="live", broker_client=broker)
    decision = await engine.execute(make_opportunity(), make_ev(reward_to_risk=0.9), make_risk(recommended_stake=10.0), make_contract())
    assert decision.action == "skip"
    assert "slippage" in decision.reason or "ask_price" in decision.reason
    assert len(broker.buy_calls) == 0


@pytest.mark.asyncio
async def test_proposal_failure_returns_error_decision_not_raised_exception():
    broker = FakeBrokerClient(raise_on_proposal=True)
    engine = ExecutionEngine(ExecutionConfig(mode="live"), platform_environment="live", broker_client=broker)
    decision = await engine.execute(make_opportunity(), make_ev(), make_risk(), make_contract())
    assert decision.action == "error"
    assert "Proposal request failed" in decision.reason


@pytest.mark.asyncio
async def test_buy_failure_returns_error_decision_not_raised_exception():
    broker = FakeBrokerClient(raise_on_buy=True)
    engine = ExecutionEngine(ExecutionConfig(mode="live"), platform_environment="live", broker_client=broker)
    decision = await engine.execute(make_opportunity(), make_ev(), make_risk(), make_contract())
    assert decision.action == "error"
    assert "Buy request failed" in decision.reason


@pytest.mark.asyncio
async def test_live_buy_uses_recommended_stake_not_hypothetical_contract_stake():
    broker = FakeBrokerClient()
    engine = ExecutionEngine(ExecutionConfig(mode="live"), platform_environment="live", broker_client=broker)
    await engine.execute(make_opportunity(), make_ev(), make_risk(recommended_stake=37.5), make_contract())
    assert broker.proposal_calls[0][2] == 37.5
