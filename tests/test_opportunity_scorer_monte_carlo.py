import pytest

from configs.opportunity_schema import OpportunityScoringConfig, QualityWeights
from expected_value.types import EVEstimate
from monte_carlo.types import PricePathSimulationResult
from opportunity.scorer import TradeOpportunityScorer
from probability.types import ProbabilityEstimate
from regime.types import RegimeClassification, RegimeLabel
from risk.types import RiskAssessment


def make_ev(epoch=1000, is_positive_ev=True, expected_value_pct=0.10, risk_adjusted_score=0.5) -> EVEstimate:
    return EVEstimate(
        symbol="STPRNG100", epoch=epoch, direction=1, probability_used=0.6, stake=10.0, payout=19.0,
        expected_value=expected_value_pct * 10.0, expected_value_pct=expected_value_pct,
        reward_to_risk=0.9, win_component=0.0, loss_component=0.0, outcome_std=1.0,
        risk_adjusted_score=risk_adjusted_score, is_positive_ev=is_positive_ev,
        rejection_reason=None if is_positive_ev else "EV too low",
    )


def make_risk(epoch=1000, approved=True) -> RiskAssessment:
    return RiskAssessment(
        symbol="STPRNG100", epoch=epoch, approved=approved, recommended_stake=10.0 if approved else 0.0,
        kelly_fraction_raw=0.1, kelly_fraction_applied=0.025, risk_of_ruin=0.001,
        current_drawdown_pct=0.0, daily_loss_pct=0.0, consecutive_losses=0,
        expected_shortfall_pct=float("nan"),
        veto_reasons=[] if approved else ["Circuit breaker triggered"],
    )


def make_regime(epoch=1000, regime=RegimeLabel.STRONG_TREND, confidence=0.8) -> RegimeClassification:
    return RegimeClassification(
        symbol="STPRNG100", epoch=epoch, detector_name="test", regime=regime,
        confidence=confidence, probabilities={regime: confidence},
    )


def make_probability(epoch=1000, confidence=0.7, uncertainty=0.2) -> ProbabilityEstimate:
    return ProbabilityEstimate(
        symbol="STPRNG100", epoch=epoch, model_name="test", prob_up=confidence, prob_down=1 - confidence,
        uncertainty=uncertainty, expected_direction=1, confidence=confidence,
    )


def make_mc_result(prob_favorable=0.9) -> PricePathSimulationResult:
    return PricePathSimulationResult(
        symbol="STPRNG100", epoch=1000, direction=1, n_paths=1000, horizon_ticks=10, current_price=100.0,
        mu_per_tick=0.001, sigma_per_tick=0.01, prob_favorable=prob_favorable, expected_favorable_duration_ticks=3.0,
        terminal_return_mean=0.01, terminal_return_std=0.03, mfe_mean=0.02, mfe_p95=0.05, mae_mean=-0.01, mae_p95=0.02,
    )


def test_without_mc_result_component_is_zero_and_backward_compatible():
    config = OpportunityScoringConfig()
    scorer = TradeOpportunityScorer(config)
    result = scorer.evaluate(make_ev(), make_risk(), make_regime(), make_probability())
    assert result.components.mc_confidence_component == 0.0


def test_with_mc_result_but_zero_weight_does_not_change_score():
    config = OpportunityScoringConfig()  # mc_confidence_weight defaults to 0.0
    scorer = TradeOpportunityScorer(config)
    without = scorer.evaluate(make_ev(), make_risk(), make_regime(), make_probability())
    with_mc = scorer.evaluate(make_ev(), make_risk(), make_regime(), make_probability(), make_mc_result())
    assert with_mc.components.mc_confidence_component > 0.0  # component itself is computed...
    assert with_mc.quality_score == pytest.approx(without.quality_score)  # ...but doesn't affect score at weight 0


def test_with_mc_result_and_nonzero_weight_changes_score():
    weights = QualityWeights(
        ev_weight=0.25, risk_adjusted_weight=0.20, regime_confidence_weight=0.15,
        probability_confidence_weight=0.15, certainty_weight=0.15, mc_confidence_weight=0.10,
    )
    config = OpportunityScoringConfig(quality_weights=weights)
    scorer = TradeOpportunityScorer(config)

    high_mc = scorer.evaluate(make_ev(epoch=1), make_risk(epoch=1), make_regime(epoch=1), make_probability(epoch=1), make_mc_result(prob_favorable=0.95))
    low_mc = scorer.evaluate(make_ev(epoch=2), make_risk(epoch=2), make_regime(epoch=2), make_probability(epoch=2), make_mc_result(prob_favorable=0.55))
    assert high_mc.quality_score > low_mc.quality_score


def test_invalid_mc_result_contributes_zero_component():
    config = OpportunityScoringConfig()
    scorer = TradeOpportunityScorer(config)
    invalid_mc = PricePathSimulationResult(
        symbol="STPRNG100", epoch=1000, direction=0, n_paths=1000, horizon_ticks=10, current_price=100.0,
        mu_per_tick=float("nan"), sigma_per_tick=0.01, prob_favorable=float("nan"), expected_favorable_duration_ticks=float("nan"),
        terminal_return_mean=float("nan"), terminal_return_std=float("nan"), mfe_mean=float("nan"), mfe_p95=float("nan"),
        mae_mean=float("nan"), mae_p95=float("nan"),
    )
    result = scorer.evaluate(make_ev(), make_risk(), make_regime(), make_probability(), invalid_mc)
    assert result.components.mc_confidence_component == 0.0
