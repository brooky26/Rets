import pytest

from configs.opportunity_schema import OpportunityScoringConfig, QualityWeights
from expected_value.types import EVEstimate
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


def make_config(**overrides) -> OpportunityScoringConfig:
    defaults = dict(
        base_confidence_threshold=0.55, threshold_min=0.40, threshold_max=0.85,
        threshold_adjustment_step=0.02, rolling_window_size=20, target_trade_frequency=0.30,
        frequency_band_low=0.7, frequency_band_high=1.3, min_samples_for_adjustment=10,
        adjustment_cooldown=10, per_regime_adjustment=True,
    )
    defaults.update(overrides)
    return OpportunityScoringConfig(**defaults)


def test_quality_score_in_bounds():
    scorer = TradeOpportunityScorer(make_config())
    result = scorer.evaluate(make_ev(), make_risk(), make_regime(), make_probability())
    assert 0.0 <= result.quality_score <= 1.0


def test_higher_ev_pct_increases_quality_score():
    scorer = TradeOpportunityScorer(make_config())
    low = scorer.evaluate(make_ev(epoch=1, expected_value_pct=0.02), make_risk(epoch=1), make_regime(epoch=1), make_probability(epoch=1))
    high = scorer.evaluate(make_ev(epoch=2, expected_value_pct=0.25), make_risk(epoch=2), make_regime(epoch=2), make_probability(epoch=2))
    assert high.quality_score > low.quality_score


def test_higher_regime_confidence_increases_quality_score():
    scorer = TradeOpportunityScorer(make_config())
    low = scorer.evaluate(make_ev(epoch=1), make_risk(epoch=1), make_regime(epoch=1, confidence=0.3), make_probability(epoch=1))
    high = scorer.evaluate(make_ev(epoch=2), make_risk(epoch=2), make_regime(epoch=2, confidence=0.95), make_probability(epoch=2))
    assert high.quality_score > low.quality_score


def test_higher_uncertainty_decreases_quality_score():
    scorer = TradeOpportunityScorer(make_config())
    certain = scorer.evaluate(make_ev(epoch=1), make_risk(epoch=1), make_regime(epoch=1), make_probability(epoch=1, uncertainty=0.05))
    uncertain = scorer.evaluate(make_ev(epoch=2), make_risk(epoch=2), make_regime(epoch=2), make_probability(epoch=2, uncertainty=0.8))
    assert certain.quality_score > uncertain.quality_score


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        QualityWeights(ev_weight=0.5, risk_adjusted_weight=0.5, regime_confidence_weight=0.5,
                        probability_confidence_weight=0.0, certainty_weight=0.0)


def test_negative_ev_always_vetoed_regardless_of_quality_score():
    config = make_config(threshold_min=0.01, base_confidence_threshold=0.01)
    scorer = TradeOpportunityScorer(config)
    result = scorer.evaluate(make_ev(is_positive_ev=False), make_risk(), make_regime(), make_probability())
    assert result.approved is False
    assert any("EV gate" in r for r in result.veto_reasons)


def test_risk_rejection_always_vetoed_regardless_of_quality_score():
    config = make_config(threshold_min=0.01, base_confidence_threshold=0.01)
    scorer = TradeOpportunityScorer(config)
    result = scorer.evaluate(make_ev(), make_risk(approved=False), make_regime(), make_probability())
    assert result.approved is False
    assert any("Risk gate" in r for r in result.veto_reasons)


def test_high_quality_score_with_clean_upstream_gates_is_approved():
    scorer = TradeOpportunityScorer(make_config(base_confidence_threshold=0.3, threshold_min=0.2))
    result = scorer.evaluate(
        make_ev(expected_value_pct=0.20, risk_adjusted_score=1.0), make_risk(),
        make_regime(confidence=0.9), make_probability(confidence=0.9, uncertainty=0.05),
    )
    assert result.approved is True
    assert result.veto_reasons == []


def test_threshold_eases_down_under_sustained_starvation():
    config = make_config(
        base_confidence_threshold=0.70, threshold_min=0.40, target_trade_frequency=0.30,
        rolling_window_size=20, min_samples_for_adjustment=20, adjustment_cooldown=20,
        threshold_adjustment_step=0.05,
    )
    scorer = TradeOpportunityScorer(config)
    initial_threshold = scorer.current_threshold(RegimeLabel.STRONG_TREND)

    for i in range(25):
        scorer.evaluate(
            make_ev(epoch=i, expected_value_pct=0.01, risk_adjusted_score=0.05),
            make_risk(epoch=i), make_regime(epoch=i, confidence=0.3),
            make_probability(epoch=i, confidence=0.55, uncertainty=0.6),
        )

    final_threshold = scorer.current_threshold(RegimeLabel.STRONG_TREND)
    assert final_threshold < initial_threshold


def test_threshold_never_drops_below_floor():
    config = make_config(
        base_confidence_threshold=0.45, threshold_min=0.40, target_trade_frequency=0.30,
        rolling_window_size=15, min_samples_for_adjustment=15, adjustment_cooldown=15,
        threshold_adjustment_step=0.05,
    )
    scorer = TradeOpportunityScorer(config)

    for i in range(200):
        scorer.evaluate(
            make_ev(epoch=i, expected_value_pct=0.01, risk_adjusted_score=0.01),
            make_risk(epoch=i), make_regime(epoch=i, confidence=0.1),
            make_probability(epoch=i, confidence=0.51, uncertainty=0.9),
        )

    assert scorer.current_threshold(RegimeLabel.STRONG_TREND) >= config.threshold_min


def test_threshold_tightens_under_sustained_overtrading():
    config = make_config(
        base_confidence_threshold=0.30, threshold_min=0.20, threshold_max=0.85, target_trade_frequency=0.20,
        rolling_window_size=20, min_samples_for_adjustment=20, adjustment_cooldown=20,
        threshold_adjustment_step=0.05,
    )
    scorer = TradeOpportunityScorer(config)
    initial_threshold = scorer.current_threshold(RegimeLabel.STRONG_TREND)

    for i in range(25):
        scorer.evaluate(
            make_ev(epoch=i, expected_value_pct=0.25, risk_adjusted_score=1.0),
            make_risk(epoch=i), make_regime(epoch=i, confidence=0.95),
            make_probability(epoch=i, confidence=0.95, uncertainty=0.02),
        )

    final_threshold = scorer.current_threshold(RegimeLabel.STRONG_TREND)
    assert final_threshold > initial_threshold


def test_threshold_never_exceeds_ceiling():
    config = make_config(
        base_confidence_threshold=0.80, threshold_max=0.85, target_trade_frequency=0.05,
        rolling_window_size=15, min_samples_for_adjustment=15, adjustment_cooldown=15,
        threshold_adjustment_step=0.05,
    )
    scorer = TradeOpportunityScorer(config)

    for i in range(200):
        scorer.evaluate(
            make_ev(epoch=i, expected_value_pct=0.30, risk_adjusted_score=1.0),
            make_risk(epoch=i), make_regime(epoch=i, confidence=0.99),
            make_probability(epoch=i, confidence=0.99, uncertainty=0.01),
        )

    assert scorer.current_threshold(RegimeLabel.STRONG_TREND) <= config.threshold_max


def test_no_adjustment_before_cooldown_elapses():
    config = make_config(
        base_confidence_threshold=0.70, adjustment_cooldown=100, min_samples_for_adjustment=5,
        rolling_window_size=20,
    )
    scorer = TradeOpportunityScorer(config)
    initial = scorer.current_threshold(RegimeLabel.STRONG_TREND)

    for i in range(30):
        scorer.evaluate(
            make_ev(epoch=i, expected_value_pct=0.01), make_risk(epoch=i),
            make_regime(epoch=i, confidence=0.1), make_probability(epoch=i, confidence=0.51, uncertainty=0.9),
        )
    assert scorer.current_threshold(RegimeLabel.STRONG_TREND) == initial


def test_no_adjustment_before_min_samples():
    config = make_config(
        base_confidence_threshold=0.70, adjustment_cooldown=5, min_samples_for_adjustment=100,
        rolling_window_size=20,
    )
    scorer = TradeOpportunityScorer(config)
    initial = scorer.current_threshold(RegimeLabel.STRONG_TREND)

    for i in range(30):
        scorer.evaluate(
            make_ev(epoch=i, expected_value_pct=0.01), make_risk(epoch=i),
            make_regime(epoch=i, confidence=0.1), make_probability(epoch=i, confidence=0.51, uncertainty=0.9),
        )
    assert scorer.current_threshold(RegimeLabel.STRONG_TREND) == initial


def test_per_regime_thresholds_are_independent():
    config = make_config(
        base_confidence_threshold=0.70, threshold_min=0.40, target_trade_frequency=0.30,
        rolling_window_size=20, min_samples_for_adjustment=20, adjustment_cooldown=20,
        threshold_adjustment_step=0.05, per_regime_adjustment=True,
    )
    scorer = TradeOpportunityScorer(config)

    for i in range(25):
        scorer.evaluate(
            make_ev(epoch=i, expected_value_pct=0.01), make_risk(epoch=i),
            make_regime(epoch=i, regime=RegimeLabel.STRONG_TREND, confidence=0.1),
            make_probability(epoch=i, confidence=0.51, uncertainty=0.9),
        )
    for i in range(25):
        scorer.evaluate(
            make_ev(epoch=1000 + i, expected_value_pct=0.25, risk_adjusted_score=1.0),
            make_risk(epoch=1000 + i),
            make_regime(epoch=1000 + i, regime=RegimeLabel.RANGE, confidence=0.95),
            make_probability(epoch=1000 + i, confidence=0.95, uncertainty=0.02),
        )

    trend_threshold = scorer.current_threshold(RegimeLabel.STRONG_TREND)
    range_threshold = scorer.current_threshold(RegimeLabel.RANGE)
    assert trend_threshold < 0.70
    assert range_threshold > 0.70


def test_global_mode_shares_one_threshold_across_regimes():
    config = make_config(per_regime_adjustment=False, base_confidence_threshold=0.55)
    scorer = TradeOpportunityScorer(config)
    scorer.evaluate(make_ev(epoch=1), make_risk(epoch=1), make_regime(epoch=1, regime=RegimeLabel.STRONG_TREND), make_probability(epoch=1))
    scorer.evaluate(make_ev(epoch=2), make_risk(epoch=2), make_regime(epoch=2, regime=RegimeLabel.RANGE), make_probability(epoch=2))
    assert scorer.current_threshold(RegimeLabel.STRONG_TREND) == scorer.current_threshold(RegimeLabel.RANGE)


def test_frequency_stats_none_before_any_evaluations():
    scorer = TradeOpportunityScorer(make_config())
    assert scorer.frequency_stats(RegimeLabel.STRONG_TREND) is None


def test_frequency_stats_reflects_history():
    scorer = TradeOpportunityScorer(make_config(base_confidence_threshold=0.3, threshold_min=0.2))
    for i in range(5):
        scorer.evaluate(
            make_ev(epoch=i, expected_value_pct=0.20, risk_adjusted_score=1.0), make_risk(epoch=i),
            make_regime(epoch=i, confidence=0.9), make_probability(epoch=i, confidence=0.9, uncertainty=0.05),
        )
    stats = scorer.frequency_stats(RegimeLabel.STRONG_TREND)
    assert stats is not None
    assert stats.window_size == 5
    assert stats.approved_count == 5
    assert stats.observed_frequency == pytest.approx(1.0)
