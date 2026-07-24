import math

import pytest

from configs.regime_schema import RuleBasedRegimeConfig
from regime.rule_based import RuleBasedRegimeDetector
from regime.types import RegimeLabel
from state_encoder.types import MarketState


def make_state(
    symbol="STPRNG100",
    epoch=1000,
    trend=0.0,
    momentum=0.0,
    acceleration=0.0,
    volatility=0.0,
    noise=0.0,
    persistence=0.0,
    compression_expansion=0.0,
    complexity=0.0,
    uncertainty=0.0,
    liquidity=0.0,
    market_phase=0.0,
) -> MarketState:
    return MarketState(
        symbol=symbol,
        epoch=epoch,
        trend=trend,
        momentum=momentum,
        acceleration=acceleration,
        volatility=volatility,
        noise=noise,
        persistence=persistence,
        compression_expansion=compression_expansion,
        complexity=complexity,
        uncertainty=uncertainty,
        liquidity=liquidity,
        market_phase=market_phase,
    )


@pytest.fixture
def detector() -> RuleBasedRegimeDetector:
    return RuleBasedRegimeDetector(RuleBasedRegimeConfig())


def test_strong_trend_detected(detector):
    state = make_state(trend=0.8, persistence=0.5)
    result = detector.classify(state)
    assert result.regime == RegimeLabel.STRONG_TREND
    assert result.confidence > 0.5


def test_weak_trend_detected(detector):
    state = make_state(trend=0.3, persistence=0.0)
    result = detector.classify(state)
    assert result.regime == RegimeLabel.WEAK_TREND


def test_mean_reversion_detected(detector):
    state = make_state(trend=0.1, persistence=-0.5)
    result = detector.classify(state)
    assert result.regime == RegimeLabel.MEAN_REVERSION


def test_compression_detected(detector):
    state = make_state(trend=0.05, persistence=0.0, compression_expansion=-0.7)
    result = detector.classify(state)
    assert result.regime == RegimeLabel.COMPRESSION


def test_expansion_without_trend_confirmation(detector):
    state = make_state(trend=0.1, persistence=0.0, compression_expansion=0.7)
    result = detector.classify(state)
    assert result.regime == RegimeLabel.EXPANSION


def test_breakout_requires_expansion_and_trend(detector):
    state = make_state(trend=0.7, compression_expansion=0.6)
    result = detector.classify(state)
    assert result.regime == RegimeLabel.BREAKOUT


def test_high_volatility_detected(detector):
    state = make_state(trend=0.0, persistence=0.0, compression_expansion=0.0, volatility=0.8)
    result = detector.classify(state)
    assert result.regime == RegimeLabel.HIGH_VOLATILITY


def test_low_volatility_detected(detector):
    state = make_state(trend=0.0, persistence=0.0, compression_expansion=0.0, volatility=-0.8)
    result = detector.classify(state)
    assert result.regime == RegimeLabel.LOW_VOLATILITY


def test_random_walk_detected(detector):
    state = make_state(trend=0.01, persistence=0.02, compression_expansion=0.0, volatility=0.0)
    result = detector.classify(state)
    assert result.regime == RegimeLabel.RANDOM_WALK


def test_range_fallback(detector):
    # Nothing extreme, but persistence just outside the random-walk band.
    state = make_state(trend=0.1, persistence=0.2, compression_expansion=0.1, volatility=0.1)
    result = detector.classify(state)
    assert result.regime == RegimeLabel.RANGE


def test_confidence_bounded_zero_to_one(detector):
    for trend in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        state = make_state(trend=trend)
        result = detector.classify(state)
        assert 0.0 <= result.confidence <= 1.0


def test_invalid_state_returns_nan_confidence(detector):
    state = make_state(trend=float("nan"))
    result = detector.classify(state)
    assert math.isnan(result.confidence)
    assert result.probabilities == {}


def test_probabilities_sum_to_confidence_on_argmax_only():
    detector = RuleBasedRegimeDetector(RuleBasedRegimeConfig())
    state = make_state(trend=0.8, persistence=0.5)
    result = detector.classify(state)
    assert result.probabilities[result.regime] == pytest.approx(result.confidence)
    other_mass = sum(v for k, v in result.probabilities.items() if k != result.regime)
    assert other_mass == pytest.approx(0.0)


def test_detector_name_set():
    detector = RuleBasedRegimeDetector(RuleBasedRegimeConfig())
    assert detector.name == "rule_based"


def test_stronger_trend_yields_higher_confidence(detector):
    weak = detector.classify(make_state(trend=0.61))
    strong = detector.classify(make_state(trend=0.99))
    assert strong.confidence >= weak.confidence
