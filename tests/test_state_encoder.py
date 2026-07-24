import math

import numpy as np
import pytest

from configs.state_encoder_schema import StateEncoderConfig
from features.types import FeatureVector
from state_encoder.encoder import MarketStateEncoder
from state_encoder.types import DIMENSION_NAMES, MarketState

# Keys the default StateEncoderConfig requires from the feature vector.
REQUIRED_KEYS = [
    "momentum_20", "roc_20", "macd_line",
    "rsi", "roc_5", "velocity",
    "acceleration",
    "atr", "std_20",
    "entropy",
    "hurst_exponent", "fractal_dimension",
    "std_10", "std_50",
]


def make_feature_vector(symbol="STPRNG100", epoch=1000, **overrides) -> FeatureVector:
    defaults = {
        "momentum_20": 0.5,
        "roc_20": 1.2,
        "macd_line": 0.1,
        "rsi": 55.0,
        "roc_5": 0.3,
        "velocity": 0.05,
        "acceleration": 0.01,
        "atr": 2.0,
        "std_20": 1.5,
        "entropy": 2.5,
        "hurst_exponent": 0.5,
        "fractal_dimension": 1.5,
        "std_10": 1.0,
        "std_50": 1.0,
    }
    defaults.update(overrides)
    return FeatureVector(symbol=symbol, epoch=epoch, values=defaults)


def test_encode_returns_all_dimensions():
    encoder = MarketStateEncoder(StateEncoderConfig())
    state = encoder.encode(make_feature_vector())
    assert isinstance(state, MarketState)
    for name in DIMENSION_NAMES:
        assert hasattr(state, name)


def test_all_dimensions_bounded_in_minus_one_one_except_where_documented():
    encoder = MarketStateEncoder(StateEncoderConfig())
    # Feed a variety of vectors to exercise the normalizer beyond the trivial first-sample case.
    rng = np.random.default_rng(0)
    state = None
    for i in range(50):
        fv = make_feature_vector(
            epoch=1000 + i,
            momentum_20=float(rng.normal(0, 5)),
            roc_20=float(rng.normal(0, 5)),
            macd_line=float(rng.normal(0, 1)),
            rsi=float(np.clip(rng.normal(50, 20), 0, 100)),
            roc_5=float(rng.normal(0, 2)),
            velocity=float(rng.normal(0, 0.5)),
            acceleration=float(rng.normal(0, 0.1)),
            atr=float(abs(rng.normal(2, 1)) + 0.1),
            std_20=float(abs(rng.normal(1.5, 0.5)) + 0.1),
            entropy=float(abs(rng.normal(2.5, 1))),
            hurst_exponent=float(np.clip(rng.normal(0.5, 0.15), 0.01, 0.99)),
            fractal_dimension=float(np.clip(rng.normal(1.5, 0.2), 1.01, 1.99)),
            std_10=float(abs(rng.normal(1.0, 0.5)) + 0.1),
            std_50=float(abs(rng.normal(1.0, 0.3)) + 0.1),
        )
        state = encoder.encode(fv)

    for name in DIMENSION_NAMES:
        value = getattr(state, name)
        assert -1.0 <= value <= 1.0, f"{name}={value} out of bounds"


def test_liquidity_is_explicit_placeholder_zero():
    encoder = MarketStateEncoder(StateEncoderConfig())
    state = encoder.encode(make_feature_vector())
    assert state.liquidity == 0.0


def test_persistence_zero_at_hurst_half():
    encoder = MarketStateEncoder(StateEncoderConfig())
    state = encoder.encode(make_feature_vector(hurst_exponent=0.5))
    assert state.persistence == pytest.approx(0.0, abs=1e-9)


def test_persistence_positive_above_half_negative_below():
    encoder = MarketStateEncoder(StateEncoderConfig())
    trending = encoder.encode(make_feature_vector(epoch=1, hurst_exponent=0.8))
    mean_reverting = encoder.encode(make_feature_vector(epoch=2, hurst_exponent=0.2))
    assert trending.persistence > 0
    assert mean_reverting.persistence < 0


def test_complexity_affine_mapped_from_fractal_dimension():
    encoder = MarketStateEncoder(StateEncoderConfig())
    # fractal_dimension=1.5 is the center -> complexity should be 0
    state = encoder.encode(make_feature_vector(fractal_dimension=1.5))
    assert state.complexity == pytest.approx(0.0, abs=1e-9)


def test_compression_expansion_sign():
    encoder = MarketStateEncoder(StateEncoderConfig())
    # std_10 >> std_50 -> expansion (positive)
    expanding = encoder.encode(make_feature_vector(epoch=1, std_10=5.0, std_50=1.0))
    # std_10 << std_50 -> compression (negative)
    compressing = encoder.encode(make_feature_vector(epoch=2, std_10=0.2, std_50=1.0))
    assert expanding.compression_expansion > 0
    assert compressing.compression_expansion < 0


def test_missing_required_feature_key_raises_clear_error():
    encoder = MarketStateEncoder(StateEncoderConfig())
    incomplete = FeatureVector(symbol="STPRNG100", epoch=1, values={"rsi": 50.0})
    with pytest.raises(KeyError):
        encoder.encode(incomplete)


def test_nan_in_source_feature_propagates_as_nan_not_crash():
    encoder = MarketStateEncoder(StateEncoderConfig())
    fv = make_feature_vector(momentum_20=float("nan"))
    state = encoder.encode(fv)
    assert math.isnan(state.trend)  # trend depends on momentum_20
    assert not state.is_valid


def test_encode_is_deterministic_given_same_normalizer_state():
    """Two encoders fed the identical sequence of vectors must produce identical output —
    this is what guarantees live and backtest runs agree given the same data order."""
    config = StateEncoderConfig()
    encoder_a = MarketStateEncoder(config)
    encoder_b = MarketStateEncoder(config)

    rng = np.random.default_rng(42)
    vectors = [
        make_feature_vector(epoch=i, momentum_20=float(rng.normal(0, 3)))
        for i in range(20)
    ]

    for fv in vectors:
        state_a = encoder_a.encode(fv)
        state_b = encoder_b.encode(fv)
        assert state_a == state_b


def test_update_normalizer_false_does_not_mutate_running_stats():
    encoder = MarketStateEncoder(StateEncoderConfig())
    fv1 = make_feature_vector(epoch=1, momentum_20=10.0)
    encoder.encode(fv1)  # updates stats
    count_before = encoder.normalizer.sample_count("momentum_20")

    fv2 = make_feature_vector(epoch=2, momentum_20=999.0)
    encoder.encode(fv2, update_normalizer=False)
    count_after = encoder.normalizer.sample_count("momentum_20")

    assert count_before == count_after  # read-only encode should not have updated stats


def test_market_phase_combines_trend_and_compression():
    encoder = MarketStateEncoder(StateEncoderConfig())
    state = encoder.encode(make_feature_vector())
    assert -1.0 <= state.market_phase <= 1.0


def test_as_vector_matches_dimension_order():
    encoder = MarketStateEncoder(StateEncoderConfig())
    state = encoder.encode(make_feature_vector())
    vec = state.as_vector()
    assert len(vec) == len(DIMENSION_NAMES)
    for i, name in enumerate(DIMENSION_NAMES):
        assert vec[i] == pytest.approx(getattr(state, name)) or (
            np.isnan(vec[i]) and np.isnan(getattr(state, name))
        )
