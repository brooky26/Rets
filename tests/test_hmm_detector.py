import numpy as np
import pytest

from configs.regime_schema import GaussianHMMConfig
from regime.hmm_detector import GaussianHMMRegimeDetector, HMMNotFittedError
from regime.types import RegimeLabel
from state_encoder.types import MarketState


def make_state(symbol="STPRNG100", epoch=0, **overrides) -> MarketState:
    defaults = dict(
        trend=0.0, momentum=0.0, acceleration=0.0, volatility=0.0, noise=0.0,
        persistence=0.0, compression_expansion=0.0, complexity=0.0,
        uncertainty=0.0, liquidity=0.0, market_phase=0.0,
    )
    defaults.update(overrides)
    return MarketState(symbol=symbol, epoch=epoch, **defaults)


def make_training_states(n_per_regime=100, seed=0) -> list[MarketState]:
    """Two synthetic regimes: one clearly trending, one clearly random-walk-ish."""
    rng = np.random.default_rng(seed)
    states = []
    epoch = 0
    for _ in range(n_per_regime):
        states.append(
            make_state(
                epoch=epoch,
                trend=float(np.clip(rng.normal(0.8, 0.1), -1, 1)),
                volatility=float(np.clip(rng.normal(0.3, 0.1), -1, 1)),
                persistence=float(np.clip(rng.normal(0.6, 0.1), -1, 1)),
                compression_expansion=float(np.clip(rng.normal(0.1, 0.1), -1, 1)),
            )
        )
        epoch += 1
    for _ in range(n_per_regime):
        states.append(
            make_state(
                epoch=epoch,
                trend=float(np.clip(rng.normal(0.0, 0.05), -1, 1)),
                volatility=float(np.clip(rng.normal(0.0, 0.1), -1, 1)),
                persistence=float(np.clip(rng.normal(0.0, 0.05), -1, 1)),
                compression_expansion=float(np.clip(rng.normal(0.0, 0.1), -1, 1)),
            )
        )
        epoch += 1
    return states


@pytest.fixture
def config() -> GaussianHMMConfig:
    return GaussianHMMConfig(
        n_states=2,
        observation_dims=["trend", "volatility", "persistence", "compression_expansion"],
        em_max_iterations=50,
        random_seed=1,
    )


def test_classify_before_fit_raises(config):
    detector = GaussianHMMRegimeDetector(config)
    with pytest.raises(HMMNotFittedError):
        detector.classify(make_state())


def test_fit_raises_on_insufficient_data(config):
    detector = GaussianHMMRegimeDetector(config)
    with pytest.raises(ValueError, match="at least"):
        detector.fit([make_state(epoch=i) for i in range(3)])


def test_fit_then_classify_succeeds(config):
    detector = GaussianHMMRegimeDetector(config)
    detector.fit(make_training_states())
    assert detector.is_fitted
    result = detector.classify(make_state(epoch=9999, trend=0.8, persistence=0.6))
    assert result.regime in list(RegimeLabel)
    assert 0.0 <= result.confidence <= 1.0


def test_fit_labels_trending_state_correctly(config):
    """
    The synthetic training data has one obviously-trending cluster
    (trend~0.8, persistence~0.6) — the fitted HMM should learn a hidden
    state whose label reflects that, and classifying a fresh trending
    observation should land on a trend-family label.
    """
    detector = GaussianHMMRegimeDetector(config)
    detector.fit(make_training_states())

    result = detector.classify(make_state(epoch=9999, trend=0.8, volatility=0.3, persistence=0.6, compression_expansion=0.1))
    assert result.regime in (RegimeLabel.STRONG_TREND, RegimeLabel.WEAK_TREND, RegimeLabel.BREAKOUT)


def test_classify_maintains_separate_state_per_symbol(config):
    """
    Interleaving classify() calls across two symbols must give each
    symbol exactly the same result as if its sequence had been replayed
    alone — i.e. one symbol's observations must never leak into another
    symbol's running forward-filter state.
    """
    detector_interleaved = GaussianHMMRegimeDetector(config)
    detector_interleaved.fit(make_training_states())

    seq_a = [make_state(symbol="A", epoch=i, trend=t, persistence=0.6 * t)
             for i, t in enumerate([0.8, 0.7, 0.75])]
    seq_b = [make_state(symbol="B", epoch=i, trend=t, persistence=0.0)
             for i, t in enumerate([0.0, 0.05, -0.02])]

    interleaved_results_a = []
    interleaved_results_b = []
    for a, b in zip(seq_a, seq_b):
        interleaved_results_a.append(detector_interleaved.classify(a))
        interleaved_results_b.append(detector_interleaved.classify(b))

    # Now replay each symbol alone on fresh detectors sharing the same fit.
    detector_a_only = GaussianHMMRegimeDetector(config)
    detector_a_only.fit(make_training_states())
    solo_results_a = [detector_a_only.classify(s) for s in seq_a]

    detector_b_only = GaussianHMMRegimeDetector(config)
    detector_b_only.fit(make_training_states())
    solo_results_b = [detector_b_only.classify(s) for s in seq_b]

    for interleaved, solo in zip(interleaved_results_a, solo_results_a):
        assert interleaved.regime == solo.regime
        assert interleaved.confidence == pytest.approx(solo.confidence, abs=1e-9)
    for interleaved, solo in zip(interleaved_results_b, solo_results_b):
        assert interleaved.regime == solo.regime
        assert interleaved.confidence == pytest.approx(solo.confidence, abs=1e-9)


def test_probabilities_sum_to_one_across_labels(config):
    detector = GaussianHMMRegimeDetector(config)
    detector.fit(make_training_states())
    result = detector.classify(make_state(epoch=1, trend=0.5))
    total = sum(result.probabilities.values())
    assert total == pytest.approx(1.0, abs=1e-6)


def test_invalid_state_returns_nan_confidence(config):
    detector = GaussianHMMRegimeDetector(config)
    detector.fit(make_training_states())
    result = detector.classify(make_state(epoch=1, trend=float("nan")))
    assert result.confidence != result.confidence  # NaN


def test_reset_symbol_clears_running_state(config):
    detector = GaussianHMMRegimeDetector(config)
    detector.fit(make_training_states())
    detector.classify(make_state(symbol="A", epoch=1, trend=0.8))
    assert "A" in detector._alpha_by_symbol
    detector.reset_symbol("A")
    assert "A" not in detector._alpha_by_symbol


def test_detector_name_set(config):
    detector = GaussianHMMRegimeDetector(config)
    assert detector.name == "gaussian_hmm"


def test_streaming_classify_matches_batch_forward_filter(config):
    """
    Classifying a sequence of states one at a time via classify() must
    produce the same filtered distribution as a batch forward pass over
    the equivalent observation matrix — otherwise live and backtest
    regime assignments could silently diverge.
    """
    detector = GaussianHMMRegimeDetector(config)
    training_states = make_training_states()
    detector.fit(training_states)

    test_states = make_training_states(n_per_regime=20, seed=99)
    symbol_states = [make_state(symbol="X", epoch=i, **{
        k: getattr(s, k) for k in ["trend", "momentum", "acceleration", "volatility",
                                    "noise", "persistence", "compression_expansion",
                                    "complexity", "uncertainty", "liquidity", "market_phase"]
    }) for i, s in enumerate(test_states)]

    streamed_probs = []
    for s in symbol_states:
        result = detector.classify(s)
        streamed_probs.append(result.probabilities)

    # Recompute via batch forward_filter directly on the underlying HMM.
    X = detector._to_observation_matrix(symbol_states)
    batch_alpha = detector._hmm.forward_filter(X)

    for t, probs in enumerate(streamed_probs):
        label_probs_from_batch = {label: 0.0 for label in RegimeLabel}
        for k, p in enumerate(batch_alpha[t]):
            label_probs_from_batch[detector._state_labels[k]] += float(p)
        for label in RegimeLabel:
            assert probs[label] == pytest.approx(label_probs_from_batch[label], abs=1e-6)
