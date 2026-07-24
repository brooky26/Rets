import math

import pytest

from configs.duration_selection_schema import DurationSelectionConfig
from configs.ev_schema import ExpectedValueConfig
from configs.monte_carlo_schema import MonteCarloPricePathConfig
from monte_carlo.duration_selector import DurationSelector, _hurst_from_persistence
from monte_carlo.price_paths import MonteCarloPricePathSimulator
from probability.types import ProbabilityEstimate

NAN = float("nan")


def make_probability(direction: int = 1, confidence: float = 0.75, valid: bool = True) -> ProbabilityEstimate:
    if not valid:
        return ProbabilityEstimate(
            symbol="stpRNG", epoch=100, model_name="fused",
            prob_up=NAN, prob_down=NAN, uncertainty=NAN, expected_direction=0, confidence=NAN,
        )
    prob_up = confidence if direction == 1 else 1.0 - confidence
    return ProbabilityEstimate(
        symbol="stpRNG", epoch=100, model_name="fused",
        prob_up=prob_up, prob_down=1.0 - prob_up, uncertainty=0.3,
        expected_direction=direction, confidence=confidence,
    )


def make_selector(mc_max_standard_error=0.05, mc_min_paths=200, n_paths=2000, **overrides) -> DurationSelector:
    config = DurationSelectionConfig(
        mc_max_standard_error=mc_max_standard_error, mc_min_paths=mc_min_paths, **overrides
    )
    ev_config = ExpectedValueConfig(min_ev_threshold=0.0)
    mc_config = MonteCarloPricePathConfig(n_paths=n_paths, random_seed=42)
    simulator = MonteCarloPricePathSimulator(mc_config)
    return DurationSelector(config, ev_config, simulator)


def test_hurst_from_persistence_inverts_the_encoder_affine_mapping():
    # persistence = (H - 0.5) / 0.5  =>  H = 0.5 + 0.5 * persistence
    assert _hurst_from_persistence(0.0) == pytest.approx(0.5)
    assert _hurst_from_persistence(1.0) == pytest.approx(1.0)
    assert _hurst_from_persistence(-1.0) == pytest.approx(0.0)


def test_no_direction_returns_no_selection_and_no_candidates():
    selector = make_selector()
    probability = make_probability(direction=1, confidence=0.6)
    probability = ProbabilityEstimate(
        symbol="stpRNG", epoch=100, model_name="fused",
        prob_up=0.5, prob_down=0.5, uncertainty=0.3, expected_direction=0, confidence=0.5,
    )
    result = selector.select(
        "stpRNG", 100, current_price=100.0, fused_probability=probability,
        mu_per_tick=0.0001, sigma_per_tick=0.01, hurst_persistence=0.2,
        stake=10.0, assumed_payout_ratio=1.9,
    )
    assert result.chosen is None
    assert result.all_candidates == ()


def test_invalid_fused_probability_returns_no_selection():
    selector = make_selector()
    probability = make_probability(valid=False)
    result = selector.select(
        "stpRNG", 100, current_price=100.0, fused_probability=probability,
        mu_per_tick=0.0001, sigma_per_tick=0.01, hurst_persistence=0.2,
        stake=10.0, assumed_payout_ratio=1.9,
    )
    assert result.chosen is None
    assert result.all_candidates == ()


def test_every_candidate_duration_is_evaluated_regardless_of_outcome():
    selector = make_selector(candidate_durations_ticks=[3, 5, 8, 12, 20])
    probability = make_probability(direction=1, confidence=0.8)
    result = selector.select(
        "stpRNG", 100, current_price=100.0, fused_probability=probability,
        mu_per_tick=0.0005, sigma_per_tick=0.01, hurst_persistence=0.4,
        stake=10.0, assumed_payout_ratio=1.9,
    )
    assert len(result.all_candidates) == 5
    assert {c.duration_ticks for c in result.all_candidates} == {3, 5, 8, 12, 20}


def test_uses_monte_carlo_when_confident():
    # Large n_paths + a strong, unambiguous drift => MC should easily clear
    # the standard-error bar and be selected as the method for at least
    # one candidate.
    selector = make_selector(n_paths=5000, mc_max_standard_error=0.05, mc_min_paths=200)
    probability = make_probability(direction=1, confidence=0.9)
    result = selector.select(
        "stpRNG", 100, current_price=100.0, fused_probability=probability,
        mu_per_tick=0.002, sigma_per_tick=0.01, hurst_persistence=0.3,
        stake=10.0, assumed_payout_ratio=1.9,
    )
    methods = {c.method for c in result.all_candidates}
    assert "monte_carlo" in methods


def test_falls_back_to_hurst_when_mu_sigma_are_nan():
    selector = make_selector()
    probability = make_probability(direction=1, confidence=0.75)
    result = selector.select(
        "stpRNG", 100, current_price=100.0, fused_probability=probability,
        mu_per_tick=NAN, sigma_per_tick=NAN, hurst_persistence=0.2,
        stake=10.0, assumed_payout_ratio=1.9,
    )
    assert all(c.method == "hurst_fallback" for c in result.all_candidates)


def test_falls_back_to_hurst_when_mc_standard_error_too_large():
    # Very few paths + a near-coinflip drift => wide standard error, should
    # reject MC and use the fallback for every candidate.
    selector = make_selector(n_paths=100, mc_min_paths=5000)  # min_paths set unreachably high
    probability = make_probability(direction=1, confidence=0.55)
    result = selector.select(
        "stpRNG", 100, current_price=100.0, fused_probability=probability,
        mu_per_tick=0.00001, sigma_per_tick=0.02, hurst_persistence=0.1,
        stake=10.0, assumed_payout_ratio=1.9,
    )
    assert all(c.method == "hurst_fallback" for c in result.all_candidates)


def test_hurst_fallback_persistent_regime_decays_slower_than_mean_reverting():
    """H > 0.5 (persistent/trending) should preserve more of the edge at a
    LONGER duration than H < 0.5 (mean-reverting), relative to the same
    starting confidence at the reference duration."""
    selector = make_selector(hurst_reference_duration_ticks=5)
    probability = make_probability(direction=1, confidence=0.8)

    persistent_result = selector.select(
        "stpRNG", 100, current_price=100.0, fused_probability=probability,
        mu_per_tick=NAN, sigma_per_tick=NAN, hurst_persistence=0.8,  # H = 0.9, strongly persistent
        stake=10.0, assumed_payout_ratio=1.9,
    )
    mean_reverting_result = selector.select(
        "stpRNG", 100, current_price=100.0, fused_probability=probability,
        mu_per_tick=NAN, sigma_per_tick=NAN, hurst_persistence=-0.8,  # H = 0.1, strongly anti-persistent
        stake=10.0, assumed_payout_ratio=1.9,
    )

    def confidence_at(result, duration):
        match = [c for c in result.all_candidates if c.duration_ticks == duration][0]
        return match.probability_estimate.confidence

    long_duration = max(selector._config.candidate_durations_ticks)
    persistent_confidence = confidence_at(persistent_result, long_duration)
    mean_reverting_confidence = confidence_at(mean_reverting_result, long_duration)

    assert persistent_confidence > mean_reverting_confidence


def test_no_viable_candidate_when_every_ev_is_non_positive():
    # A confidence right at 0.5 (no edge) makes EV <= 0 for every candidate.
    selector = make_selector()
    probability = make_probability(direction=1, confidence=0.50001)
    result = selector.select(
        "stpRNG", 100, current_price=100.0, fused_probability=probability,
        mu_per_tick=NAN, sigma_per_tick=NAN, hurst_persistence=0.0,
        stake=10.0, assumed_payout_ratio=1.5,  # low payout ratio makes breakeven confidence higher than 0.50001
    )
    assert result.chosen is None
    assert len(result.all_candidates) > 0  # still evaluated and logged


def test_chosen_candidate_has_the_highest_risk_adjusted_score_among_viable():
    selector = make_selector()
    probability = make_probability(direction=1, confidence=0.85)
    result = selector.select(
        "stpRNG", 100, current_price=100.0, fused_probability=probability,
        mu_per_tick=NAN, sigma_per_tick=NAN, hurst_persistence=0.5,
        stake=10.0, assumed_payout_ratio=1.9,
    )
    assert result.chosen is not None
    viable = [c for c in result.all_candidates if c.ev_estimate.is_valid and c.ev_estimate.is_positive_ev]
    best = max(viable, key=lambda c: c.ev_estimate.risk_adjusted_score)
    assert result.chosen.duration_ticks == best.duration_ticks


def test_result_contract_property_matches_chosen_candidate():
    selector = make_selector()
    probability = make_probability(direction=1, confidence=0.85)
    result = selector.select(
        "stpRNG", 100, current_price=100.0, fused_probability=probability,
        mu_per_tick=NAN, sigma_per_tick=NAN, hurst_persistence=0.5,
        stake=10.0, assumed_payout_ratio=1.9,
    )
    assert result.chosen is not None
    assert result.contract is not None
    assert result.contract.duration_ticks == result.chosen.duration_ticks
    assert result.contract.stake == 10.0


def test_no_mc_simulator_configured_always_uses_hurst_fallback():
    config = DurationSelectionConfig()
    ev_config = ExpectedValueConfig()
    selector = DurationSelector(config, ev_config, mc_simulator=None)
    probability = make_probability(direction=1, confidence=0.8)
    result = selector.select(
        "stpRNG", 100, current_price=100.0, fused_probability=probability,
        mu_per_tick=0.001, sigma_per_tick=0.01, hurst_persistence=0.3,
        stake=10.0, assumed_payout_ratio=1.9,
    )
    assert all(c.method == "hurst_fallback" for c in result.all_candidates)
