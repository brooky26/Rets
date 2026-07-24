import numpy as np
import pytest

from champion_challenger.comparator import ChampionChallengerComparator
from configs.champion_challenger_schema import ChampionChallengerConfig


def make_config(**overrides) -> ChampionChallengerConfig:
    defaults = dict(
        min_trades_required=30, confidence_level=0.95, min_improvement_threshold=0.0,
        n_bootstrap_resamples=1000, random_seed=1,
    )
    defaults.update(overrides)
    return ChampionChallengerConfig(**defaults)


def test_insufficient_sample_size_rejects_without_bootstrapping():
    comparator = ChampionChallengerComparator(make_config(min_trades_required=30))
    decision = comparator.compare(
        "champ-v1", "chal-v2",
        champion_returns=[0.1] * 40,
        challenger_returns=[0.2] * 10,
    )
    assert decision.promote is False
    assert "Insufficient sample size" in decision.reason
    assert decision.bootstrap_lower_bound != decision.bootstrap_lower_bound


def test_clear_genuine_improvement_promotes():
    rng = np.random.default_rng(0)
    champion_returns = list(rng.normal(0.0, 0.05, 200))
    challenger_returns = list(rng.normal(0.15, 0.05, 200))
    comparator = ChampionChallengerComparator(make_config())
    decision = comparator.compare("champ-v1", "chal-v2", champion_returns, challenger_returns)
    assert decision.promote is True
    assert decision.bootstrap_lower_bound > 0


def test_no_real_difference_does_not_promote():
    rng = np.random.default_rng(1)
    champion_returns = list(rng.normal(0.05, 0.1, 200))
    challenger_returns = list(rng.normal(0.05, 0.1, 200))
    comparator = ChampionChallengerComparator(make_config())
    decision = comparator.compare("champ-v1", "chal-v2", champion_returns, challenger_returns)
    assert decision.promote is False


def test_challenger_worse_than_champion_does_not_promote():
    rng = np.random.default_rng(2)
    champion_returns = list(rng.normal(0.10, 0.05, 200))
    challenger_returns = list(rng.normal(0.02, 0.05, 200))
    comparator = ChampionChallengerComparator(make_config())
    decision = comparator.compare("champ-v1", "chal-v2", champion_returns, challenger_returns)
    assert decision.promote is False
    assert decision.mean_improvement < 0


def test_higher_confidence_level_is_stricter():
    rng_a = np.random.default_rng(3)
    champion_returns = list(rng_a.normal(0.0, 0.1, 100))
    rng_b = np.random.default_rng(4)
    challenger_returns = list(rng_b.normal(0.03, 0.1, 100))

    lenient = ChampionChallengerComparator(make_config(confidence_level=0.80, random_seed=10))
    strict = ChampionChallengerComparator(make_config(confidence_level=0.99, random_seed=10))

    lenient_decision = lenient.compare("c", "x", champion_returns, challenger_returns)
    strict_decision = strict.compare("c", "x", champion_returns, challenger_returns)

    assert strict_decision.bootstrap_lower_bound <= lenient_decision.bootstrap_lower_bound


def test_min_improvement_threshold_requires_a_real_margin():
    rng = np.random.default_rng(5)
    champion_returns = list(rng.normal(0.0, 0.02, 300))
    challenger_returns = list(rng.normal(0.01, 0.02, 300))

    high_threshold = ChampionChallengerComparator(make_config(min_improvement_threshold=0.05))
    decision_strict = high_threshold.compare("c", "x", champion_returns, challenger_returns)

    assert decision_strict.promote is False


def test_champion_and_challenger_means_computed_correctly():
    comparator = ChampionChallengerComparator(make_config(min_trades_required=2))
    decision = comparator.compare("c", "x", [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert decision.champion_mean_return == pytest.approx(2.0)
    assert decision.challenger_mean_return == pytest.approx(5.0)
    assert decision.mean_improvement == pytest.approx(3.0)


def test_reason_string_present_for_both_outcomes():
    comparator = ChampionChallengerComparator(make_config(min_trades_required=2))
    promote_decision = comparator.compare("c", "x", [0.0] * 50, [1.0] * 50)
    reject_decision = comparator.compare("c", "x", [1.0] * 50, [0.0] * 50)
    assert len(promote_decision.reason) > 0
    assert len(reject_decision.reason) > 0
