from unittest.mock import patch

import numpy as np
import pytest

from champion_challenger.types import PromotionDecision
from configs.champion_challenger_schema import ChampionChallengerConfig
from configs.ev_schema import ExpectedValueConfig
from configs.opportunity_schema import OpportunityScoringConfig
from configs.probability_schema import BayesianLogisticConfig
from configs.regime_schema import GaussianHMMConfig, RuleBasedRegimeConfig
from configs.risk_schema import RiskConfig
from expected_value.types import ContractSpec, ContractType
from regime.hmm_detector import GaussianHMMRegimeDetector
from regime.promotion import compare_regime_detectors, simulate_trades_with_regime_detector
from regime.rule_based import RuleBasedRegimeDetector
from state_encoder.types import MarketState


def make_states_and_closes(n: int, seed: int, signal_strength: float = 0.5):
    rng = np.random.default_rng(seed)
    trend_vals = rng.uniform(-1, 1, n)
    vol_vals = rng.uniform(0.05, 0.3, n)
    states = [
        MarketState(
            symbol="stpRNG", epoch=i, trend=float(trend_vals[i]), momentum=0.0, acceleration=0.0,
            volatility=float(vol_vals[i]), noise=0.1, persistence=0.2, compression_expansion=0.0,
            complexity=0.0, uncertainty=0.1, liquidity=0.0, market_phase=0.0,
        )
        for i in range(n)
    ]
    closes = [100.0]
    for i in range(n - 1):
        p_up = float(np.clip(0.5 + signal_strength * trend_vals[i], 0.05, 0.95))
        up = rng.uniform(0, 1) < p_up
        closes.append(closes[-1] + (1.0 if up else -1.0))
    return states, closes


def make_common_configs(min_trades_required=5):
    return dict(
        probability_config=BayesianLogisticConfig(feature_dims=["trend"], prior_precision=1.0),
        ev_config=ExpectedValueConfig(min_ev_threshold=0.0),
        risk_config=RiskConfig(),
        opportunity_config=OpportunityScoringConfig(base_confidence_threshold=0.3, threshold_min=0.2),
        contract=ContractSpec(contract_type=ContractType.RISE_FALL, stake=10.0, payout=19.0, duration_ticks=5),
        champion_challenger_config=ChampionChallengerConfig(min_trades_required=min_trades_required),
        hmm_config=GaussianHMMConfig(n_states=3),
        rule_based_config=RuleBasedRegimeConfig(),
        starting_equity=1000.0,
    )


def test_rejects_mismatched_lengths():
    states, closes = make_states_and_closes(300, seed=0)
    with pytest.raises(ValueError, match="same length"):
        compare_regime_detectors(states, closes[:-1], **make_common_configs())


def test_rejects_bad_train_fraction():
    states, closes = make_states_and_closes(300, seed=0)
    with pytest.raises(ValueError, match="train_fraction"):
        compare_regime_detectors(states, closes, **make_common_configs(), train_fraction=1.5)


def test_raises_when_not_enough_data_for_a_split():
    states, closes = make_states_and_closes(3, seed=0)
    with pytest.raises(ValueError, match="Not enough data"):
        compare_regime_detectors(states, closes, **make_common_configs())


def test_runs_end_to_end_and_returns_a_promotion_decision():
    states, closes = make_states_and_closes(400, seed=1, signal_strength=0.7)
    decision, hmm_detector = compare_regime_detectors(states, closes, **make_common_configs())

    assert isinstance(decision, PromotionDecision)
    assert decision.champion_id == "rule_based_regime"
    assert decision.challenger_id == "hmm_regime"
    # HMM fit should succeed on well-behaved synthetic data of this size.
    assert hmm_detector is not None
    assert isinstance(hmm_detector, GaussianHMMRegimeDetector)


def test_hmm_fit_failure_falls_back_gracefully_with_no_promotion():
    states, closes = make_states_and_closes(400, seed=2)

    with patch.object(GaussianHMMRegimeDetector, "fit", side_effect=RuntimeError("degenerate covariance")):
        decision, hmm_detector = compare_regime_detectors(states, closes, **make_common_configs())

    assert hmm_detector is None
    assert decision.promote is False
    assert decision.challenger_id == "hmm_regime"


def test_simulate_trades_returns_a_list_of_floats_without_raising():
    states, closes = make_states_and_closes(300, seed=3, signal_strength=0.8)
    configs = make_common_configs()
    from probability.bayesian_logistic import BayesianLogisticRegression

    labels = (np.diff(np.array(closes)) > 0).astype(int)
    X = np.array([[getattr(s, "trend")] for s in states[:-1]])
    model = BayesianLogisticRegression(configs["probability_config"]).fit(X, labels)

    detector = RuleBasedRegimeDetector(configs["rule_based_config"])
    pnls = simulate_trades_with_regime_detector(
        states, closes, model, configs["ev_config"], configs["risk_config"],
        configs["opportunity_config"], configs["contract"], detector, configs["starting_equity"],
    )
    assert isinstance(pnls, list)
    assert all(isinstance(p, float) for p in pnls)


def test_champion_is_always_rule_based_challenger_is_always_hmm():
    states, closes = make_states_and_closes(400, seed=4, signal_strength=0.6)
    decision, _ = compare_regime_detectors(states, closes, **make_common_configs())
    assert decision.champion_id == "rule_based_regime"
    assert decision.challenger_id == "hmm_regime"
