import asyncio

import numpy as np
import pytest

from configs.ensemble_schema import EnsembleFusionConfig
from configs.ev_schema import ExpectedValueConfig
from configs.execution_schema import ExecutionConfig
from configs.meta_learning_schema import MetaLearningConfig
from configs.monte_carlo_schema import MonteCarloPricePathConfig
from configs.opportunity_schema import OpportunityScoringConfig
from configs.paper_trading_schema import PaperTradingConfig
from configs.post_trade_schema import PostTradeAnalysisConfig
from configs.probability_schema import BaggedGBMConfig, BayesianLogisticConfig
from configs.regime_schema import RuleBasedRegimeConfig
from configs.risk_schema import RiskConfig
from configs.state_encoder_schema import StateEncoderConfig
from data.types import Candle
from features.types import FeatureVector
from paper_trading.orchestrator import PaperTradingOrchestrator
from regime.rule_based import RuleBasedRegimeDetector
from state_encoder.encoder import MarketStateEncoder
from state_encoder.types import MarketState


def make_synthetic_states_and_closes(n_total: int, seed: int, signal_strength: float = 0.3, symbol: str = "stpRNG"):
    rng = np.random.default_rng(seed)
    trend_vals = rng.uniform(-1, 1, n_total)
    states = [
        MarketState(
            symbol=symbol, epoch=i, trend=float(trend_vals[i]), momentum=0.0, acceleration=0.0,
            volatility=0.1, noise=0.1, persistence=0.0, compression_expansion=0.0, complexity=0.0,
            uncertainty=0.1, liquidity=0.0, market_phase=0.0,
        )
        for i in range(n_total)
    ]
    closes = [100.0]
    for i in range(n_total - 1):
        p_up = float(np.clip(0.5 + signal_strength * trend_vals[i], 0.05, 0.95))
        up = rng.uniform(0, 1) < p_up
        closes.append(closes[-1] + (1.0 if up else -1.0))
    return states, closes


def make_candle(symbol: str, epoch: int, close: float) -> Candle:
    return Candle(symbol=symbol, epoch=epoch, granularity=60, open=close, high=close, low=close, close=close)


def make_vector(symbol: str, epoch: int) -> FeatureVector:
    return FeatureVector(symbol=symbol, epoch=epoch, values={})


def make_fusion_orchestrator(**paper_overrides) -> PaperTradingOrchestrator:
    paper_defaults = dict(min_bootstrap_candles=300, starting_equity=1000.0)
    paper_defaults.update(paper_overrides)
    return PaperTradingOrchestrator(
        paper_config=PaperTradingConfig(**paper_defaults),
        probability_config=BayesianLogisticConfig(feature_dims=["trend"], prior_precision=1.0),
        ev_config=ExpectedValueConfig(min_ev_threshold=0.0),
        risk_config=RiskConfig(),
        opportunity_config=OpportunityScoringConfig(base_confidence_threshold=0.3, threshold_min=0.2),
        post_trade_config=PostTradeAnalysisConfig(),
        execution_config=ExecutionConfig(mode="paper"),
        platform_environment="development",
        regime_detector=RuleBasedRegimeDetector(RuleBasedRegimeConfig()),
        state_encoder=MarketStateEncoder(StateEncoderConfig()),
        fusion_config=EnsembleFusionConfig(),
        meta_learning_config=MetaLearningConfig(model_names=["bayesian_logistic", "bagged_gbm", "monte_carlo_gbm"]),
        bagged_gbm_config=BaggedGBMConfig(feature_dims=["trend"], n_estimators=5),
        monte_carlo_config=MonteCarloPricePathConfig(n_paths=200, horizon_ticks=5, mu_estimation_window=20),
    )


def test_default_orchestrator_has_no_fusion_engine():
    from tests.test_paper_trading_orchestrator import make_orchestrator
    orch = make_orchestrator()
    assert orch._fusion_engine is None
    assert orch._weight_learner is None


def test_fusion_orchestrator_bootstraps_all_configured_models():
    orch = make_fusion_orchestrator()
    states, closes = make_synthetic_states_and_closes(320, seed=1)
    ok = orch.bootstrap("stpRNG", states, closes)
    assert ok
    assert "stpRNG" in orch._probability_models
    assert "stpRNG" in orch._bagged_gbm_models
    assert "stpRNG" in orch._recent_closes
    assert orch._fusion_engine is not None
    assert orch._weight_learner is not None


def test_fusion_orchestrator_runs_candles_without_error():
    orch = make_fusion_orchestrator()
    states, closes = make_synthetic_states_and_closes(320, seed=2)
    orch.bootstrap("stpRNG", states, closes)

    async def run():
        results = []
        for i in range(320, 360):
            candle = make_candle("stpRNG", i, closes[-1] + float(i))
            vector = make_vector("stpRNG", i)
            orch._state_encoder.encode = lambda v, i=i: MarketState(
                symbol="stpRNG", epoch=i, trend=0.5, momentum=0.0, acceleration=0.0, volatility=0.1,
                noise=0.1, persistence=0.0, compression_expansion=0.0, complexity=0.0, uncertainty=0.1,
                liquidity=0.0, market_phase=0.0,
            )
            result = await orch.on_candle("stpRNG", candle, vector)
            results.append(result)
        return results

    results = asyncio.run(run())
    assert len(results) == 40
    assert any(r["decision"] is not None for r in results)


# --------------------------------------------------------------------- #
# Data-sufficiency shrinkage — end-to-end through bootstrap() into the
# stored self._sufficiency dict actually consulted by _predict_probability.
# --------------------------------------------------------------------- #

def test_bootstrap_computes_full_sufficiency_when_data_exceeds_target():
    orch = make_fusion_orchestrator(min_bootstrap_candles=300, bagged_gbm_target_samples=100)
    states, closes = make_synthetic_states_and_closes(320, seed=2)  # 319 usable >> 100 target
    orch.bootstrap("stpRNG", states, closes)

    assert orch._sufficiency["stpRNG"]["bayesian_logistic"] == 1.0
    assert orch._sufficiency["stpRNG"]["bagged_gbm"] == 1.0


def test_bootstrap_computes_partial_sufficiency_when_data_below_target():
    orch = make_fusion_orchestrator(min_bootstrap_candles=100, bagged_gbm_target_samples=1000)
    states, closes = make_synthetic_states_and_closes(150, seed=3)  # 149 usable << 1000 target
    orch.bootstrap("stpRNG", states, closes)

    assert orch._sufficiency["stpRNG"]["bayesian_logistic"] == 1.0  # its own floor IS its target
    assert 0.0 < orch._sufficiency["stpRNG"]["bagged_gbm"] < 1.0


def test_predict_probability_with_partial_sufficiency_does_not_crash_and_produces_valid_estimate():
    orch = make_fusion_orchestrator(min_bootstrap_candles=100, bagged_gbm_target_samples=1000)
    states, closes = make_synthetic_states_and_closes(150, seed=4)
    orch.bootstrap("stpRNG", states, closes)

    strong_state = MarketState(
        symbol="stpRNG", epoch=9999, trend=0.8, momentum=0.0, acceleration=0.0,
        volatility=0.1, noise=0.1, persistence=0.0, compression_expansion=0.0,
        complexity=0.0, uncertainty=0.1, liquidity=0.0, market_phase=0.0,
    )
    regime = orch._regime_detector.classify(strong_state)
    probability = orch._predict_probability("stpRNG", strong_state, regime, mc_result=None)

    assert probability.is_valid
    assert 0.5 <= probability.confidence <= 1.0
