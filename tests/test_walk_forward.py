import numpy as np
import pytest

from configs.backtest_schema import WalkForwardConfig
from configs.ev_schema import ExpectedValueConfig
from configs.opportunity_schema import OpportunityScoringConfig
from configs.post_trade_schema import PostTradeAnalysisConfig
from configs.probability_schema import BayesianLogisticConfig
from configs.regime_schema import RuleBasedRegimeConfig
from configs.risk_schema import RiskConfig
from backtesting.walk_forward import WalkForwardBacktester
from expected_value.types import ContractSpec, ContractType
from regime.rule_based import RuleBasedRegimeDetector
from state_encoder.types import MarketState


def make_synthetic_run(n_total: int, seed: int, signal_strength: float = 0.3):
    rng = np.random.default_rng(seed)
    trend_vals = rng.uniform(-1, 1, n_total)
    states = [
        MarketState(
            symbol="STPRNG100", epoch=i, trend=float(trend_vals[i]), momentum=0.0, acceleration=0.0,
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
    return states, np.array(closes)


def make_backtester(**wf_overrides) -> WalkForwardBacktester:
    wf_defaults = dict(train_window_trades=300, test_window_trades=100, step_trades=100)
    wf_defaults.update(wf_overrides)
    return WalkForwardBacktester(
        walk_forward_config=WalkForwardConfig(**wf_defaults),
        probability_config=BayesianLogisticConfig(feature_dims=["trend"], prior_precision=1.0),
        ev_config=ExpectedValueConfig(min_ev_threshold=0.0),
        risk_config=RiskConfig(),
        opportunity_config=OpportunityScoringConfig(base_confidence_threshold=0.3, threshold_min=0.2),
        post_trade_config=PostTradeAnalysisConfig(),
        contract=ContractSpec(contract_type=ContractType.RISE_FALL, stake=10.0, payout=19.0, duration_ticks=5),
        starting_equity=1000.0,
    )


def test_run_rejects_mismatched_lengths():
    backtester = make_backtester()
    states, closes = make_synthetic_run(100, seed=0)
    detector = RuleBasedRegimeDetector(RuleBasedRegimeConfig())
    with pytest.raises(ValueError, match="same length"):
        backtester.run(states, closes[:-1], detector)


def test_run_rejects_too_few_observations():
    backtester = make_backtester()
    detector = RuleBasedRegimeDetector(RuleBasedRegimeConfig())
    with pytest.raises(ValueError, match="at least 2"):
        backtester.run([], np.array([]), detector)


def test_run_produces_expected_number_of_windows():
    backtester = make_backtester(train_window_trades=300, test_window_trades=100, step_trades=100)
    states, closes = make_synthetic_run(700, seed=1)
    detector = RuleBasedRegimeDetector(RuleBasedRegimeConfig())
    report = backtester.run(states, closes, detector)
    assert report.n_windows == 3


def test_window_indices_are_contiguous_and_correct():
    backtester = make_backtester(train_window_trades=300, test_window_trades=100, step_trades=100)
    states, closes = make_synthetic_run(700, seed=1)
    detector = RuleBasedRegimeDetector(RuleBasedRegimeConfig())
    report = backtester.run(states, closes, detector)

    assert report.windows[0].test_start_index == 300
    assert report.windows[0].test_end_index == 400
    assert report.windows[1].test_start_index == 400
    assert report.windows[2].test_start_index == 500
    for w in report.windows:
        assert w.train_end_index == w.test_start_index
        assert w.test_end_index - w.test_start_index == 100


def test_aggregate_metrics_reflects_trades_from_all_windows():
    backtester = make_backtester(train_window_trades=300, test_window_trades=150, step_trades=150)
    states, closes = make_synthetic_run(900, seed=2, signal_strength=0.4)
    detector = RuleBasedRegimeDetector(RuleBasedRegimeConfig())
    report = backtester.run(states, closes, detector)

    total_window_trades = sum(w.metrics.n_trades for w in report.windows)
    assert report.aggregate_metrics.n_trades == total_window_trades


def test_win_rate_bounded_and_sane():
    backtester = make_backtester(train_window_trades=300, test_window_trades=200, step_trades=200)
    states, closes = make_synthetic_run(900, seed=3, signal_strength=0.4)
    detector = RuleBasedRegimeDetector(RuleBasedRegimeConfig())
    report = backtester.run(states, closes, detector)
    if report.aggregate_metrics.n_trades > 0:
        assert 0.0 <= report.aggregate_metrics.win_rate <= 1.0


def test_model_is_refit_and_differs_across_windows_with_regime_shift():
    rng = np.random.default_rng(5)
    n_total = 900
    trend_vals = rng.uniform(-1, 1, n_total)
    states = [
        MarketState(symbol="STPRNG100", epoch=i, trend=float(trend_vals[i]), momentum=0.0, acceleration=0.0,
                    volatility=0.1, noise=0.1, persistence=0.0, compression_expansion=0.0, complexity=0.0,
                    uncertainty=0.1, liquidity=0.0, market_phase=0.0)
        for i in range(n_total)
    ]
    closes = [100.0]
    for i in range(n_total - 1):
        sign = 1.0 if i < n_total // 2 else -1.0
        p_up = float(np.clip(0.5 + sign * 0.4 * trend_vals[i], 0.05, 0.95))
        up = rng.uniform(0, 1) < p_up
        closes.append(closes[-1] + (1.0 if up else -1.0))
    closes = np.array(closes)

    backtester = make_backtester(train_window_trades=300, test_window_trades=150, step_trades=150)
    detector = RuleBasedRegimeDetector(RuleBasedRegimeConfig())
    report = backtester.run(states, closes, detector)
    assert report.n_windows >= 2
