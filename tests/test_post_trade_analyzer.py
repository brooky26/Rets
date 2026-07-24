import numpy as np
import pytest

from configs.post_trade_schema import PostTradeAnalysisConfig
from post_trade.analyzer import PostTradeAnalyzer
from post_trade.types import CompletedTrade
from regime.types import RegimeLabel


def make_trade(pnl, stake=10.0, predicted_probability=0.6, epoch=1000, regime=RegimeLabel.STRONG_TREND, quality_score=0.6) -> CompletedTrade:
    return CompletedTrade(
        symbol="STPRNG100", entry_epoch=epoch, exit_epoch=epoch + 5, direction=1,
        stake=stake, pnl=pnl, predicted_probability=predicted_probability,
        regime_at_entry=regime, quality_score_at_entry=quality_score, exit_reason="expired",
    )


def make_config(**overrides) -> PostTradeAnalysisConfig:
    defaults = dict(n_calibration_bins=10, rolling_window_trades=200)
    defaults.update(overrides)
    return PostTradeAnalysisConfig(**defaults)


def test_empty_analyzer_returns_nan_metrics():
    analyzer = PostTradeAnalyzer(make_config())
    metrics = analyzer.compute_metrics()
    assert metrics.n_trades == 0
    assert metrics.win_rate != metrics.win_rate


def test_win_rate_hand_computed():
    analyzer = PostTradeAnalyzer(make_config())
    for pnl in [9.0, 9.0, -10.0, 9.0, -10.0]:
        analyzer.record_trade(make_trade(pnl))
    metrics = analyzer.compute_metrics()
    assert metrics.win_rate == pytest.approx(0.6)


def test_profit_factor_hand_computed():
    analyzer = PostTradeAnalyzer(make_config())
    for pnl in [10.0, 20.0, -5.0, -5.0]:
        analyzer.record_trade(make_trade(pnl))
    metrics = analyzer.compute_metrics()
    assert metrics.profit_factor == pytest.approx(3.0)


def test_profit_factor_infinite_with_no_losses():
    analyzer = PostTradeAnalyzer(make_config())
    for pnl in [10.0, 20.0]:
        analyzer.record_trade(make_trade(pnl))
    metrics = analyzer.compute_metrics()
    assert metrics.profit_factor == float("inf")


def test_expectancy_hand_computed():
    analyzer = PostTradeAnalyzer(make_config())
    for pnl in [10.0, -10.0, 5.0]:
        analyzer.record_trade(make_trade(pnl))
    metrics = analyzer.compute_metrics()
    assert metrics.expectancy == pytest.approx(5.0 / 3)


def test_average_return_pct_hand_computed():
    analyzer = PostTradeAnalyzer(make_config())
    analyzer.record_trade(make_trade(pnl=5.0, stake=10.0))
    analyzer.record_trade(make_trade(pnl=-2.0, stake=10.0))
    metrics = analyzer.compute_metrics()
    assert metrics.average_return_pct == pytest.approx((0.5 - 0.2) / 2)


def test_max_consecutive_losses_hand_computed():
    analyzer = PostTradeAnalyzer(make_config())
    for pnl in [5.0, -1.0, -1.0, -1.0, 5.0, -1.0]:
        analyzer.record_trade(make_trade(pnl))
    metrics = analyzer.compute_metrics()
    assert metrics.max_consecutive_losses == 3


def test_max_consecutive_losses_zero_when_all_wins():
    analyzer = PostTradeAnalyzer(make_config())
    for pnl in [5.0, 5.0, 5.0]:
        analyzer.record_trade(make_trade(pnl))
    metrics = analyzer.compute_metrics()
    assert metrics.max_consecutive_losses == 0


def test_max_drawdown_pct_hand_computed():
    analyzer = PostTradeAnalyzer(make_config())
    analyzer.record_trade(make_trade(pnl=5.0, stake=10.0))
    analyzer.record_trade(make_trade(pnl=5.0, stake=10.0))
    analyzer.record_trade(make_trade(pnl=-6.0, stake=10.0))
    analyzer.record_trade(make_trade(pnl=-6.0, stake=10.0))
    metrics = analyzer.compute_metrics()
    assert metrics.max_drawdown_pct == pytest.approx(0.84, abs=1e-9)


def test_max_drawdown_pct_caps_at_100_percent_after_any_full_loss():
    """
    Documented, expected behavior (see analyzer.py's module docstring):
    a binary-option loss always has return_pct == -1.0 (the full stake is
    forfeited), which zeroes the compounded normalized equity curve the
    moment it occurs — capping max_drawdown_pct at exactly 100% and
    making calmar_ratio a large negative number, regardless of how many
    profitable trades came before or after. This is intrinsic to how
    binary-option returns compound, not a bug in the formula.
    """
    analyzer = PostTradeAnalyzer(make_config())
    for pnl in [50.0, 50.0, 50.0, -10.0, 9.0, 9.0]:  # three big wins, one full loss, two more wins
        analyzer.record_trade(make_trade(pnl, stake=10.0))
    metrics = analyzer.compute_metrics()
    assert metrics.max_drawdown_pct == pytest.approx(1.0, abs=1e-9)
    assert metrics.calmar_ratio < 0


def test_recovery_factor_hand_computed():
    analyzer = PostTradeAnalyzer(make_config())
    for pnl in [20.0, 20.0, -10.0]:
        analyzer.record_trade(make_trade(pnl, stake=10.0))
    metrics = analyzer.compute_metrics()
    assert metrics.recovery_factor == pytest.approx(3.0)


def test_sharpe_ratio_hand_computed():
    analyzer = PostTradeAnalyzer(make_config())
    returns = [0.5, -0.2, 0.3, -0.1]
    for r in returns:
        analyzer.record_trade(make_trade(pnl=r * 10.0, stake=10.0))
    metrics = analyzer.compute_metrics()
    expected_sharpe = np.mean(returns) / np.std(returns, ddof=1)
    assert metrics.sharpe_ratio == pytest.approx(expected_sharpe)


def test_sortino_ratio_hand_computed():
    analyzer = PostTradeAnalyzer(make_config())
    returns = [0.5, -0.2, 0.3, -0.4]
    for r in returns:
        analyzer.record_trade(make_trade(pnl=r * 10.0, stake=10.0))
    metrics = analyzer.compute_metrics()
    downside = np.minimum(returns, 0.0)
    expected_downside_dev = np.sqrt(np.mean(downside**2))
    expected_sortino = np.mean(returns) / expected_downside_dev
    assert metrics.sortino_ratio == pytest.approx(expected_sortino)


def test_brier_score_hand_computed():
    analyzer = PostTradeAnalyzer(make_config())
    analyzer.record_trade(make_trade(pnl=9.0, predicted_probability=0.8))
    analyzer.record_trade(make_trade(pnl=-10.0, predicted_probability=0.6))
    metrics = analyzer.compute_metrics()
    assert metrics.brier_score == pytest.approx((0.04 + 0.36) / 2)


def test_perfect_calibration_gives_zero_ece():
    analyzer = PostTradeAnalyzer(make_config(n_calibration_bins=10))
    for i in range(70):
        analyzer.record_trade(make_trade(pnl=9.0, predicted_probability=0.7, epoch=i))
    for i in range(30):
        analyzer.record_trade(make_trade(pnl=-10.0, predicted_probability=0.7, epoch=100 + i))
    metrics = analyzer.compute_metrics()
    assert metrics.expected_calibration_error == pytest.approx(0.0, abs=1e-9)


def test_miscalibration_gives_positive_ece():
    analyzer = PostTradeAnalyzer(make_config(n_calibration_bins=10))
    for i in range(50):
        analyzer.record_trade(make_trade(pnl=9.0, predicted_probability=0.9, epoch=i))
    for i in range(50):
        analyzer.record_trade(make_trade(pnl=-10.0, predicted_probability=0.9, epoch=100 + i))
    metrics = analyzer.compute_metrics()
    assert metrics.expected_calibration_error > 0.3


def test_calibration_bins_only_include_nonempty_buckets():
    analyzer = PostTradeAnalyzer(make_config(n_calibration_bins=10))
    analyzer.record_trade(make_trade(pnl=9.0, predicted_probability=0.65))
    metrics = analyzer.compute_metrics()
    assert len(metrics.calibration_bins) == 1
    assert metrics.calibration_bins[0].n_trades == 1


def test_rolling_window_uses_only_most_recent_trades():
    analyzer = PostTradeAnalyzer(make_config(rolling_window_trades=3))
    for pnl in [-10.0, -10.0, -10.0, 9.0, 9.0, 9.0]:
        analyzer.record_trade(make_trade(pnl))
    metrics = analyzer.compute_metrics()
    assert metrics.n_trades == 3
    assert metrics.win_rate == pytest.approx(1.0)


def test_explicit_window_overrides_config_default():
    analyzer = PostTradeAnalyzer(make_config(rolling_window_trades=3))
    for pnl in [-10.0, -10.0, -10.0, 9.0, 9.0, 9.0]:
        analyzer.record_trade(make_trade(pnl))
    metrics = analyzer.compute_metrics(window=6)
    assert metrics.n_trades == 6
    assert metrics.win_rate == pytest.approx(0.5)


def test_single_trade_does_not_crash_and_has_zero_std_based_metrics():
    analyzer = PostTradeAnalyzer(make_config())
    analyzer.record_trade(make_trade(pnl=9.0))
    metrics = analyzer.compute_metrics()
    assert metrics.n_trades == 1
    assert metrics.sharpe_ratio == 0.0


def test_was_win_property():
    assert make_trade(pnl=5.0).was_win is True
    assert make_trade(pnl=-5.0).was_win is False
    assert make_trade(pnl=0.0).was_win is False


def test_return_pct_property():
    trade = make_trade(pnl=5.0, stake=20.0)
    assert trade.return_pct == pytest.approx(0.25)


def test_all_trades_returns_full_history_regardless_of_window():
    analyzer = PostTradeAnalyzer(make_config(rolling_window_trades=2))
    for pnl in [1.0, 2.0, 3.0]:
        analyzer.record_trade(make_trade(pnl))
    assert len(analyzer.all_trades()) == 3
