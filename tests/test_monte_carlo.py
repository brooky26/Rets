import numpy as np
import pytest

from backtesting.monte_carlo import MonteCarloStressTester, circular_block_bootstrap
from configs.backtest_schema import MonteCarloStressConfig
from risk.ruin import risk_of_ruin


def test_circular_block_bootstrap_output_shape():
    rng = np.random.default_rng(0)
    x = np.arange(20, dtype=float)
    paths = circular_block_bootstrap(x, block_size=3, n_paths=50, rng=rng)
    assert paths.shape == (50, 20)


def test_circular_block_bootstrap_block_size_one_matches_original_values():
    rng = np.random.default_rng(1)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    paths = circular_block_bootstrap(x, block_size=1, n_paths=100, rng=rng)
    assert np.all(np.isin(paths, x))


def test_circular_block_bootstrap_preserves_within_block_order():
    rng = np.random.default_rng(2)
    x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    block_size = 3
    paths = circular_block_bootstrap(x, block_size=block_size, n_paths=1, rng=rng)
    path = paths[0]
    for block_start in range(0, len(path) - 1, block_size):
        for i in range(block_start, min(block_start + block_size - 1, len(path) - 1)):
            a, b = path[i], path[i + 1]
            idx_a = int(np.where(x == a)[0][0])
            expected_b = x[(idx_a + 1) % len(x)]
            assert b == expected_b


def test_circular_block_bootstrap_rejects_empty_series():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="empty"):
        circular_block_bootstrap(np.array([]), block_size=1, n_paths=10, rng=rng)


def test_circular_block_bootstrap_rejects_block_size_exceeding_length():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="block_size"):
        circular_block_bootstrap(np.array([1.0, 2.0]), block_size=5, n_paths=10, rng=rng)


def test_stress_tester_rejects_insufficient_history():
    config = MonteCarloStressConfig(n_paths=100)
    tester = MonteCarloStressTester(config)
    with pytest.raises(ValueError, match="at least 2"):
        tester.run([10.0])


def test_stress_tester_basic_run_produces_valid_result():
    config = MonteCarloStressConfig(n_paths=500, block_size=5, starting_capital=1000.0, random_seed=1)
    tester = MonteCarloStressTester(config)
    pnls = [9.0, 9.0, -10.0, 9.0, -10.0, 9.0, 9.0, -10.0] * 20
    result = tester.run(pnls)
    assert result.n_paths == 500
    assert 0.0 <= result.probability_of_ruin <= 1.0
    assert result.final_equity_p5 <= result.final_equity_median <= result.final_equity_p95
    assert 0.0 <= result.max_drawdown_pct_mean <= 1.0


def test_stress_tester_all_positive_pnls_never_ruins():
    config = MonteCarloStressConfig(n_paths=300, block_size=3, starting_capital=100.0, random_seed=2)
    tester = MonteCarloStressTester(config)
    pnls = [5.0] * 50
    result = tester.run(pnls)
    assert result.probability_of_ruin == 0.0


def test_stress_tester_all_negative_pnls_always_ruins_eventually():
    config = MonteCarloStressConfig(n_paths=300, block_size=3, starting_capital=20.0, random_seed=3)
    tester = MonteCarloStressTester(config)
    pnls = [-5.0] * 50
    result = tester.run(pnls)
    assert result.probability_of_ruin == 1.0


def test_empirical_ruin_probability_cross_validates_analytical_formula():
    """
    Generate a large i.i.d. pool of trade outcomes matching
    risk_of_ruin()'s assumptions exactly (win +stake with probability p,
    lose -stake with probability 1-p), block-bootstrap with block_size=1
    (matching the i.i.d. assumption), and confirm the empirical
    probability_of_ruin from Monte Carlo agrees with the analytical
    Cramér-Lundberg result from Stage 7 within a generous tolerance.
    """
    p = 0.55
    reward_to_risk = 1.0
    stake = 10.0
    capital_units = 5.0
    starting_capital = capital_units * stake

    rng = np.random.default_rng(42)
    n_historical = 4000
    outcomes = rng.uniform(0, 1, n_historical) < p
    pnls = np.where(outcomes, stake * reward_to_risk, -stake)

    config = MonteCarloStressConfig(
        n_paths=1500, block_size=1, starting_capital=starting_capital, random_seed=7
    )
    tester = MonteCarloStressTester(config)
    result = tester.run(list(pnls))

    analytical = risk_of_ruin(
        win_probability=p, reward_to_risk=reward_to_risk, capital_in_stake_units=capital_units
    )

    assert abs(result.probability_of_ruin - analytical) < 0.07
