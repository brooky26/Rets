import numpy as np
import pytest

from configs.ev_schema import ExpectedValueConfig
from configs.execution_schema import ExecutionConfig
from configs.opportunity_schema import OpportunityScoringConfig
from configs.paper_trading_schema import PaperTradingConfig
from configs.post_trade_schema import PostTradeAnalysisConfig
from configs.probability_schema import BayesianLogisticConfig
from configs.regime_schema import RuleBasedRegimeConfig
from configs.risk_schema import RiskConfig
from configs.state_encoder_schema import StateEncoderConfig
from data.types import Candle
from features.types import FeatureVector
from paper_trading.orchestrator import PaperTradingOrchestrator, SymbolEvaluation
from regime.rule_based import RuleBasedRegimeDetector
from state_encoder.encoder import MarketStateEncoder
from state_encoder.types import MarketState


def make_synthetic_states_and_closes(n_total: int, seed: int, signal_strength: float = 0.3, symbol: str = "stpRNG"):
    """Same generator as test_walk_forward.py's make_synthetic_run — reused
    deliberately so the live orchestrator is tested against the exact same
    kind of signal the backtester already validates against."""
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


class FakeBrokerClient:
    """Minimal BrokerClient double for live-mode orchestrator tests —
    fixed proposal/buy responses, and a scriptable check_contract_status
    so tests can control exactly when/how a contract "settles"."""

    def __init__(self, status_responses=None, raise_on_status=False):
        # status_responses: list of dicts (or exceptions to raise) consumed
        # one per check_contract_status call; the last entry repeats once exhausted.
        self._status_responses = status_responses or [{"is_sold": 0, "payout": 19.0, "buy_price": 10.0}]
        self._raise_on_status = raise_on_status
        self.status_calls = []

    async def fetch_proposal(self, symbol, contract_type_code, stake, duration_ticks, currency):
        return {"id": "prop1", "ask_price": stake, "payout": stake * 1.9}

    async def buy(self, proposal_id, price):
        return {"contract_id": "c1", "buy_price": price, "payout": price * 1.9}

    async def check_contract_status(self, contract_id: str) -> dict:
        self.status_calls.append(contract_id)
        if self._raise_on_status:
            raise RuntimeError("simulated network failure")
        idx = min(len(self.status_calls) - 1, len(self._status_responses) - 1)
        return self._status_responses[idx]


def make_orchestrator(broker_client=None, execution_mode="paper", **paper_overrides) -> PaperTradingOrchestrator:
    paper_defaults = dict(min_bootstrap_candles=300, starting_equity=1000.0)
    paper_defaults.update(paper_overrides)
    return PaperTradingOrchestrator(
        paper_config=PaperTradingConfig(**paper_defaults),
        probability_config=BayesianLogisticConfig(feature_dims=["trend"], prior_precision=1.0),
        ev_config=ExpectedValueConfig(min_ev_threshold=0.0),
        risk_config=RiskConfig(),
        opportunity_config=OpportunityScoringConfig(base_confidence_threshold=0.3, threshold_min=0.2),
        post_trade_config=PostTradeAnalysisConfig(),
        execution_config=ExecutionConfig(mode=execution_mode),
        platform_environment="live" if execution_mode == "live" else "development",
        regime_detector=RuleBasedRegimeDetector(RuleBasedRegimeConfig()),
        state_encoder=MarketStateEncoder(StateEncoderConfig()),
        broker_client=broker_client,
    )


def make_candle(symbol: str, epoch: int, close: float) -> Candle:
    return Candle(symbol=symbol, epoch=epoch, granularity=60, open=close, high=close, low=close, close=close)


def make_vector(symbol: str, epoch: int) -> FeatureVector:
    """State encoder.encode() is monkeypatched in every test that needs a
    specific MarketState, so the actual feature values here are never read
    for real — this just needs to be a non-None FeatureVector."""
    return FeatureVector(symbol=symbol, epoch=epoch, values={})


# --------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------- #

def test_bootstrap_rejects_mismatched_lengths():
    orch = make_orchestrator()
    states, closes = make_synthetic_states_and_closes(50, seed=0)
    with pytest.raises(ValueError, match="same length"):
        orch.bootstrap("stpRNG", states, closes[:-1])


def test_bootstrap_returns_false_with_insufficient_history():
    orch = make_orchestrator(min_bootstrap_candles=300)
    states, closes = make_synthetic_states_and_closes(50, seed=0)  # far fewer than 300+1
    result = orch.bootstrap("stpRNG", states, closes)
    assert result is False
    assert not orch.is_bootstrapped("stpRNG")


def test_bootstrap_returns_true_and_registers_model_with_sufficient_history():
    orch = make_orchestrator(min_bootstrap_candles=300)
    states, closes = make_synthetic_states_and_closes(400, seed=1)
    result = orch.bootstrap("stpRNG", states, closes)
    assert result is True
    assert orch.is_bootstrapped("stpRNG")
    assert not orch.is_bootstrapped("stpRNG2")  # independent per symbol


# --------------------------------------------------------------------- #
# on_candle before bootstrap: no-op, doesn't raise
# --------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_on_candle_before_bootstrap_is_a_safe_noop():
    orch = make_orchestrator()
    candle = make_candle("stpRNG", epoch=1, close=100.0)
    vector = make_vector("stpRNG", epoch=1)
    result = await orch.on_candle("stpRNG", candle, vector)
    assert result == {"settled": None, "decision": None, "rankings": None, "duration_selection": None}


@pytest.mark.asyncio
async def test_on_candle_with_none_vector_after_bootstrap_only_settles():
    orch = make_orchestrator(min_bootstrap_candles=300)
    states, closes = make_synthetic_states_and_closes(400, seed=2)
    orch.bootstrap("stpRNG", states, closes)

    candle = make_candle("stpRNG", epoch=1, close=100.0)
    result = await orch.on_candle("stpRNG", candle, vector=None)
    assert result["settled"] is None  # nothing was pending
    assert result["decision"] is None  # no vector, no new decision


# --------------------------------------------------------------------- #
# Settlement math
# --------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_settlement_pnl_on_win_matches_payout_minus_stake():
    orch = make_orchestrator(min_bootstrap_candles=50, stake=10.0, assumed_payout_ratio=1.9)
    states, closes = make_synthetic_states_and_closes(200, seed=3, signal_strength=0.9)
    orch.bootstrap("stpRNG", states, closes)

    # Manufacture an approved "buy" directly via a strongly-trending state,
    # bypassing the need to search for one probabilistically.
    strong_up_state = MarketState(
        symbol="stpRNG", epoch=9999, trend=1.0, momentum=0.0, acceleration=0.0,
        volatility=0.1, noise=0.1, persistence=0.0, compression_expansion=0.0,
        complexity=0.0, uncertainty=0.1, liquidity=0.0, market_phase=0.0,
    )
    probability = orch._probability_models["stpRNG"].predict(strong_up_state)
    assert probability.expected_direction in (1, -1)  # sanity: model has an opinion

    vector = make_vector("stpRNG", epoch=100)
    entry_candle = make_candle("stpRNG", epoch=100, close=100.0)

    # Monkeypatch the encoder's encode() to return our manufactured state,
    # isolating this test to settlement arithmetic rather than needing a
    # real feature vector that happens to encode to trend=1.0.
    orch._state_encoder.encode = lambda v, update_normalizer=True: strong_up_state

    entry_result = await orch.on_candle("stpRNG", entry_candle, vector)
    decision = entry_result["decision"]
    assert decision is not None
    assert decision.action == "buy"

    direction = probability.expected_direction
    # Settle with a close that matches the predicted direction (a win).
    winning_close = 100.0 + 1.0 if direction == 1 else 100.0 - 1.0
    settle_candle = make_candle("stpRNG", epoch=101, close=winning_close)
    settle_result = await orch.on_candle("stpRNG", settle_candle, vector=None)

    settled = settle_result["settled"]
    assert settled is not None
    expected_pnl = decision.payout - decision.stake
    assert settled.pnl == pytest.approx(expected_pnl)
    assert settled.was_win is True
    assert orch.equity == pytest.approx(1000.0 + expected_pnl)


@pytest.mark.asyncio
async def test_settlement_pnl_on_loss_is_negative_stake():
    orch = make_orchestrator(min_bootstrap_candles=50, stake=10.0, assumed_payout_ratio=1.9)
    states, closes = make_synthetic_states_and_closes(200, seed=4, signal_strength=0.9)
    orch.bootstrap("stpRNG", states, closes)

    strong_up_state = MarketState(
        symbol="stpRNG", epoch=9999, trend=1.0, momentum=0.0, acceleration=0.0,
        volatility=0.1, noise=0.1, persistence=0.0, compression_expansion=0.0,
        complexity=0.0, uncertainty=0.1, liquidity=0.0, market_phase=0.0,
    )
    probability = orch._probability_models["stpRNG"].predict(strong_up_state)
    orch._state_encoder.encode = lambda v, update_normalizer=True: strong_up_state

    vector = make_vector("stpRNG", epoch=100)
    entry_candle = make_candle("stpRNG", epoch=100, close=100.0)
    entry_result = await orch.on_candle("stpRNG", entry_candle, vector)
    decision = entry_result["decision"]
    assert decision.action == "buy"

    direction = probability.expected_direction
    # Settle with a close that CONTRADICTS the predicted direction (a loss).
    losing_close = 100.0 - 1.0 if direction == 1 else 100.0 + 1.0
    settle_candle = make_candle("stpRNG", epoch=101, close=losing_close)
    settle_result = await orch.on_candle("stpRNG", settle_candle, vector=None)

    settled = settle_result["settled"]
    assert settled is not None
    assert settled.pnl == pytest.approx(-decision.stake)
    assert settled.was_win is False
    assert orch.equity == pytest.approx(1000.0 - decision.stake)


# --------------------------------------------------------------------- #
# Independence across symbols
# --------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_symbols_have_independent_pending_trades():
    orch = make_orchestrator(min_bootstrap_candles=50)
    states_a, closes_a = make_synthetic_states_and_closes(200, seed=5, symbol="stpRNG")
    states_b, closes_b = make_synthetic_states_and_closes(200, seed=6, symbol="stpRNG2")
    orch.bootstrap("stpRNG", states_a, closes_a)
    orch.bootstrap("stpRNG2", states_b, closes_b)

    # Only stpRNG has a candle processed — stpRNG2 must remain untouched.
    candle = make_candle("stpRNG", epoch=1, close=100.0)
    await orch.on_candle("stpRNG", candle, vector=None)

    assert orch._pending_trades["stpRNG2"] is None
    assert "stpRNG2" in orch._pending_trades  # registered by its own bootstrap, unaffected


# --------------------------------------------------------------------- #
# Live settlement: real contract outcome, not next-candle fiction
# --------------------------------------------------------------------- #

async def _open_live_trade(broker, seed=10):
    """Bootstrap, force a strongly-trending state so a buy is approved,
    and drive one entry candle through a live-mode orchestrator. Returns
    (orch, decision) for the caller to inspect/settle further."""
    orch = make_orchestrator(broker_client=broker, execution_mode="live", min_bootstrap_candles=50, stake=10.0)
    states, closes = make_synthetic_states_and_closes(200, seed=seed, signal_strength=0.9)
    orch.bootstrap("stpRNG", states, closes)

    strong_up_state = MarketState(
        symbol="stpRNG", epoch=9999, trend=1.0, momentum=0.0, acceleration=0.0,
        volatility=0.1, noise=0.1, persistence=0.0, compression_expansion=0.0,
        complexity=0.0, uncertainty=0.1, liquidity=0.0, market_phase=0.0,
    )
    orch._state_encoder.encode = lambda v, update_normalizer=True: strong_up_state

    vector = make_vector("stpRNG", epoch=100)
    entry_candle = make_candle("stpRNG", epoch=100, close=100.0)
    entry_result = await orch.on_candle("stpRNG", entry_candle, vector)
    return orch, entry_result["decision"]


@pytest.mark.asyncio
async def test_live_buy_creates_pending_trade_with_real_contract_id():
    broker = FakeBrokerClient()
    orch, decision = await _open_live_trade(broker)
    assert decision.action == "buy"
    assert decision.contract_id == "c1"
    pending = orch._pending_trades["stpRNG"]
    assert pending is not None
    assert pending.contract_id == "c1"
    assert pending.is_awaiting_real_settlement is True


@pytest.mark.asyncio
async def test_live_trade_not_settled_by_next_candle_fiction_while_broker_reports_open():
    broker = FakeBrokerClient(status_responses=[{"is_sold": 0, "payout": 19.0, "buy_price": 10.0}])
    orch, decision = await _open_live_trade(broker)

    # Next candle closes strongly in the predicted direction — under the
    # old paper rule this would immediately settle as a WIN. It must not,
    # because the broker says the contract is still open.
    next_candle = make_candle("stpRNG", epoch=101, close=200.0)
    result = await orch.on_candle("stpRNG", next_candle, vector=None)

    assert result["settled"] is None
    assert orch._pending_trades["stpRNG"] is not None
    assert orch.equity == pytest.approx(1000.0)  # untouched — no fictional pnl applied
    assert broker.status_calls == ["c1"]


@pytest.mark.asyncio
async def test_live_trade_settles_with_real_broker_pnl_on_win():
    broker = FakeBrokerClient(status_responses=[
        {"is_sold": 1, "profit": 9.0, "payout": 19.0, "buy_price": 10.0, "sell_price": 19.0},
    ])
    orch, decision = await _open_live_trade(broker)

    settle_candle = make_candle("stpRNG", epoch=101, close=100.5)  # arbitrary — irrelevant to live settlement
    result = await orch.on_candle("stpRNG", settle_candle, vector=None)

    settled = result["settled"]
    assert settled is not None
    assert settled.exit_reason == "broker_settled"
    assert settled.pnl == pytest.approx(9.0)
    assert orch.equity == pytest.approx(1000.0 + 9.0)
    assert orch._pending_trades["stpRNG"] is None  # cleared, free to open a new trade


@pytest.mark.asyncio
async def test_live_trade_settles_with_real_broker_pnl_on_loss():
    broker = FakeBrokerClient(status_responses=[
        {"is_sold": 1, "profit": -10.0, "payout": 19.0, "buy_price": 10.0, "sell_price": 0.0},
    ])
    orch, decision = await _open_live_trade(broker)

    settle_candle = make_candle("stpRNG", epoch=101, close=100.5)
    result = await orch.on_candle("stpRNG", settle_candle, vector=None)

    settled = result["settled"]
    assert settled.pnl == pytest.approx(-10.0)
    assert orch.equity == pytest.approx(1000.0 - 10.0)


@pytest.mark.asyncio
async def test_live_trade_polling_failure_leaves_trade_pending_not_settled():
    broker = FakeBrokerClient(raise_on_status=True)
    orch, decision = await _open_live_trade(broker)

    settle_candle = make_candle("stpRNG", epoch=101, close=100.5)
    result = await orch.on_candle("stpRNG", settle_candle, vector=None)

    assert result["settled"] is None
    assert orch._pending_trades["stpRNG"] is not None
    assert orch.equity == pytest.approx(1000.0)


@pytest.mark.asyncio
async def test_no_second_live_position_opened_while_one_is_pending():
    broker = FakeBrokerClient(status_responses=[{"is_sold": 0, "payout": 19.0, "buy_price": 10.0}])
    orch, decision = await _open_live_trade(broker)
    first_contract_id = orch._pending_trades["stpRNG"].contract_id

    # Feed another candle with the same strongly-trending state that
    # produced the first buy — if the guard weren't in place, this would
    # try to open a second live position on top of the still-open one.
    next_candle = make_candle("stpRNG", epoch=101, close=101.0)
    vector = make_vector("stpRNG", epoch=101)
    result = await orch.on_candle("stpRNG", next_candle, vector)

    assert result["decision"] is None  # execution never even ran this candle
    assert orch._pending_trades["stpRNG"].contract_id == first_contract_id


@pytest.mark.asyncio
async def test_live_trade_settlement_reason_is_broker_settled_not_next_candle_close():
    broker = FakeBrokerClient(status_responses=[
        {"is_sold": 1, "profit": 9.0, "payout": 19.0, "buy_price": 10.0, "sell_price": 19.0},
    ])
    orch, decision = await _open_live_trade(broker)
    settle_candle = make_candle("stpRNG", epoch=101, close=100.5)
    result = await orch.on_candle("stpRNG", settle_candle, vector=None)
    assert result["settled"].exit_reason != "settled_next_candle_close"
    assert result["settled"].exit_reason == "broker_settled"

@pytest.mark.asyncio
async def test_full_run_produces_trades_and_computable_metrics():
    orch = make_orchestrator(min_bootstrap_candles=200, stake=10.0)
    states, closes = make_synthetic_states_and_closes(600, seed=7, signal_strength=0.6)

    bootstrap_states = states[:250]
    bootstrap_closes = closes[:250]
    orch.bootstrap("stpRNG", bootstrap_states, bootstrap_closes)
    assert orch.is_bootstrapped("stpRNG")

    live_states = states[250:]
    live_closes = closes[250:]
    for i, (state, close) in enumerate(zip(live_states, live_closes)):
        orch._state_encoder.encode = lambda v, update_normalizer=True, s=state: s
        candle = make_candle("stpRNG", epoch=1000 + i, close=close)
        vector = make_vector("stpRNG", epoch=1000 + i)
        await orch.on_candle("stpRNG", candle, vector)

    # With signal_strength=0.6 and permissive thresholds, expect at least
    # some approved trades over 350 live candles.
    assert orch.n_trades_recorded > 0
    # compute_metrics() defaults to a rolling window (PostTradeAnalysisConfig.
    # rolling_window_trades, 200 by default) — request the full history
    # explicitly to compare against n_trades_recorded.
    metrics = orch.metrics(window=orch.n_trades_recorded)
    assert metrics.n_trades == orch.n_trades_recorded
    # equity moved away from the starting point (win or loss, doesn't matter which)
    assert orch.equity != 1000.0


# --------------------------------------------------------------------- #
# Cross-symbol ranking fix — regression tests for the reported bug:
# a symbol could get traded while a genuinely stronger, currently-
# untraded symbol sat idle with no comparison ever taking place. Root
# cause was TWO combined issues: (1) one TradeOpportunityScorer instance
# PER symbol, letting each symbol's adaptive threshold independently
# collapse toward threshold_min; (2) no cross-symbol comparison at all
# before executing a trade.
# --------------------------------------------------------------------- #

def test_orchestrator_uses_one_shared_scorer_not_per_symbol_instances():
    orch = make_orchestrator()
    assert not hasattr(orch, "_scorers")  # the removed per-symbol dict
    from opportunity.scorer import TradeOpportunityScorer
    assert isinstance(orch._scorer, TradeOpportunityScorer)


def test_better_available_competitor_returns_none_when_symbol_is_best():
    orch = make_orchestrator()
    orch._latest_evaluations["stpRNG2"] = SymbolEvaluation(quality_score=0.5, epoch=1, approved=False)
    orch._latest_evaluations["stpRNG3"] = SymbolEvaluation(quality_score=0.6, epoch=1, approved=False)
    result = orch._better_available_competitor("stpRNG", quality_score=0.9)
    assert result is None


def test_better_available_competitor_returns_the_highest_scoring_available_symbol():
    orch = make_orchestrator()
    orch._latest_evaluations["stpRNG2"] = SymbolEvaluation(quality_score=0.5, epoch=1, approved=False)
    orch._latest_evaluations["stpRNG3"] = SymbolEvaluation(quality_score=0.95, epoch=1, approved=False)
    orch._latest_evaluations["stpRNG4"] = SymbolEvaluation(quality_score=0.7, epoch=1, approved=False)
    result = orch._better_available_competitor("stpRNG", quality_score=0.6)
    assert result == "stpRNG3"


def test_better_available_competitor_ignores_symbols_with_a_pending_trade():
    orch = make_orchestrator()
    # stpRNG3 scores highest, but already has an open position — it isn't
    # actually available to compete for this decision, so it must be
    # ignored in favor of stpRNG4 despite stpRNG4's lower score.
    orch._latest_evaluations["stpRNG3"] = SymbolEvaluation(quality_score=0.99, epoch=1, approved=False)
    orch._latest_evaluations["stpRNG4"] = SymbolEvaluation(quality_score=0.7, epoch=1, approved=False)
    orch._pending_trades["stpRNG3"] = "placeholder-not-none"  # any non-None sentinel
    orch._pending_trades["stpRNG4"] = None
    result = orch._better_available_competitor("stpRNG", quality_score=0.6)
    assert result == "stpRNG4"


@pytest.mark.asyncio
async def test_weaker_symbol_is_deferred_when_a_better_available_symbol_exists():
    orch = make_orchestrator(min_bootstrap_candles=50)
    states, closes = make_synthetic_states_and_closes(200, seed=10, signal_strength=0.9)
    orch.bootstrap("stpRNG", states, closes)

    # stpRNG2 has no pending trade and a much higher last-known score.
    orch._pending_trades["stpRNG2"] = None
    orch._latest_evaluations["stpRNG2"] = SymbolEvaluation(quality_score=0.99, epoch=1, approved=False)

    strong_state = MarketState(
        symbol="stpRNG", epoch=9999, trend=1.0, momentum=0.0, acceleration=0.0,
        volatility=0.1, noise=0.1, persistence=0.0, compression_expansion=0.0,
        complexity=0.0, uncertainty=0.1, liquidity=0.0, market_phase=0.0,
    )
    orch._state_encoder.encode = lambda v, update_normalizer=True: strong_state

    candle = make_candle("stpRNG", epoch=100, close=100.0)
    vector = make_vector("stpRNG", epoch=100)
    result = await orch.on_candle("stpRNG", candle, vector)

    assert result["decision"].action != "buy"
    assert any("Deferred" in reason and "stpRNG2" in reason
               for reason in result["decision"].reason.split(";"))
    assert orch._pending_trades["stpRNG"] is None  # no trade was opened


@pytest.mark.asyncio
async def test_symbol_still_trades_when_it_is_the_best_available():
    orch = make_orchestrator(min_bootstrap_candles=50)
    states, closes = make_synthetic_states_and_closes(200, seed=11, signal_strength=0.9)
    orch.bootstrap("stpRNG", states, closes)

    # A weaker competitor is available but should not block stpRNG, since
    # stpRNG's own score will be the highest.
    orch._pending_trades["stpRNG2"] = None
    orch._latest_evaluations["stpRNG2"] = SymbolEvaluation(quality_score=0.01, epoch=1, approved=False)

    strong_state = MarketState(
        symbol="stpRNG", epoch=9999, trend=1.0, momentum=0.0, acceleration=0.0,
        volatility=0.1, noise=0.1, persistence=0.0, compression_expansion=0.0,
        complexity=0.0, uncertainty=0.1, liquidity=0.0, market_phase=0.0,
    )
    orch._state_encoder.encode = lambda v, update_normalizer=True: strong_state

    candle = make_candle("stpRNG", epoch=100, close=100.0)
    vector = make_vector("stpRNG", epoch=100)
    result = await orch.on_candle("stpRNG", candle, vector)

    assert result["decision"].action == "buy"
    assert orch._pending_trades["stpRNG"] is not None


@pytest.mark.asyncio
async def test_current_rankings_reflects_latest_evaluation_after_on_candle():
    orch = make_orchestrator(min_bootstrap_candles=50)
    states, closes = make_synthetic_states_and_closes(200, seed=12, signal_strength=0.9)
    orch.bootstrap("stpRNG", states, closes)

    strong_state = MarketState(
        symbol="stpRNG", epoch=9999, trend=1.0, momentum=0.0, acceleration=0.0,
        volatility=0.1, noise=0.1, persistence=0.0, compression_expansion=0.0,
        complexity=0.0, uncertainty=0.1, liquidity=0.0, market_phase=0.0,
    )
    orch._state_encoder.encode = lambda v, update_normalizer=True: strong_state

    candle = make_candle("stpRNG", epoch=100, close=100.0)
    vector = make_vector("stpRNG", epoch=100)
    await orch.on_candle("stpRNG", candle, vector)

    rankings = orch.current_rankings()
    assert "stpRNG" in rankings
    assert rankings["stpRNG"].epoch == 100


def test_update_regime_detector_swaps_the_active_detector():
    from configs.regime_schema import GaussianHMMConfig
    from regime.hmm_detector import GaussianHMMRegimeDetector

    orch = make_orchestrator()
    original_detector = orch._regime_detector
    new_detector = GaussianHMMRegimeDetector(GaussianHMMConfig(n_states=3))

    orch.update_regime_detector(new_detector)

    assert orch._regime_detector is new_detector
    assert orch._regime_detector is not original_detector
