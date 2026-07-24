"""
Smoke test for main.py's run() wiring itself — not the individual modules
(each already has its own thorough test suite), but the *connections*
between them: does bootstrap actually get called, does a real Tick
flowing through on_tick -> aggregator -> on_candle actually reach the
orchestrator without raising. Compiling main.py catches syntax errors;
this catches wiring mistakes (wrong attribute names, closures referencing
the wrong variable, argument order swaps) that only show up when the
whole thing actually runs once.

DerivWebSocketClient's network methods (fetch_bootstrap_history,
run_forever) are mocked — this is not testing the Deriv connection layer,
which already has its own test suite. `run_forever` is faked to simply
feed a sequence of real Ticks through the client's stored `_on_tick`
callback, which is exactly the seam main.py wires up for real.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from configs.schema import PlatformConfig
from data.types import Candle, DataQualityFlag, Tick


def make_test_config(tmp_path) -> PlatformConfig:
    raw = {
        "market_data": {
            "connection": {"app_id": "12345", "symbols": ["stpRNG"]},
            "historical": {"candle_granularity_seconds": 60},
            "storage": {"backend": "sqlite", "sqlite_path": str(tmp_path / "test_ticks.db")},
        },
        "paper_trading": {"enabled": True, "min_bootstrap_candles": 5},
        "probability_estimation": {"bayesian_logistic": {"feature_dims": ["trend"]}},
        "opportunity_scoring": {"base_confidence_threshold": 0.3, "threshold_min": 0.2},
    }
    return PlatformConfig(**raw)


def make_bootstrap_candles(symbol: str, n: int) -> list[Candle]:
    return [
        Candle(
            symbol=symbol, epoch=i * 60, granularity=60,
            open=100.0 + i * 0.1, high=100.5 + i * 0.1, low=99.5 + i * 0.1,
            close=100.0 + i * 0.1, quality=DataQualityFlag.OK,
        )
        for i in range(n)
    ]


def make_live_ticks(symbol: str, n_candles: int, ticks_per_candle: int, granularity: int) -> list[Tick]:
    """Enough ticks, spaced within/across `granularity`-second windows, for
    CandleAggregator to actually emit `n_candles` completed candles."""
    ticks = []
    base_epoch = 100_000
    price = 105.0
    for c in range(n_candles):
        for t in range(ticks_per_candle):
            epoch = base_epoch + c * granularity + t * (granularity // ticks_per_candle)
            price += 0.01
            ticks.append(Tick(symbol=symbol, epoch=epoch, quote=price, received_at=datetime.now(timezone.utc)))
    return ticks


@pytest.mark.asyncio
async def test_run_bootstraps_then_processes_live_ticks_without_raising(tmp_path):
    import main as main_module

    config = make_test_config(tmp_path)
    config_path = tmp_path / "config.yaml"  # never actually read; load_config is patched below

    bootstrap_candles = make_bootstrap_candles("stpRNG", 30)
    live_ticks = make_live_ticks("stpRNG", n_candles=6, ticks_per_candle=4, granularity=60)

    captured = {"on_tick": None}

    original_init = main_module.DerivWebSocketClient.__init__

    def capturing_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        captured["on_tick"] = self._on_tick

    async def fake_run_forever(self):
        for tick in live_ticks:
            await captured["on_tick"](tick)

    with patch("main.load_config", return_value=config), \
         patch.object(main_module.DerivWebSocketClient, "__init__", new=capturing_init), \
         patch.object(
             main_module.DerivWebSocketClient, "fetch_bootstrap_history",
             new=AsyncMock(return_value={"stpRNG": bootstrap_candles}),
         ), \
         patch.object(main_module.DerivWebSocketClient, "run_forever", new=fake_run_forever), \
         patch.object(main_module.DerivWebSocketClient, "stop", new=AsyncMock()):
        await main_module.run(str(config_path))

    assert captured["on_tick"] is not None  # client was actually constructed and wired


@pytest.mark.asyncio
async def test_run_with_duration_selection_and_hmm_promotion_enabled_does_not_raise(tmp_path):
    """
    Same wiring smoke test as above, but exercising the two newest startup
    paths together: compute_bootstrap_target_candle_count ->
    fetch_bootstrap_history(..., target_candle_count) ->
    run_hmm_promotion_if_enabled -> orchestrator constructed with
    duration_selection_config set. fetch_bootstrap_history is mocked (its
    own pagination logic has a dedicated test suite); this only confirms
    main.py's wiring of the new config toggles doesn't raise end-to-end.
    """
    import main as main_module

    raw = {
        "market_data": {
            "connection": {"app_id": "12345", "symbols": ["stpRNG"]},
            "historical": {"candle_granularity_seconds": 60},
            "storage": {"backend": "sqlite", "sqlite_path": str(tmp_path / "test_ticks2.db")},
        },
        "paper_trading": {
            "enabled": True, "min_bootstrap_candles": 20,
            "use_ensemble_fusion": True, "use_duration_selection": True,
        },
        "probability_estimation": {
            "bayesian_logistic": {"feature_dims": ["trend"]},
            "bagged_gbm": {"feature_dims": ["trend"], "n_estimators": 5},
        },
        "opportunity_scoring": {"base_confidence_threshold": 0.3, "threshold_min": 0.2},
        "regime_detection": {"enable_hmm_promotion": True, "hmm": {"n_states": 2}},
        "champion_challenger": {"min_trades_required": 3},
        "duration_selection": {"candidate_durations_ticks": [3, 5]},
    }
    config = PlatformConfig(**raw)
    config_path = tmp_path / "config2.yaml"

    bootstrap_candles = make_bootstrap_candles("stpRNG", 60)  # enough for bootstrap + HMM train/holdout split
    live_ticks = make_live_ticks("stpRNG", n_candles=4, ticks_per_candle=4, granularity=60)

    captured = {"on_tick": None}
    original_init = main_module.DerivWebSocketClient.__init__

    def capturing_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        captured["on_tick"] = self._on_tick

    async def fake_run_forever(self):
        for tick in live_ticks:
            await captured["on_tick"](tick)

    with patch("main.load_config", return_value=config), \
         patch.object(main_module.DerivWebSocketClient, "__init__", new=capturing_init), \
         patch.object(
             main_module.DerivWebSocketClient, "fetch_bootstrap_history",
             new=AsyncMock(return_value={"stpRNG": bootstrap_candles}),
         ), \
         patch.object(main_module.DerivWebSocketClient, "run_forever", new=fake_run_forever), \
         patch.object(main_module.DerivWebSocketClient, "stop", new=AsyncMock()):
        await main_module.run(str(config_path))

    assert captured["on_tick"] is not None
