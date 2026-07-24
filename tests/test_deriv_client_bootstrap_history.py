import asyncio
import json
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest

from configs.schema import DataIntegrityConfig, DerivConnectionConfig, HistoricalDataConfig
from data.deriv_client import DerivWebSocketClient
from data.integrity import IntegrityValidator


class _FakeBootstrapWebSocket:
    """
    Simulates a real server round-trip for ticks_history requests: `send()`
    parses the request and immediately enqueues the matching canned
    response, which the async iterator (consumed by `_listen()`) then
    yields on its own turn of the event loop — exercising the actual
    request -> _listen() -> _resolve_pending() -> future-resolves path,
    not a shortcut around it.
    """

    def __init__(self, candles_by_symbol: dict[str, list[dict]]):
        self._candles_by_symbol = candles_by_symbol
        self.sent_messages: list[dict] = []
        self._queue: asyncio.Queue = asyncio.Queue()

    async def send(self, raw: str) -> None:
        request = json.loads(raw)
        self.sent_messages.append(request)
        symbol = request["ticks_history"]
        response = {
            "msg_type": "history",
            "req_id": request["req_id"],
            "candles": self._candles_by_symbol.get(symbol, []),
        }
        await self._queue.put(json.dumps(response))

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        while True:
            msg = await self._queue.get()
            yield msg


def make_client():
    connection_config = DerivConnectionConfig(app_id="12345")
    historical_config = HistoricalDataConfig()
    validator = IntegrityValidator(DataIntegrityConfig())

    async def on_tick(tick):
        pass

    return DerivWebSocketClient(
        connection_config=connection_config,
        historical_config=historical_config,
        integrity_validator=validator,
        on_tick=on_tick,
    )


@pytest.mark.asyncio
async def test_fetch_bootstrap_history_returns_candles_per_symbol():
    fake_ws = _FakeBootstrapWebSocket(
        {
            "stpRNG": [{"epoch": 1000, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05}],
            "stpRNG2": [{"epoch": 1000, "open": 2.0, "high": 2.1, "low": 1.9, "close": 2.05}],
        }
    )

    @asynccontextmanager
    async def fake_connect(url, **kwargs):
        yield fake_ws

    client = make_client()
    with patch("data.deriv_client.websockets.connect", fake_connect):
        results = await client.fetch_bootstrap_history(["stpRNG", "stpRNG2"], granularity_seconds=60)

    assert set(results.keys()) == {"stpRNG", "stpRNG2"}
    assert len(results["stpRNG"]) == 1
    assert results["stpRNG"][0].close == 1.05
    assert len(results["stpRNG2"]) == 1
    assert results["stpRNG2"][0].close == 2.05


@pytest.mark.asyncio
async def test_fetch_bootstrap_history_sends_one_request_per_symbol_with_granularity():
    fake_ws = _FakeBootstrapWebSocket({"stpRNG": [], "stpRNG2": []})

    @asynccontextmanager
    async def fake_connect(url, **kwargs):
        yield fake_ws

    client = make_client()
    with patch("data.deriv_client.websockets.connect", fake_connect):
        await client.fetch_bootstrap_history(["stpRNG", "stpRNG2"], granularity_seconds=120)

    assert len(fake_ws.sent_messages) == 2
    symbols_requested = {m["ticks_history"] for m in fake_ws.sent_messages}
    assert symbols_requested == {"stpRNG", "stpRNG2"}
    assert all(m["granularity"] == 120 for m in fake_ws.sent_messages)


@pytest.mark.asyncio
async def test_fetch_bootstrap_history_clears_ws_reference_after_close():
    fake_ws = _FakeBootstrapWebSocket({"stpRNG": []})

    @asynccontextmanager
    async def fake_connect(url, **kwargs):
        yield fake_ws

    client = make_client()
    with patch("data.deriv_client.websockets.connect", fake_connect):
        await client.fetch_bootstrap_history(["stpRNG"])

    assert client._ws is None  # bootstrap connection fully torn down, run_forever() starts fresh


# --------------------------------------------------------------------- #
# Pagination — fetch_bootstrap_history(target_candle_count=...) should
# walk back across multiple requests, using the previous batch's oldest
# epoch as the next request's `end`, until target/lookback/exhaustion.
# --------------------------------------------------------------------- #

class _PaginatingFakeWebSocket:
    """
    Holds a full synthetic candle history for one or more symbols and
    actually respects the requested `end`/`count` — returning the
    `count` candles ending at (or before) `end`, exactly like a real
    paginated ticks_history server would. This is what makes it possible
    to test that successive requests correctly walk backward rather
    than the fixed-response fake used by the simpler tests above.
    """

    def __init__(self, full_history_by_symbol: dict[str, list[dict]]):
        self._full_history = full_history_by_symbol  # ascending by epoch
        self.sent_messages: list[dict] = []
        self._queue: asyncio.Queue = asyncio.Queue()

    async def send(self, raw: str) -> None:
        request = json.loads(raw)
        self.sent_messages.append(request)
        symbol = request["ticks_history"]
        count = request["count"]
        end = request["end"]
        full = self._full_history.get(symbol, [])

        if end == "latest":
            eligible = full
        else:
            eligible = [c for c in full if c["epoch"] <= end]

        page = eligible[-count:] if len(eligible) > count else eligible
        response = {"msg_type": "history", "req_id": request["req_id"], "candles": page}
        await self._queue.put(json.dumps(response))

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        while True:
            msg = await self._queue.get()
            yield msg


def make_synthetic_history(symbol: str, n: int, granularity: int, start_epoch: int = 1_000_000) -> list[dict]:
    return [
        {
            "epoch": start_epoch + i * granularity,
            "open": 100.0 + i * 0.01, "high": 100.5 + i * 0.01,
            "low": 99.5 + i * 0.01, "close": 100.0 + i * 0.01,
        }
        for i in range(n)
    ]


def make_client_with_pagination_delay(request_count_max: int, lookback_days: int = 30, pagination_delay: float = 0.0):
    connection_config = DerivConnectionConfig(app_id="12345")
    historical_config = HistoricalDataConfig(
        request_count_max=request_count_max, lookback_days=lookback_days,
        pagination_delay_seconds=pagination_delay,
    )
    validator = IntegrityValidator(DataIntegrityConfig())

    async def on_tick(tick):
        pass

    return DerivWebSocketClient(
        connection_config=connection_config,
        historical_config=historical_config,
        integrity_validator=validator,
        on_tick=on_tick,
    )


@pytest.mark.asyncio
async def test_pagination_walks_back_across_multiple_pages_to_reach_target():
    granularity = 60
    full_history = make_synthetic_history("stpRNG", n=250, granularity=granularity)
    fake_ws = _PaginatingFakeWebSocket({"stpRNG": full_history})

    @asynccontextmanager
    async def fake_connect(url, **kwargs):
        yield fake_ws

    client = make_client_with_pagination_delay(request_count_max=100, lookback_days=365)
    with patch("data.deriv_client.websockets.connect", fake_connect):
        results = await client.fetch_bootstrap_history(
            ["stpRNG"], granularity_seconds=granularity, target_candle_count=220,
        )

    assert len(results["stpRNG"]) >= 220
    # Must actually have taken multiple requests (100 per page < 220 target).
    assert len(fake_ws.sent_messages) >= 3
    # Result must be sorted ascending with no duplicate epochs.
    epochs = [c.epoch for c in results["stpRNG"]]
    assert epochs == sorted(epochs)
    assert len(epochs) == len(set(epochs))


@pytest.mark.asyncio
async def test_pagination_stops_when_history_is_exhausted_before_target():
    granularity = 60
    full_history = make_synthetic_history("stpRNG", n=150, granularity=granularity)  # less than target
    fake_ws = _PaginatingFakeWebSocket({"stpRNG": full_history})

    @asynccontextmanager
    async def fake_connect(url, **kwargs):
        yield fake_ws

    client = make_client_with_pagination_delay(request_count_max=100, lookback_days=365)
    with patch("data.deriv_client.websockets.connect", fake_connect):
        results = await client.fetch_bootstrap_history(
            ["stpRNG"], granularity_seconds=granularity, target_candle_count=1000,  # unreachable
        )

    # Should stop gracefully with everything that actually exists, not hang or raise.
    assert len(results["stpRNG"]) == 150


@pytest.mark.asyncio
async def test_pagination_respects_lookback_days_limit():
    granularity = 60  # 1 candle/minute
    # 3000 candles at 1/minute = 50 hours of history available.
    full_history = make_synthetic_history("stpRNG", n=3000, granularity=granularity)
    fake_ws = _PaginatingFakeWebSocket({"stpRNG": full_history})

    @asynccontextmanager
    async def fake_connect(url, **kwargs):
        yield fake_ws

    # lookback_days=1 should cut off pagination at ~1440 candles (1 day of
    # 1-minute candles), well short of the full 3000 available or the huge target.
    client = make_client_with_pagination_delay(request_count_max=200, lookback_days=1)
    with patch("data.deriv_client.websockets.connect", fake_connect):
        results = await client.fetch_bootstrap_history(
            ["stpRNG"], granularity_seconds=granularity, target_candle_count=3000,
        )

    assert len(results["stpRNG"]) < 3000  # did not fetch everything available
    # Cutoff can only be checked AFTER a full page is accumulated, so pagination
    # can overshoot the exact 1-day boundary (~1440 candles) by up to one page
    # (200) — 1640 is the correct worst case, not a bug in the stopping logic.
    assert len(results["stpRNG"]) <= 1440 + 200


@pytest.mark.asyncio
async def test_default_target_reproduces_original_single_request_behavior():
    """No target_candle_count specified — must behave exactly as before
    pagination existed: exactly one request per symbol."""
    granularity = 60
    full_history = make_synthetic_history("stpRNG", n=500, granularity=granularity)
    fake_ws = _PaginatingFakeWebSocket({"stpRNG": full_history})

    @asynccontextmanager
    async def fake_connect(url, **kwargs):
        yield fake_ws

    client = make_client_with_pagination_delay(request_count_max=100, lookback_days=365)
    with patch("data.deriv_client.websockets.connect", fake_connect):
        results = await client.fetch_bootstrap_history(["stpRNG"], granularity_seconds=granularity)

    assert len(fake_ws.sent_messages) == 1
    assert len(results["stpRNG"]) == 100
