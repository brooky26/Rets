from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from data.storage import SupabaseTickStore
from data.types import Tick


def make_tick(epoch: int, quote: float, symbol: str = "stpRNG") -> Tick:
    return Tick(
        symbol=symbol,
        epoch=epoch,
        quote=quote,
        received_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _mock_response(status: int, payload):
    resp = MagicMock()
    resp.status = status
    resp.json = AsyncMock(return_value=payload)
    resp.text = AsyncMock(return_value=str(payload))

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _mock_session(post_response_cm=None, get_response_cm=None):
    session = MagicMock()
    if post_response_cm is not None:
        session.post = MagicMock(return_value=post_response_cm)
    if get_response_cm is not None:
        session.get = MagicMock(return_value=get_response_cm)
    session.close = AsyncMock()
    return session


@pytest.fixture
def store():
    return SupabaseTickStore(
        supabase_url="https://project.supabase.co",
        supabase_key="test_key",
        table="ticks",
        write_batch_size=2,
        flush_interval_seconds=100,
    )


def test_requires_url_and_key():
    with pytest.raises(ValueError):
        SupabaseTickStore(supabase_url="", supabase_key="k")
    with pytest.raises(ValueError):
        SupabaseTickStore(supabase_url="https://x.supabase.co", supabase_key="")


@pytest.mark.asyncio
async def test_write_ticks_flushes_at_batch_size_via_upsert_post(store):
    post_cm = _mock_response(201, [])
    session = _mock_session(post_response_cm=post_cm)
    store._session = session
    store._flush_task = None

    await store.write_ticks([make_tick(1000, 100.0), make_tick(1001, 101.0)])

    session.post.assert_called_once()
    call_kwargs = session.post.call_args
    url = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs["url"]
    assert "on_conflict=symbol,epoch" in url
    assert "ticks" in url
    sent_rows = call_kwargs.kwargs["json"]
    assert len(sent_rows) == 2
    assert sent_rows[0]["symbol"] == "stpRNG"
    assert store._buffer == []  # cleared after flush


@pytest.mark.asyncio
async def test_write_ticks_buffers_below_batch_size(store):
    session = _mock_session()
    store._session = session

    await store.write_ticks([make_tick(1000, 100.0)])  # batch size is 2

    session.post.assert_not_called()
    assert len(store._buffer) == 1


@pytest.mark.asyncio
async def test_flush_raises_on_http_error(store):
    post_cm = _mock_response(500, {"message": "server error"})
    session = _mock_session(post_response_cm=post_cm)
    store._session = session

    with pytest.raises(RuntimeError, match="Supabase tick upsert failed"):
        await store.write_ticks([make_tick(1000, 100.0), make_tick(1001, 101.0)])


@pytest.mark.asyncio
async def test_read_ticks_parses_rows_and_filters_by_url(store):
    rows_payload = [
        {
            "symbol": "stpRNG",
            "epoch": 1000,
            "quote": 100.0,
            "received_at": "2026-01-01T00:00:00+00:00",
            "quality": "ok",
        }
    ]
    get_cm = _mock_response(200, rows_payload)
    session = _mock_session(get_response_cm=get_cm)
    store._session = session

    result = await store.read_ticks("stpRNG", 1000, 2000)

    session.get.assert_called_once()
    url = session.get.call_args.args[0]
    assert "symbol=eq.stpRNG" in url
    assert "epoch=gte.1000" in url
    assert "epoch=lte.2000" in url
    assert len(result) == 1
    assert result[0].symbol == "stpRNG"
    assert result[0].epoch == 1000
    assert result[0].quote == 100.0


@pytest.mark.asyncio
async def test_read_ticks_raises_on_http_error(store):
    get_cm = _mock_response(404, {"message": "not found"})
    session = _mock_session(get_response_cm=get_cm)
    store._session = session

    with pytest.raises(RuntimeError, match="Supabase tick read failed"):
        await store.read_ticks("stpRNG", 1000, 2000)


@pytest.mark.asyncio
async def test_close_flushes_remaining_buffer_and_closes_session(store):
    post_cm = _mock_response(201, [])
    session = _mock_session(post_response_cm=post_cm)
    store._session = session
    store._buffer = [make_tick(1000, 100.0)]  # below batch size, still pending

    await store.close()

    session.post.assert_called_once()  # flush happened
    session.close.assert_awaited_once()


def test_build_tick_store_selects_sqlite_backend(tmp_path):
    from configs.schema import StorageConfig
    from main import build_tick_store
    from data.storage import SQLiteTickStore

    cfg = StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "t.db"))
    store = build_tick_store(cfg)
    assert isinstance(store, SQLiteTickStore)


def test_build_tick_store_selects_supabase_backend():
    from configs.schema import StorageConfig
    from main import build_tick_store

    cfg = StorageConfig(
        backend="supabase",
        supabase_url="https://x.supabase.co",
        supabase_key="sk_test",
    )
    store = build_tick_store(cfg)
    assert isinstance(store, SupabaseTickStore)
