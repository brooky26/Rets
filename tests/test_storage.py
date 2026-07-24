import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from data.storage import SQLiteTickStore
from data.types import Tick


def make_tick(epoch: int, quote: float, symbol: str = "STPRNG100") -> Tick:
    return Tick(
        symbol=symbol,
        epoch=epoch,
        quote=quote,
        received_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def db_path():
    with tempfile.TemporaryDirectory() as tmp:
        yield str(Path(tmp) / "ticks.db")


@pytest.mark.asyncio
async def test_write_and_read_ticks(db_path):
    store = SQLiteTickStore(db_path, write_batch_size=2, flush_interval_seconds=100)
    ticks = [make_tick(1000 + i, 100.0 + i) for i in range(5)]
    await store.write_ticks(ticks)
    await store._flush()  # force flush remaining buffered ticks

    read_back = await store.read_ticks("STPRNG100", 1000, 1004)
    assert len(read_back) == 5
    assert [t.epoch for t in read_back] == [1000, 1001, 1002, 1003, 1004]
    await store.close()


@pytest.mark.asyncio
async def test_read_ticks_respects_symbol_and_range(db_path):
    store = SQLiteTickStore(db_path, write_batch_size=100, flush_interval_seconds=100)
    await store.write_ticks(
        [
            make_tick(1000, 100.0, symbol="STPRNG100"),
            make_tick(1001, 200.0, symbol="STPRNG200"),
            make_tick(1010, 101.0, symbol="STPRNG100"),
        ]
    )
    await store._flush()

    result = await store.read_ticks("STPRNG100", 0, 1005)
    assert len(result) == 1
    assert result[0].epoch == 1000
    await store.close()


@pytest.mark.asyncio
async def test_upsert_on_duplicate_primary_key(db_path):
    store = SQLiteTickStore(db_path, write_batch_size=100, flush_interval_seconds=100)
    await store.write_ticks([make_tick(1000, 100.0)])
    await store._flush()
    await store.write_ticks([make_tick(1000, 999.0)])  # same (symbol, epoch)
    await store._flush()

    result = await store.read_ticks("STPRNG100", 1000, 1000)
    assert len(result) == 1
    assert result[0].quote == 999.0
    await store.close()
