from datetime import datetime, timezone

import pytest

from data.candle_aggregator import CandleAggregator
from data.types import DataQualityFlag, Tick


def make_tick(epoch: int, quote: float, symbol: str = "STPRNG100", quality=DataQualityFlag.OK) -> Tick:
    return Tick(
        symbol=symbol,
        epoch=epoch,
        quote=quote,
        received_at=datetime.now(timezone.utc),
        quality=quality,
    )


def test_no_candle_emitted_within_first_window():
    agg = CandleAggregator(granularity_seconds=60)
    # all three land in bucket [960, 1020)
    assert agg.on_tick(make_tick(960, 100.0)) is None
    assert agg.on_tick(make_tick(970, 101.0)) is None
    assert agg.on_tick(make_tick(980, 99.0)) is None


def test_candle_emitted_when_new_window_starts():
    agg = CandleAggregator(granularity_seconds=60)
    agg.on_tick(make_tick(960, 100.0))   # bucket [960, 1020)
    agg.on_tick(make_tick(970, 105.0))
    agg.on_tick(make_tick(980, 95.0))
    agg.on_tick(make_tick(990, 102.0))
    candle = agg.on_tick(make_tick(1020, 103.0))  # crosses into next 60s bucket
    assert candle is not None
    assert candle.open == 100.0
    assert candle.high == 105.0
    assert candle.low == 95.0
    assert candle.close == 102.0


def test_ohlc_correctness_over_multiple_windows():
    agg = CandleAggregator(granularity_seconds=10)
    # window [0,10): prices 10, 12, 8 -> O=10 H=12 L=8 C=8
    agg.on_tick(make_tick(0, 10.0))
    agg.on_tick(make_tick(3, 12.0))
    agg.on_tick(make_tick(7, 8.0))
    # crossing to window [10,20) closes the first candle
    candle1 = agg.on_tick(make_tick(10, 9.0))
    assert (candle1.open, candle1.high, candle1.low, candle1.close) == (10.0, 12.0, 8.0, 8.0)
    assert candle1.epoch == 0


def test_symbols_are_independent():
    agg = CandleAggregator(granularity_seconds=60)
    agg.on_tick(make_tick(1000, 100.0, symbol="A"))
    agg.on_tick(make_tick(1000, 200.0, symbol="B"))
    candle_a = agg.on_tick(make_tick(1060, 101.0, symbol="A"))
    candle_b_should_be_none = agg.on_tick(make_tick(1010, 201.0, symbol="B"))
    assert candle_a is not None
    assert candle_a.symbol == "A"
    assert candle_b_should_be_none is None  # B still in its first window


def test_duplicate_and_out_of_order_ticks_ignored():
    agg = CandleAggregator(granularity_seconds=60)
    agg.on_tick(make_tick(1000, 100.0))
    result = agg.on_tick(make_tick(1000, 999.0, quality=DataQualityFlag.DUPLICATE))
    assert result is None
    # The duplicate's garbage price should not have polluted the open candle.
    flushed = agg.flush("STPRNG100")
    assert flushed.high == 100.0
    assert flushed.low == 100.0


def test_suspect_price_jump_flagged_on_candle():
    agg = CandleAggregator(granularity_seconds=60)
    agg.on_tick(make_tick(1000, 100.0))
    agg.on_tick(make_tick(1010, 500.0, quality=DataQualityFlag.PRICE_JUMP_SUSPECT))
    candle = agg.on_tick(make_tick(1060, 101.0))
    assert candle.quality == DataQualityFlag.PRICE_JUMP_SUSPECT


def test_flush_returns_open_candle_and_clears_it():
    agg = CandleAggregator(granularity_seconds=60)
    agg.on_tick(make_tick(1000, 100.0))
    agg.on_tick(make_tick(1010, 103.0))
    flushed = agg.flush("STPRNG100")
    assert flushed is not None
    assert flushed.close == 103.0
    assert agg.flush("STPRNG100") is None  # nothing left to flush


def test_zero_or_negative_granularity_rejected():
    with pytest.raises(ValueError):
        CandleAggregator(granularity_seconds=0)
    with pytest.raises(ValueError):
        CandleAggregator(granularity_seconds=-5)


def test_tick_count_tracked_per_candle():
    agg = CandleAggregator(granularity_seconds=60)
    agg.on_tick(make_tick(960, 100.0))
    agg.on_tick(make_tick(970, 101.0))
    agg.on_tick(make_tick(980, 102.0))
    candle = agg.on_tick(make_tick(1020, 103.0))  # closes the first candle
    assert candle.tick_count == 3


def test_tick_count_excludes_duplicates():
    agg = CandleAggregator(granularity_seconds=60)
    agg.on_tick(make_tick(960, 100.0))
    agg.on_tick(make_tick(960, 999.0, quality=DataQualityFlag.DUPLICATE))
    flushed = agg.flush("STPRNG100")
    assert flushed.tick_count == 1
