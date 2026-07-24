"""
Tick-to-Candle Aggregator.

The Deriv WebSocket client streams ticks; the Feature Engineering
Pipeline consumes candles (OHLC). This module bridges the two: it buckets
incoming ticks into fixed-width time windows (`granularity_seconds`) and
emits a completed Candle each time a window closes.

Bucketing rule: a tick with epoch `e` belongs to the window starting at
`floor(e / granularity) * granularity`. A window is considered closed
(and its candle emitted) the moment a tick arrives belonging to a later
window — this is a standard "watermark on next observation" approach,
appropriate here since Deriv's synthetic indices tick continuously with
no exchange-close gaps to wait out.
"""

from __future__ import annotations

from data.types import Candle, DataQualityFlag, Tick


class _OpenCandle:
    __slots__ = ("symbol", "epoch", "granularity", "open", "high", "low", "close", "has_suspect_tick", "tick_count")

    def __init__(self, symbol: str, epoch: int, granularity: int, price: float) -> None:
        self.symbol = symbol
        self.epoch = epoch
        self.granularity = granularity
        self.open = price
        self.high = price
        self.low = price
        self.close = price
        self.has_suspect_tick = False
        self.tick_count = 1

    def update(self, price: float) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.tick_count += 1

    def to_candle(self) -> Candle:
        return Candle(
            symbol=self.symbol,
            epoch=self.epoch,
            granularity=self.granularity,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            quality=DataQualityFlag.PRICE_JUMP_SUSPECT
            if self.has_suspect_tick
            else DataQualityFlag.OK,
            tick_count=self.tick_count,
        )


class CandleAggregator:
    def __init__(self, granularity_seconds: int) -> None:
        if granularity_seconds <= 0:
            raise ValueError("granularity_seconds must be positive")
        self._granularity = granularity_seconds
        self._open_candles: dict[str, _OpenCandle] = {}

    def _bucket_epoch(self, epoch: int) -> int:
        return (epoch // self._granularity) * self._granularity

    def on_tick(self, tick: Tick) -> Candle | None:
        """
        Feed one tick. Returns a completed Candle if this tick closed the
        previous window for its symbol, else None.
        """
        bucket = self._bucket_epoch(tick.epoch)
        open_candle = self._open_candles.get(tick.symbol)

        # Drop duplicates/out-of-order ticks entirely — they shouldn't
        # perturb OHLC aggregation. (The integrity validator upstream is
        # responsible for flagging these; here we simply don't aggregate them.)
        if tick.quality in (DataQualityFlag.DUPLICATE, DataQualityFlag.OUT_OF_ORDER):
            return None

        if open_candle is None:
            self._open_candles[tick.symbol] = _OpenCandle(
                tick.symbol, bucket, self._granularity, tick.quote
            )
            return None

        if bucket == open_candle.epoch:
            open_candle.update(tick.quote)
            if tick.quality == DataQualityFlag.PRICE_JUMP_SUSPECT:
                open_candle.has_suspect_tick = True
            return None

        # New window started -> close out the previous candle and start a new one.
        completed = open_candle.to_candle()
        self._open_candles[tick.symbol] = _OpenCandle(
            tick.symbol, bucket, self._granularity, tick.quote
        )
        return completed

    def flush(self, symbol: str) -> Candle | None:
        """Force-close the current open candle for a symbol (e.g. on shutdown)."""
        open_candle = self._open_candles.pop(symbol, None)
        return open_candle.to_candle() if open_candle is not None else None
