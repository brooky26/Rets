"""
Core data contracts for the Market Data Layer.

These types are the single source of truth for "what a tick/candle looks
like" across the whole platform. Every later module (features, state
encoder, regime detection, etc.) imports from here rather than redefining
its own tick schema — this is what "identical feature generation during
training, validation and live trading" in the spec depends on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum


class DataQualityFlag(str, Enum):
    OK = "ok"
    GAP_DETECTED = "gap_detected"
    PRICE_JUMP_SUSPECT = "price_jump_suspect"
    DUPLICATE = "duplicate"
    OUT_OF_ORDER = "out_of_order"


@dataclass(frozen=True, slots=True)
class Tick:
    """A single price observation for one symbol."""

    symbol: str
    epoch: int              # Unix seconds, as received from Deriv
    quote: float             # price
    received_at: datetime    # local wall-clock time we received it (UTC)
    quality: DataQualityFlag = DataQualityFlag.OK

    @property
    def timestamp(self) -> datetime:
        return datetime.fromtimestamp(self.epoch, tz=timezone.utc)


@dataclass(frozen=True, slots=True)
class Candle:
    """An OHLC candle, either from history or aggregated live from ticks."""

    symbol: str
    epoch: int         # candle open time, Unix seconds
    granularity: int    # seconds
    open: float
    high: float
    low: float
    close: float
    quality: DataQualityFlag = DataQualityFlag.OK
    tick_count: int = 1  # number of ticks aggregated into this candle; 1 for historical/backfilled candles

    @property
    def timestamp(self) -> datetime:
        return datetime.fromtimestamp(self.epoch, tz=timezone.utc)


@dataclass(frozen=True, slots=True)
class ConnectionEvent:
    """Emitted on connect/disconnect/reconnect for monitoring & logging."""

    event: str          # "connected" | "disconnected" | "reconnecting" | "reconnect_failed"
    attempt: int
    detail: str
    occurred_at: datetime
