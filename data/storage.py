"""
Local persistence for the Market Data Layer.

SQLite is the default backend (single-file, zero-ops, fine for research
and single-instance paper/live trading). A Postgres backend can be added
later behind the same `TickStore` interface without touching callers —
this is the "interfaces and abstraction to allow future replacement"
principle from the spec applied to storage.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Protocol

from data.types import DataQualityFlag, Tick


class TickStore(Protocol):
    async def write_ticks(self, ticks: list[Tick]) -> None: ...
    async def read_ticks(
        self, symbol: str, start_epoch: int, end_epoch: int
    ) -> list[Tick]: ...
    async def close(self) -> None: ...


class SQLiteTickStore:
    """
    Batched, buffered SQLite writer. Ticks are accumulated in memory and
    flushed either when `write_batch_size` is reached or every
    `flush_interval_seconds`, whichever comes first — avoids a disk write
    per tick under high-frequency streaming.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS ticks (
        symbol TEXT NOT NULL,
        epoch INTEGER NOT NULL,
        quote REAL NOT NULL,
        received_at TEXT NOT NULL,
        quality TEXT NOT NULL,
        PRIMARY KEY (symbol, epoch)
    );
    CREATE INDEX IF NOT EXISTS idx_ticks_symbol_epoch ON ticks(symbol, epoch);
    """

    def __init__(
        self,
        db_path: str,
        write_batch_size: int = 500,
        flush_interval_seconds: float = 2.0,
    ) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._write_batch_size = write_batch_size
        self._flush_interval_seconds = flush_interval_seconds
        self._buffer: list[Tick] = []
        self._lock = asyncio.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(self._SCHEMA)
        self._conn.commit()
        self._flush_task: asyncio.Task | None = None

    async def start(self) -> None:
        self._flush_task = asyncio.create_task(self._periodic_flush())

    async def _periodic_flush(self) -> None:
        while True:
            await asyncio.sleep(self._flush_interval_seconds)
            await self._flush()

    async def write_ticks(self, ticks: list[Tick]) -> None:
        async with self._lock:
            self._buffer.extend(ticks)
            if len(self._buffer) >= self._write_batch_size:
                await self._flush_locked()

    async def _flush(self) -> None:
        async with self._lock:
            await self._flush_locked()

    async def _flush_locked(self) -> None:
        if not self._buffer:
            return
        rows = [
            (t.symbol, t.epoch, t.quote, t.received_at.isoformat(), t.quality.value)
            for t in self._buffer
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO ticks (symbol, epoch, quote, received_at, quality) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        self._buffer.clear()

    async def read_ticks(
        self, symbol: str, start_epoch: int, end_epoch: int
    ) -> list[Tick]:
        cur = self._conn.execute(
            "SELECT symbol, epoch, quote, received_at, quality FROM ticks "
            "WHERE symbol = ? AND epoch >= ? AND epoch <= ? ORDER BY epoch ASC",
            (symbol, start_epoch, end_epoch),
        )
        result = []
        for row in cur.fetchall():
            from datetime import datetime

            result.append(
                Tick(
                    symbol=row[0],
                    epoch=row[1],
                    quote=row[2],
                    received_at=datetime.fromisoformat(row[3]),
                    quality=DataQualityFlag(row[4]),
                )
            )
        return result

    async def close(self) -> None:
        if self._flush_task is not None:
            self._flush_task.cancel()
        await self._flush()
        self._conn.close()


class SupabaseTickStore:
    """
    Postgres-backed TickStore via Supabase's REST (PostgREST) API.

    Same buffered/batched write pattern as SQLiteTickStore (accumulate in
    memory, flush on batch size or interval), same `TickStore` Protocol —
    swapping this in for `SQLiteTickStore` requires no caller changes.
    Exists specifically because Railway's filesystem is ephemeral: a local
    SQLite file does not survive a redeploy/restart, but this does.

    Upsert semantics match SQLite's `INSERT OR REPLACE`: conflicts on
    (symbol, epoch) overwrite the existing row. Requires a unique
    constraint/index on (symbol, epoch) in the Supabase table for the
    upsert's `on_conflict` to work correctly — see the table DDL in
    README's Supabase setup section.
    """

    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        table: str = "ticks",
        write_batch_size: int = 500,
        flush_interval_seconds: float = 2.0,
        request_timeout_seconds: float = 15.0,
    ) -> None:
        if not supabase_url or not supabase_key:
            raise ValueError("SupabaseTickStore requires both supabase_url and supabase_key.")
        self._url = supabase_url.rstrip("/")
        self._key = supabase_key
        self._table = table
        self._write_batch_size = write_batch_size
        self._flush_interval_seconds = flush_interval_seconds
        self._timeout = request_timeout_seconds
        self._buffer: list[Tick] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None
        self._session: "aiohttp.ClientSession | None" = None

    def _headers(self, prefer: str | None = None) -> dict:
        headers = {
            "apikey": self._key,
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }
        if prefer:
            headers["Prefer"] = prefer
        return headers

    async def start(self) -> None:
        import aiohttp

        self._session = aiohttp.ClientSession()
        self._flush_task = asyncio.create_task(self._periodic_flush())

    async def _periodic_flush(self) -> None:
        while True:
            await asyncio.sleep(self._flush_interval_seconds)
            await self._flush()

    async def write_ticks(self, ticks: list[Tick]) -> None:
        async with self._lock:
            self._buffer.extend(ticks)
            if len(self._buffer) >= self._write_batch_size:
                await self._flush_locked()

    async def _flush(self) -> None:
        async with self._lock:
            await self._flush_locked()

    async def _flush_locked(self) -> None:
        if not self._buffer:
            return
        rows = [
            {
                "symbol": t.symbol,
                "epoch": t.epoch,
                "quote": t.quote,
                "received_at": t.received_at.isoformat(),
                "quality": t.quality.value,
            }
            for t in self._buffer
        ]
        url = f"{self._url}/rest/v1/{self._table}?on_conflict=symbol,epoch"
        assert self._session is not None, "call start() before writing"
        async with self._session.post(
            url,
            json=rows,
            headers=self._headers(prefer="resolution=merge-duplicates"),
            timeout=self._timeout,
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(
                    f"Supabase tick upsert failed ({resp.status}): {body}"
                )
        self._buffer.clear()

    async def read_ticks(
        self, symbol: str, start_epoch: int, end_epoch: int
    ) -> list[Tick]:
        from datetime import datetime

        url = (
            f"{self._url}/rest/v1/{self._table}"
            f"?symbol=eq.{symbol}&epoch=gte.{start_epoch}&epoch=lte.{end_epoch}"
            f"&order=epoch.asc"
        )
        assert self._session is not None, "call start() before reading"
        async with self._session.get(
            url, headers=self._headers(), timeout=self._timeout
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(
                    f"Supabase tick read failed ({resp.status}): {body}"
                )
            rows = await resp.json()

        return [
            Tick(
                symbol=row["symbol"],
                epoch=row["epoch"],
                quote=row["quote"],
                received_at=datetime.fromisoformat(row["received_at"]),
                quality=DataQualityFlag(row["quality"]),
            )
            for row in rows
        ]

    async def close(self) -> None:
        if self._flush_task is not None:
            self._flush_task.cancel()
        await self._flush()
        if self._session is not None:
            await self._session.close()
