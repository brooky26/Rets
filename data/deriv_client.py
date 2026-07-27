"""
Deriv WebSocket Client — Market Data Layer entry point.

Connection layer: REST OTP bootstrap (current Deriv Options API)
-------------------------------------------------------------------
Deriv retired the legacy pattern of connecting directly to
`wss://ws.derivws.com/websockets/v3?app_id=...`. The current flow is:

  Public / unauthenticated (market data only):
      connect directly to `ws_public_url`
      (wss://api.derivws.com/trading/v1/options/ws/public)

  Authenticated (needed for anything beyond public ticks/candles):
      1. POST {rest_base_url}/trading/v1/options/accounts/{account_id}/otp
         headers: Deriv-App-ID: <app_id>, Authorization: Bearer <api_token>
      2. Response: {"data": {"url": "wss://.../ws/demo?otp=..."}}
      3. Connect directly to that URL.

OTP tokens are short-lived. This client requests a fresh OTP on every
reconnect (not just the first connect) — reusing a stale OTP after a
drop will fail authentication.

Everything downstream of "how did we get a URL to connect to" —
subscription messages, tick parsing, reconnect backoff — is unchanged
from before; the message-level JSON-RPC schema (ticks/ticks_history
requests, tick/history responses) is still what Deriv's WS speaks, just
over the new transport, with tick.symbol/epoch/quote now guaranteed
present in every tick message (previously optional).

Note on `proposal` requests specifically — CORRECTED, see below:
An earlier version of this comment claimed the field should be
"symbol", based on a unit test's assumption. That was wrong: a live
account on this API rejects "symbol" outright —
`{"code": "InputValidationFailed", "message": "Input validation
failed: Properties not allowed: symbol."}` — confirmed against real
deployment logs, same as the ticks_history precedent below. The field
IS "underlying_symbol". Additionally, "subscribe" must be sent
explicitly as `1` (mirroring ticks_history's confirmed behavior below)
— omitting it produced zero response from Deriv, ever, for any
proposal request, which is consistent with Deriv accepting the
request as schema-valid but never generating a priced response
without an explicit subscription. `tests/test_deriv_client_execution.py`
has been updated to match; don't trust that test's prior assertion of
"symbol" over live API evidence again — confirm against an actual
`proposal` request/response pair, not an unrelated endpoint's schema.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import time
from datetime import datetime, timezone
from typing import Awaitable, Callable

import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed

from configs.schema import DerivConnectionConfig, HistoricalDataConfig
from data.integrity import IntegrityValidator
from data.types import Candle, ConnectionEvent, DataQualityFlag, Tick

logger = logging.getLogger(__name__)

TickCallback = Callable[[Tick], Awaitable[None]]
ConnectionEventCallback = Callable[[ConnectionEvent], Awaitable[None]]
# Fired when a buy()/buy_direct() call already timed out locally but Deriv's
# response then arrives late and shows a contract WAS opened. Args are
# (symbol, buy_result) where buy_result is the raw `buy` dict (contract_id,
# buy_price, payout, ...). Without this hook, _on_late_buy_response could only
# log the fact — nothing downstream (risk equity, ContractOutcomeTracker,
# post-trade analytics) ever learned the contract exists. See that method's
# docstring for the full history of why this was previously log-only.
LateContractCallback = Callable[[str, dict], Awaitable[None]]


class DerivClientError(Exception):
    """Raised for unrecoverable protocol-level errors (e.g. bad app_id, OTP failure)."""


class DerivOTPBootstrap:
    """
    Handles the REST leg of the connection: exchanging an api_token for a
    fresh, ready-to-use authenticated WebSocket URL. Isolated from the
    WebSocket client itself so it's independently testable (mock the HTTP
    call, no live socket needed) and reusable by other modules later
    (e.g. execution) that also need an authenticated connection.
    """

    def __init__(self, config: DerivConnectionConfig) -> None:
        self._config = config

    async def fetch_authenticated_ws_url(self) -> str:
        if not self._config.is_authenticated_mode:
            raise DerivClientError(
                "fetch_authenticated_ws_url called without api_token/account_id configured."
            )
        endpoint = (
            f"{self._config.rest_base_url}/trading/v1/options/accounts/"
            f"{self._config.account_id}/otp"
        )
        headers = {
            "Deriv-App-ID": self._config.app_id,
            "Authorization": f"Bearer {self._config.api_token}",
        }
        timeout = aiohttp.ClientTimeout(total=self._config.request_timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, headers=headers) as resp:
                payload = await resp.json()
                if resp.status != 200:
                    errors = payload.get("errors", payload)
                    raise DerivClientError(
                        f"OTP request failed (status={resp.status}): {errors}"
                    )
                url = payload.get("data", {}).get("url")
                if not url:
                    raise DerivClientError(
                        f"OTP response missing data.url: {payload}"
                    )
                return url


async def resolve_account_id(config: DerivConnectionConfig) -> str:
    """
    Auto-discovers the Deriv account_id for `config.api_token`, so the
    operator doesn't have to look it up by hand — calls
    `GET {rest_base_url}/trading/v1/options/accounts` (the same accounts
    resource `POST .../accounts` creates and `POST .../accounts/{id}/otp`
    authenticates against), which returns every account tied to this
    token. Picks the account whose `account_type` matches
    `config.ws_account_type` ("demo" or "real") and whose `status` is
    "active".

    Raises `DerivClientError` (not a silent fallback) if:
      - no account of the requested type is active on this token — the
        operator needs to check `ws_account_type` or create one, or
      - more than one active account of that type exists — ambiguous;
        the operator should set `account_id` explicitly (env var
        `DERIV_ACCOUNT_ID`) to pick which one, rather than this
        function guessing on their behalf.
    """
    if config.api_token is None:
        raise DerivClientError("resolve_account_id called without api_token configured.")

    endpoint = f"{config.rest_base_url}/trading/v1/options/accounts"
    headers = {
        "Deriv-App-ID": config.app_id,
        "Authorization": f"Bearer {config.api_token}",
    }
    timeout = aiohttp.ClientTimeout(total=config.request_timeout_seconds)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(endpoint, headers=headers) as resp:
            payload = await resp.json()
            if resp.status != 200:
                errors = payload.get("errors", payload)
                raise DerivClientError(
                    f"Account lookup failed (status={resp.status}): {errors} — "
                    "set DERIV_ACCOUNT_ID explicitly if this keeps failing."
                )
            data = payload.get("data", [])
            accounts = data if isinstance(data, list) else [data]

    matches = [
        a for a in accounts
        if a.get("account_type") == config.ws_account_type and a.get("status") == "active"
    ]
    if not matches:
        raise DerivClientError(
            f"No active '{config.ws_account_type}' account found for this api_token "
            f"(accounts seen: {[(a.get('account_id'), a.get('account_type'), a.get('status')) for a in accounts]}). "
            "Set DERIV_ACCOUNT_ID explicitly, or double-check ws_account_type/DERIV_ACCOUNT_TYPE."
        )
    if len(matches) > 1:
        candidate_ids = [a.get("account_id") for a in matches]
        raise DerivClientError(
            f"Multiple active '{config.ws_account_type}' accounts found for this api_token: "
            f"{candidate_ids} — ambiguous which to use. Set DERIV_ACCOUNT_ID explicitly to one of these."
        )

    account_id = matches[0]["account_id"]
    logger.info(
        "Auto-resolved Deriv account_id=%s (account_type=%s) from api_token — "
        "set DERIV_ACCOUNT_ID explicitly to skip this lookup or override it.",
        account_id, config.ws_account_type,
    )
    return account_id


async def ensure_account_id(config: DerivConnectionConfig) -> DerivConnectionConfig:
    """
    Returns `config` unchanged if `account_id` is already set (explicit
    always wins — no lookup performed) or if `api_token` isn't set at all
    (public-data-only mode has no account to resolve). Otherwise calls
    `resolve_account_id` and returns a copy of `config` with `account_id`
    filled in. Callers (main.py, continuous_learning's CLI) should call
    this once at startup, before constructing anything that reads
    `config.account_id`.
    """
    if config.account_id is not None or config.api_token is None:
        return config
    resolved = await resolve_account_id(config)
    return config.model_copy(update={"account_id": resolved})


class DerivWebSocketClient:
    def __init__(
        self,
        connection_config: DerivConnectionConfig,
        historical_config: HistoricalDataConfig,
        integrity_validator: IntegrityValidator,
        on_tick: TickCallback,
        on_connection_event: ConnectionEventCallback | None = None,
        otp_bootstrap: DerivOTPBootstrap | None = None,
        on_late_contract: LateContractCallback | None = None,
    ) -> None:
        self._cfg = connection_config
        self._hist_cfg = historical_config
        self._validator = integrity_validator
        self._on_tick = on_tick
        self._on_connection_event = on_connection_event
        self._on_late_contract = on_late_contract
        self._otp = otp_bootstrap or DerivOTPBootstrap(connection_config)
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._req_id_counter = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._running = False
        self._reconnect_attempt = 0

    # ------------------------------------------------------------------ #
    # Public lifecycle
    # ------------------------------------------------------------------ #

    async def run_forever(self) -> None:
        """Connect, subscribe, stream ticks; reconnect automatically on drop."""
        self._running = True
        while self._running:
            try:
                await self._connect_and_stream()
            except (ConnectionClosed, OSError, asyncio.TimeoutError) as exc:
                await self._emit_event("disconnected", detail=str(exc))
                if not self._running:
                    break
                should_continue = await self._handle_reconnect()
                if not should_continue:
                    break
            except DerivClientError:
                raise

    async def stop(self) -> None:
        self._running = False
        if self._ws is not None:
            await self._ws.close()

    # ------------------------------------------------------------------ #
    # Connection + subscription
    # ------------------------------------------------------------------ #

    async def _resolve_connect_url(self) -> str:
        """
        Public mode: connect straight to ws_public_url.
        Authenticated mode: fetch a *fresh* OTP-embedded URL every time
        this is called — never reuse a URL from a previous connection,
        since OTP tokens are short-lived and reconnects need a new one.
        """
        if not self._cfg.is_authenticated_mode:
            return self._cfg.ws_public_url
        return await self._otp.fetch_authenticated_ws_url()

    async def _connect_and_stream(self) -> None:
        # Any future still sitting in self._pending belongs to the previous
        # (now-dead) connection — a late response can never arrive on a socket
        # that no longer exists. shield() in fetch_proposal deliberately keeps
        # timed-out futures alive so a late response on the SAME connection can
        # still be caught; across a reconnect that guarantee is moot, so clear
        # them here rather than leaking one future per timed-out request for
        # the life of the process.
        for pending_future in self._pending.values():
            if not pending_future.done():
                pending_future.cancel()
        self._pending.clear()

        url = await self._resolve_connect_url()
        async with websockets.connect(
            url, ping_interval=self._cfg.ping_interval_seconds, ping_timeout=self._cfg.ping_timeout_seconds,
        ) as ws:
            self._ws = ws
            self._reconnect_attempt = 0
            await self._emit_event("connected", detail=self._redact(url))

            for symbol in self._cfg.symbols:
                await self._subscribe_ticks(ws, symbol)

            await self._listen(ws)

    @staticmethod
    def _redact(url: str) -> str:
        """Never log a live OTP token."""
        if "otp=" in url:
            base, _, _ = url.partition("otp=")
            return f"{base}otp=<redacted>"
        return url

    async def _subscribe_ticks(self, ws, symbol: str) -> None:
        req_id = self._next_req_id()
        await ws.send(
            json.dumps({"ticks": symbol, "subscribe": 1, "req_id": req_id})
        )

    async def _listen(self, ws) -> None:
        async for raw_message in ws:
            message = json.loads(raw_message)

            if message.get("error"):
                logger.warning(
                    "Deriv API error (msg_type=%s, echo_req=%s): %s",
                    message.get("msg_type"),
                    message.get("echo_req"),
                    message["error"],
                )
                self._resolve_pending(message)  # also resolve pending requests with the error
                continue

            msg_type = message.get("msg_type")
            if msg_type == "tick":
                await self._handle_tick_message(message)
            elif msg_type in ("history", "candles", "proposal", "buy", "proposal_open_contract", "forget"):
                self._resolve_pending(message)

    async def _handle_tick_message(self, message: dict) -> None:
        # Error responses to a tick subscription still arrive with
        # msg_type == "tick" but no "tick" payload — those are caught by the
        # error check in _listen before this is called, but guard here too
        # in case a symbol is delisted/invalid mid-stream and Deriv sends a
        # tick-shaped message without the field regardless.
        tick_data = message.get("tick")
        if not tick_data:
            logger.warning("Received tick-typed message with no tick payload: %s", message)
            return
        raw_tick = Tick(
            symbol=tick_data["symbol"],
            epoch=int(tick_data["epoch"]),
            quote=float(tick_data["quote"]),
            received_at=datetime.now(timezone.utc),
        )
        validated = self._validator.validate(raw_tick)
        if validated.quality == DataQualityFlag.GAP_DETECTED:
            logger.info(
                "Gap detected for %s at epoch %d — backfill should be triggered.",
                validated.symbol,
                validated.epoch,
            )
        await self._on_tick(validated)

    # ------------------------------------------------------------------ #
    # Historical backfill
    # ------------------------------------------------------------------ #

    async def fetch_bootstrap_history(
        self, symbols: list[str], granularity_seconds: int | None = None,
        target_candle_count: int | None = None,
    ) -> dict[str, list[Candle]]:
        """
        Fetch historical candles for multiple symbols via a short-lived
        connection, independent of `run_forever()`'s long-lived streaming
        connection. Exists specifically for pipeline bootstrap: fitting a
        symbol's initial probability model needs historical candles *before*
        live tick streaming starts, but `run_forever()` connects, subscribes
        ticks, and blocks in `_listen()` — it has no hook to "just fetch
        history and return." This opens its own connection, fetches
        everything requested, and closes it; `run_forever()` opens its own
        fresh connection afterward as normal, so this has zero effect on the
        streaming connection's lifecycle, reconnect behavior, or OTP token
        usage (OTP tokens are single-use anyway, so a fresh one here is
        correct, not wasteful).

        `target_candle_count` (default: `self._hist_cfg.request_count_max`,
        i.e. a single request, the original behavior) drives pagination —
        see `_fetch_paginated_candles_for_symbol`. Different model families
        want very different amounts of history (a simple Bayesian model
        needs a few hundred candles; sequence models want as much as Deriv
        will actually give back), so the CALLER decides how much to ask
        for; this method doesn't itself know about model families.

        `fetch_historical_candles` only resolves once `_listen()` reads the
        matching "history" response off the socket and resolves the pending
        future — so a background `_listen()` task has to be running
        concurrently with the requests below, not just the bare connection.
        """
        url = await self._resolve_connect_url()
        results: dict[str, list[Candle]] = {}
        async with websockets.connect(url, ping_interval=None) as ws:
            self._ws = ws
            listen_task = asyncio.create_task(self._listen(ws))
            try:
                for symbol in symbols:
                    results[symbol] = await self._fetch_paginated_candles_for_symbol(
                        symbol, granularity_seconds, target_candle_count,
                    )
            finally:
                listen_task.cancel()
                try:
                    await listen_task
                except asyncio.CancelledError:
                    pass
        self._ws = None
        return results

    async def _fetch_paginated_candles_for_symbol(
        self, symbol: str, granularity_seconds: int | None, target_candle_count: int | None,
    ) -> list[Candle]:
        """
        Repeatedly calls `fetch_historical_candles` with progressively
        older `end_epoch` values to walk back further than one request's
        `request_count_max` cap allows, stopping at whichever comes first:

          - `target_candle_count` candles collected (default: a single
            request's worth, i.e. `request_count_max` — reproduces the
            original one-shot behavior exactly when the caller doesn't ask
            for more).
          - `lookback_days` walked back (computed from the OLDEST candle's
            own epoch each iteration, not wall-clock "now" minus days,
            since Deriv's actual returned window is what matters here).
          - Deriv returns nothing further back (fewer candles than
            requested, or an empty batch) — genuinely exhausted history for
            this symbol, not a bug; logged, not raised.

        Batches are concatenated oldest-to-newest and de-duplicated by
        epoch (defensive — overlapping `end_epoch` boundaries between
        successive requests are the most likely source of any duplicate).
        """
        target = target_candle_count or self._hist_cfg.request_count_max
        granularity = granularity_seconds or self._hist_cfg.candle_granularity_seconds
        lookback_cutoff_epoch = None  # set once we know the first batch's newest epoch

        all_candles: dict[int, Candle] = {}
        end_epoch: int | str = "latest"
        first_iteration = True

        while len(all_candles) < target:
            batch = await self.fetch_historical_candles(symbol, granularity, end_epoch)
            if not batch:
                logger.info(
                    "%s: Deriv returned no further history — stopping pagination with %d candles "
                    "(target was %d).", symbol, len(all_candles), target,
                )
                break

            if first_iteration:
                newest_epoch = max(c.epoch for c in batch)
                lookback_cutoff_epoch = newest_epoch - self._hist_cfg.lookback_days * 86400
                first_iteration = False

            for c in batch:
                all_candles[c.epoch] = c

            oldest_epoch_this_batch = min(c.epoch for c in batch)
            if lookback_cutoff_epoch is not None and oldest_epoch_this_batch <= lookback_cutoff_epoch:
                logger.info(
                    "%s: reached lookback_days=%d limit with %d candles (target was %d).",
                    symbol, self._hist_cfg.lookback_days, len(all_candles), target,
                )
                break

            if len(batch) < self._hist_cfg.request_count_max:
                # Deriv returned fewer than a full page — no more history exists
                # further back than this, regardless of target/lookback.
                logger.info(
                    "%s: received a partial page (%d < %d requested) — history exhausted with "
                    "%d candles (target was %d).",
                    symbol, len(batch), self._hist_cfg.request_count_max, len(all_candles), target,
                )
                break

            # Next page ends exactly one granularity before this batch's oldest
            # candle, so successive pages tile backward without overlap.
            end_epoch = oldest_epoch_this_batch - granularity

            if len(all_candles) < target:
                await asyncio.sleep(self._hist_cfg.pagination_delay_seconds)

        sorted_candles = sorted(all_candles.values(), key=lambda c: c.epoch)
        logger.info("%s: bootstrap history fetch complete — %d candles.", symbol, len(sorted_candles))
        return sorted_candles

    async def fetch_historical_candles(
        self, symbol: str, granularity_seconds: int | None = None, end_epoch: int | str = "latest",
    ) -> list[Candle]:
        """One-shot request/response call, independent of the streaming loop.

        `end_epoch` defaults to `"latest"` (Deriv's own sentinel for "now").
        Pagination (`fetch_paginated_history`) calls this repeatedly with
        progressively older integer `end_epoch` values to walk back further
        than one request's `request_count_max` cap allows.

        `subscribe` is sent as `1`, not `0`: as of the current
        `trading/v1/options` API, `ticks_history` rejects `subscribe: 0`
        outright (`InputValidationFailed`, `details: {"subscribe": "Not in
        enum list: 1."}` — confirmed against a real deployment's logs, not
        just documentation, which still describes the older/legacy
        behavior where 0 was accepted). Since this method is meant to be a
        single request/response, not an ongoing stream, the resulting
        subscription is immediately cancelled via `forget` right after the
        first response arrives — see `_forget_subscription_if_any` below.
        Without that cleanup, every one-shot historical fetch would leave a
        live server-side subscription running, and pagination (which calls
        this repeatedly per symbol) would quickly hit Deriv's per-connection
        subscription limit.
        """
        if self._ws is None:
            raise DerivClientError("Cannot fetch history: not connected.")

        granularity = granularity_seconds or self._hist_cfg.candle_granularity_seconds
        req_id = self._next_req_id()
        request = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": self._hist_cfg.request_count_max,
            "end": end_epoch,
            "start": 1,
            "style": "candles",
            "granularity": granularity,
            "subscribe": 1,
            "req_id": req_id,
        }
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future
        await self._ws.send(json.dumps(request))

        try:
            response = await asyncio.wait_for(
                future, timeout=self._cfg.request_timeout_seconds
            )
        finally:
            self._pending.pop(req_id, None)
        if response.get("error"):
            raise DerivClientError(f"History request failed: {response['error']}")

        await self._forget_subscription_if_any(response)

        candles_raw = response.get("candles", [])
        return [
            Candle(
                symbol=symbol,
                epoch=int(c["epoch"]),
                granularity=granularity,
                open=float(c["open"]),
                high=float(c["high"]),
                low=float(c["low"]),
                close=float(c["close"]),
            )
            for c in candles_raw
        ]

    async def _forget_subscription_if_any(self, response: dict) -> None:
        """
        Cancels the server-side subscription a `subscribe: 1` request
        creates, so a nominally "one-shot" call (`fetch_historical_candles`,
        `fetch_proposal`) doesn't leave a live stream running.
        """
        subscription_id = response.get("subscription", {}).get("id")
        if not subscription_id:
            return
        await self._forget_by_id(subscription_id)

    async def _forget_by_id(self, subscription_id: str) -> None:
        """
        Best-effort: logs and returns on any failure (timeout, error
        response) rather than raising. Shared by the normal one-shot
        cleanup path above and the late-arriving-response cleanup path in
        fetch_proposal below — in both cases, whatever this subscription
        was for has already been dealt with by the caller, so a failed
        forget shouldn't raise, it should just leak one subscription slot
        (logged, so it's visible rather than silent).
        """
        req_id = self._next_req_id()
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future
        try:
            await self._ws.send(json.dumps({"forget": subscription_id, "req_id": req_id}))
            await asyncio.wait_for(future, timeout=self._cfg.request_timeout_seconds)
        except Exception as exc:  # noqa: BLE001 — cleanup best-effort, never fails the caller
            self._pending.pop(req_id, None)
            logger.warning(
                "Failed to forget subscription %s: %s (non-fatal, but this "
                "subscription slot is now leaked until reconnect).",
                subscription_id, exc,
            )

    def _resolve_pending(self, message: dict) -> None:
        req_id = message.get("req_id")
        if req_id not in self._pending:
            return
        future = self._pending.pop(req_id)
        if future.done():
            # Already resolved (or timed out and cancelled) by the time this
            # message arrived — a genuine crash we hit in production
            # (asyncio.exceptions.InvalidStateError from calling set_result
            # on a done future). One stale/duplicate/late message must never
            # be able to take down the whole worker, so this is logged and
            # dropped rather than raised.
            logger.warning(
                "Received a message for req_id=%s after its future was already resolved — "
                "ignoring (likely a late duplicate or a client-side timeout that already fired).",
                req_id,
            )
            return
        future.set_result(message)

    # ------------------------------------------------------------------ #
    # Execution: live proposal + buy (Level 6)
    # ------------------------------------------------------------------ #

    async def fetch_proposal(
        self,
        symbol: str,
        contract_type_code: str,
        stake: float,
        duration_ticks: int,
        currency: str = "USD",
    ) -> dict:
        """
        Request a live price quote for a contract. `contract_type_code` is
        Deriv's own code (e.g. "CALL"/"PUT" for Rise/Fall) — mapping from
        our internal ContractType/direction to Deriv's codes is the
        execution engine's job (execution/engine.py), not this client's;
        this method is a thin, honest wrapper over the wire protocol only.

        Returns the raw `proposal` dict from Deriv's response (id,
        ask_price, payout, ...) — the caller decides what to do with it.
        """
        if self._ws is None:
            raise DerivClientError("Cannot fetch proposal: not connected.")

        req_id = self._next_req_id()
        request = {
            "proposal": 1,
            "amount": stake,
            "basis": "stake",
            "contract_type": contract_type_code,
            "currency": currency,
            "duration": duration_ticks,
            "duration_unit": "t",
            "underlying_symbol": symbol,
            # "subscribe": 1 is required, not optional — confirmed by live evidence:
            # sending "symbol" instead of "underlying_symbol" got an immediate, explicit
            # rejection ({"code": "InputValidationFailed", "message": "Properties not
            # allowed: symbol."}), which proves the connection/auth/transport layer is
            # fine and Deriv is actively validating this request. But even with the
            # correct field name, omitting "subscribe" produced ZERO response — no
            # error, no proposal, ever — across two full deployment logs. That matches
            # ticks_history's confirmed behavior on this same API (see
            # fetch_historical_candles below): the request is schema-valid but Deriv
            # never generates a priced response without an explicit subscription.
            "subscribe": 1,
            "req_id": req_id,
        }
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future
        await self._ws.send(json.dumps(request))

        try:
            # asyncio.shield() matters here, not just style: plain `wait_for(future, ...)`
            # CANCELS `future` itself the moment the timeout fires. That's fine for a
            # request that's genuinely dead — but confirmed in production
            # (logs_1784974444569.csv), Deriv frequently responds to `proposal`
            # requests right around/after request_timeout_seconds; the request is
            # very often not dead, just slow. If the timeout cancels `future`, the
            # late response has nothing left to resolve — `_resolve_pending` finds
            # `future.done()` already True and silently drops the message — and the
            # subscription that response just created is never forgotten. It then
            # sits there forever, and every subsequent fetch_proposal for this same
            # symbol fails immediately with {"code": "AlreadySubscribed", "message":
            # "You are already subscribed to proposal."} for the rest of this
            # connection's life. shield() lets the outer wait time out without
            # touching the inner `future`, so it's still alive in `self._pending`
            # to catch a late response below.
            response = await asyncio.wait_for(
                asyncio.shield(future), timeout=self._cfg.request_timeout_seconds
            )
        except asyncio.TimeoutError:
            # Re-raise asyncio.TimeoutError itself (bare `raise`, not a different
            # exception type) — execution/engine.py has a dedicated
            # `except asyncio.TimeoutError:` branch that depends on catching this
            # exact type to report a clean, specific reason (and, for buy(), a
            # genuinely important safety note that the order may have gone through
            # anyway). Raising a different type here (e.g. DerivClientError) would
            # silently fall through to engine.py's generic `except Exception`
            # branch instead and lose that specific handling — this happened in a
            # previous version of this fix and needs to not happen again.
            future.add_done_callback(functools.partial(self._on_late_proposal_response, symbol))
            raise
        self._pending.pop(req_id, None)
        if response.get("error"):
            raise DerivClientError(f"Proposal request failed: {response['error']}")

        # subscribe:1 leaves a live server-side subscription running (continuous
        # re-pricing pushes) — the caller here only wants a single quote, so cancel
        # it immediately, same cleanup fetch_historical_candles does for its own
        # one-shot subscribe:1 requests. Left running, every proposal fetch (one per
        # symbol per cycle, indefinitely) would leak a subscription, which would
        # eventually exhaust Deriv's per-connection subscription limit and could
        # itself start producing the exact same silent-hang symptom again later.
        await self._forget_subscription_if_any(response)

        return response["proposal"]

    def _on_late_proposal_response(self, symbol: str, future: asyncio.Future) -> None:
        """
        Done-callback attached only when a fetch_proposal call has already
        timed out locally. If Deriv's response arrives after all, this
        forgets whatever subscription it created — otherwise that
        subscription is orphaned forever and every later fetch_proposal for
        `symbol` fails immediately with AlreadySubscribed (see fetch_proposal
        above for the full explanation). This runs as an asyncio done-callback,
        so it must never raise or await directly.
        """
        if future.cancelled() or future.exception() is not None:
            return
        response = future.result()
        if response.get("error"):
            return  # Deriv rejected it — nothing was subscribed, nothing to forget.
        subscription_id = response.get("subscription", {}).get("id")
        if not subscription_id:
            return
        logger.warning(
            "%s: a fetch_proposal call that already timed out locally just "
            "succeeded server-side (subscription id=%s) — forgetting it now so "
            "it doesn't permanently block future proposal requests for this "
            "symbol with AlreadySubscribed.",
            symbol, subscription_id,
        )
        asyncio.get_event_loop().create_task(self._forget_by_id(subscription_id))

    async def buy_direct(
        self,
        symbol: str,
        contract_type_code: str,
        stake: float,
        duration_ticks: int,
        currency: str = "USD",
    ) -> dict:
        """
        Fetches a quote and buys it in ONE request/response, instead of
        fetch_proposal() then buy() as two sequential round-trips. This is
        now the default live execution path (see execution/engine.py) —
        adopted specifically to cut the number of round-trips exposed to
        Deriv's response-timing behavior in half, after production
        evidence showed proposal timeouts recurring at BOTH 15s and 25s
        request_timeout_seconds configurations (logs_1785063927056.csv,
        logs_1785066902526.csv) — a pattern that tracks our OWN configured
        timeout rather than a fixed Deriv-side latency, which is the real
        open question this method's logging below is designed to resolve,
        not just work around.

        Trade-off, stated plainly: the two-step flow let execution/engine.py
        check the live proposal's actual price/payout against the EV/Risk
        decision's assumptions BEFORE committing to a real purchase, and
        abort if it had drifted too far. This one-step call has no such
        checkpoint — you commit to whatever price Deriv fills at. Adopted
        as the default per an explicit, informed choice (this trade-off was
        raised and accepted) — not a silent removal of that safety rail.

        Returns the raw `buy` dict from Deriv's response (contract_id,
        buy_price, payout, ...).
        """
        if self._ws is None:
            raise DerivClientError("Cannot buy: not connected.")

        req_id = self._next_req_id()
        request = {
            "buy": "1",
            "price": stake,
            "parameters": {
                "amount": stake,
                "basis": "stake",
                "contract_type": contract_type_code,
                "currency": currency,
                "duration": duration_ticks,
                "duration_unit": "t",
                "underlying_symbol": symbol,
            },
            "req_id": req_id,
        }
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future

        # Deliberately logged around the send() call itself, not just at entry to
        # this method — see this method's docstring. If a future log shows a large
        # gap between "opportunity approved" and this line, the delay is in OUR
        # OWN pipeline (something blocking the event loop before we ever get to
        # sending), not in Deriv's response time. If instead the gap between this
        # line and a timeout/response is what's large, the delay is genuinely on
        # Deriv's side. Today's evidence hasn't yet distinguished these.
        send_started_at = time.monotonic()
        logger.info("%s: sending one-step buy_direct request now (req_id=%d).", symbol, req_id)
        await self._ws.send(json.dumps(request))
        logger.info(
            "%s: buy_direct request sent (took %.3fs to hand off to the socket).",
            symbol, time.monotonic() - send_started_at,
        )

        try:
            # Same shield() reasoning as fetch_proposal/buy above: a client-side
            # timeout must not cancel `future` outright, or a late-but-real
            # response has nothing left to resolve and any contract it opened
            # would go completely unnoticed — the highest-stakes version of that
            # risk in this file, since this IS the real trade now, not a proposal.
            response = await asyncio.wait_for(
                asyncio.shield(future), timeout=self._cfg.request_timeout_seconds
            )
        except asyncio.TimeoutError:
            future.add_done_callback(functools.partial(self._on_late_buy_response, f"buy_direct:{symbol}"))
            raise
        self._pending.pop(req_id, None)
        if response.get("error"):
            raise DerivClientError(f"Direct buy request failed: {response['error']}")
        return response["buy"]

    async def buy(self, proposal_id: str, price: float) -> dict:
        """
        Execute a buy against a previously-fetched proposal. Returns the
        raw `buy` dict from Deriv's response (contract_id, buy_price,
        payout, ...).

        This is the one method in the entire codebase that spends real
        money when connected to a real (non-demo) account — it does
        exactly what it's told and nothing more; every safety decision
        (should we buy at all, at what stake, is this proposal stale)
        belongs upstream in execution/engine.py, not here.
        """
        if self._ws is None:
            raise DerivClientError("Cannot buy: not connected.")

        req_id = self._next_req_id()
        request = {"buy": proposal_id, "price": price, "req_id": req_id}
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future
        await self._ws.send(json.dumps(request))

        try:
            # Same reasoning as fetch_proposal above, and higher-stakes here:
            # execution/engine.py's own comment already acknowledges a buy
            # timeout doesn't mean the buy failed — "the contract may still
            # have been bought on Deriv's side even though we never got
            # confirmation — check the account before retrying." Without
            # shield(), a late confirmation is silently dropped and that
            # check-the-account step is the ONLY way to ever find out.
            # shield() keeps this future alive so a late response can still be
            # captured and logged automatically instead of relying solely on
            # someone remembering to go look.
            response = await asyncio.wait_for(
                asyncio.shield(future), timeout=self._cfg.request_timeout_seconds
            )
        except asyncio.TimeoutError:
            future.add_done_callback(functools.partial(self._on_late_buy_response, proposal_id))
            raise
        self._pending.pop(req_id, None)
        if response.get("error"):
            raise DerivClientError(f"Buy request failed: {response['error']}")
        return response["buy"]

    def _on_late_buy_response(self, context: str, future: asyncio.Future) -> None:
        """
        Done-callback attached only when a buy() or buy_direct() call has
        already timed out locally. If Deriv's response arrives after all,
        this logs it — contract_id and all — since the caller already gave
        up and reported an error, so this is the only remaining way to
        surface that a real contract may now be open. `context` is either
        a real proposal_id (from buy()) or a "buy_direct:{symbol}" label
        (from buy_direct(), which has no proposal_id concept) — either way
        it's just for identifying which request this was in the logs. This
        runs as an asyncio done-callback, so it must never raise.
        """
        if future.cancelled() or future.exception() is not None:
            return
        response = future.result()
        if response.get("error"):
            logger.warning(
                "A buy request (%s) that already timed out locally "
                "was actually rejected server-side: %s — no contract was opened.",
                context, response["error"],
            )
            return
        buy_result = response.get("buy", {})
        logger.warning(
            "IMPORTANT: a buy request (%s) that already timed out "
            "locally actually SUCCEEDED server-side — contract_id=%s, "
            "buy_price=%s, payout=%s. This contract is real and open; %s",
            context,
            buy_result.get("contract_id"),
            buy_result.get("buy_price"),
            buy_result.get("payout"),
            "handing off to on_late_contract for tracking." if self._on_late_contract is not None
            else "NO on_late_contract callback is registered — this contract is untracked "
                 "(no settlement polling, no equity/risk accounting, no post-trade record) "
                 "until someone manually reconciles it against the account.",
        )
        if self._on_late_contract is not None:
            # context is "buy_direct:{symbol}" from buy_direct(), or a real
            # proposal_id from buy() (which has no bare symbol available here) —
            # callers should treat `context` as an opaque identifier, not assume
            # its shape. Scheduled as a task, same as _on_late_proposal_response
            # above: this runs inside an asyncio done-callback, which must never
            # raise or await directly.
            asyncio.get_event_loop().create_task(self._on_late_contract(context, buy_result))

    async def check_contract_status(self, contract_id: str) -> dict:
        """
        One-shot poll of a previously-bought contract's real status via
        Deriv's `proposal_open_contract` call. Deliberately `subscribe: 0`
        — a real push subscription would need its own long-lived
        per-contract stream management (and cleanup on settlement, and
        reconnect handling), which is more machinery than a poll-on-some-
        cadence caller (the orchestrator, once per candle) actually needs.
        If push-based settlement latency ever matters enough to justify
        that complexity, this is the method to extend, not replace —
        callers only depend on "give me the current status right now".

        Returns the raw `proposal_open_contract` dict from Deriv's
        response (is_sold, profit, payout, buy_price, sell_price,
        status, ...) unmodified — interpreting what "settled" means is
        the caller's job (see execution/outcome_tracker.py), not this
        client's.
        """
        if self._ws is None:
            raise DerivClientError("Cannot check contract status: not connected.")

        req_id = self._next_req_id()
        request = {
            "proposal_open_contract": 1,
            "contract_id": contract_id,
            "subscribe": 0,
            "req_id": req_id,
        }
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future
        await self._ws.send(json.dumps(request))

        try:
            response = await asyncio.wait_for(
                future, timeout=self._cfg.request_timeout_seconds
            )
        finally:
            self._pending.pop(req_id, None)
        if response.get("error"):
            raise DerivClientError(f"Contract status request failed: {response['error']}")
        return response["proposal_open_contract"]

    # ------------------------------------------------------------------ #
    # Reconnect logic
    # ------------------------------------------------------------------ #

    async def _handle_reconnect(self) -> bool:
        if (
            self._cfg.max_reconnect_attempts is not None
            and self._reconnect_attempt >= self._cfg.max_reconnect_attempts
        ):
            await self._emit_event(
                "reconnect_failed", detail="max_reconnect_attempts exhausted"
            )
            return False

        backoff = min(
            self._cfg.reconnect_initial_backoff_seconds
            * (self._cfg.reconnect_backoff_multiplier ** self._reconnect_attempt),
            self._cfg.reconnect_max_backoff_seconds,
        )
        self._reconnect_attempt += 1
        detail = f"attempt {self._reconnect_attempt}, waiting {backoff:.1f}s"
        if self._cfg.is_authenticated_mode:
            detail += " (will request a fresh OTP before reconnecting)"
        await self._emit_event("reconnecting", detail=detail)
        await asyncio.sleep(backoff)
        return True

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _next_req_id(self) -> int:
        self._req_id_counter += 1
        return self._req_id_counter

    async def _emit_event(self, event: str, detail: str) -> None:
        logger.info("Deriv client event: %s — %s", event, detail)
        if self._on_connection_event is not None:
            await self._on_connection_event(
                ConnectionEvent(
                    event=event,
                    attempt=self._reconnect_attempt,
                    detail=detail,
                    occurred_at=datetime.now(timezone.utc),
                )
            )
