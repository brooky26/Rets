import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from configs.schema import DataIntegrityConfig, DerivConnectionConfig, HistoricalDataConfig
from data.deriv_client import DerivClientError, DerivOTPBootstrap, DerivWebSocketClient
from data.integrity import IntegrityValidator
from data.types import Tick


def make_authed_config(**overrides) -> DerivConnectionConfig:
    defaults = dict(
        app_id="12345",
        api_token="test_token_abc",
        account_id="DOT90004580",
    )
    defaults.update(overrides)
    return DerivConnectionConfig(**defaults)


def make_public_config(**overrides) -> DerivConnectionConfig:
    defaults = dict(app_id="12345")
    defaults.update(overrides)
    return DerivConnectionConfig(**defaults)


def _mock_session_for(status: int, payload: dict):
    """
    Build a MagicMock standing in for aiohttp.ClientSession such that:
        async with aiohttp.ClientSession(...) as session:
            async with session.post(...) as resp:
                await resp.json()
    resolves to the given status/payload, and records the call kwargs
    for header assertions.
    """
    mock_response = MagicMock()
    mock_response.status = status
    mock_response.json = AsyncMock(return_value=payload)

    mock_post_cm = MagicMock()
    mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post_cm.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post_cm)

    mock_session_cm = MagicMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_cm.__aexit__ = AsyncMock(return_value=False)

    return mock_session_cm, mock_session


@pytest.mark.asyncio
async def test_otp_bootstrap_returns_url_on_success():
    config = make_authed_config()
    bootstrap = DerivOTPBootstrap(config)
    expected_url = "wss://api.derivws.com/trading/v1/options/ws/demo?otp=abc123xyz789"

    session_cm, _ = _mock_session_for(200, {"data": {"url": expected_url}})
    with patch("aiohttp.ClientSession", return_value=session_cm):
        url = await bootstrap.fetch_authenticated_ws_url()

    assert url == expected_url


@pytest.mark.asyncio
async def test_otp_bootstrap_sends_correct_headers():
    config = make_authed_config()
    bootstrap = DerivOTPBootstrap(config)

    session_cm, mock_session = _mock_session_for(
        200, {"data": {"url": "wss://x/ws/demo?otp=y"}}
    )
    with patch("aiohttp.ClientSession", return_value=session_cm):
        await bootstrap.fetch_authenticated_ws_url()

    called_args, called_kwargs = mock_session.post.call_args
    assert called_args[0] == (
        "https://api.derivws.com/trading/v1/options/accounts/DOT90004580/otp"
    )
    assert called_kwargs["headers"]["Deriv-App-ID"] == "12345"
    assert called_kwargs["headers"]["Authorization"] == "Bearer test_token_abc"


@pytest.mark.asyncio
async def test_otp_bootstrap_raises_on_401():
    config = make_authed_config()
    bootstrap = DerivOTPBootstrap(config)

    session_cm, _ = _mock_session_for(
        401,
        {"errors": [{"status": 401, "code": "Unauthorized", "message": "Invalid token"}]},
    )
    with patch("aiohttp.ClientSession", return_value=session_cm):
        with pytest.raises(DerivClientError, match="OTP request failed"):
            await bootstrap.fetch_authenticated_ws_url()


@pytest.mark.asyncio
async def test_otp_bootstrap_raises_on_missing_url_in_response():
    config = make_authed_config()
    bootstrap = DerivOTPBootstrap(config)

    session_cm, _ = _mock_session_for(200, {"data": {}})  # malformed: no url
    with patch("aiohttp.ClientSession", return_value=session_cm):
        with pytest.raises(DerivClientError, match="missing data.url"):
            await bootstrap.fetch_authenticated_ws_url()


@pytest.mark.asyncio
async def test_otp_bootstrap_raises_when_not_authenticated_mode():
    config = make_public_config()  # no api_token/account_id
    bootstrap = DerivOTPBootstrap(config)
    with pytest.raises(DerivClientError, match="without api_token/account_id"):
        await bootstrap.fetch_authenticated_ws_url()


def test_public_config_requires_no_account_id():
    # Should not raise — public mode has no account_id requirement.
    config = make_public_config()
    assert config.is_authenticated_mode is False
    assert config.account_id is None


def test_authenticated_config_allows_missing_account_id():
    """account_id is no longer required at construction time — it's optional
    and auto-resolved later from api_token via
    `data.deriv_client.ensure_account_id`/`resolve_account_id` (see
    test_deriv_account_resolution.py). This only checks construction
    doesn't raise; the resolution behavior itself is tested separately."""
    config = DerivConnectionConfig(app_id="12345", api_token="sometoken", account_id=None)
    assert config.account_id is None
    assert config.is_authenticated_mode is True


def test_default_public_url_used_when_unauthenticated():
    config = make_public_config()
    assert config.ws_public_url == "wss://api.derivws.com/trading/v1/options/ws/public"


# --------------------------------------------------------------------- #
# Regression: _listen must not crash on an error-shaped "tick" message.
#
# Deriv responds to a rejected subscription (e.g. invalid symbol) with a
# message that still carries msg_type == "tick" but no "tick" payload —
# just an "error" object. A prior version of _listen checked msg_type
# before checking for "error", so it dereferenced a nonexistent "tick"
# key and crashed the whole client on the first bad symbol. This test
# pins the fix: error-shaped messages must be logged and skipped, not
# raised, and must not prevent later legitimate tick messages in the
# same stream from being processed.
# --------------------------------------------------------------------- #


class _FakeWebSocket:
    """Minimal async-iterable standing in for a websockets connection,
    yielding pre-baked raw JSON messages one at a time."""

    def __init__(self, raw_messages: list[str]) -> None:
        self._messages = raw_messages

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for raw in self._messages:
            yield raw


def _make_client(on_tick=None) -> DerivWebSocketClient:
    conn_cfg = make_public_config()
    hist_cfg = HistoricalDataConfig()
    validator = IntegrityValidator(DataIntegrityConfig())
    return DerivWebSocketClient(
        connection_config=conn_cfg,
        historical_config=hist_cfg,
        integrity_validator=validator,
        on_tick=on_tick or AsyncMock(),
    )


@pytest.mark.asyncio
async def test_listen_does_not_crash_on_error_shaped_tick_message():
    error_message = json.dumps(
        {
            "msg_type": "tick",
            "echo_req": {"ticks": "invalidSymbol", "subscribe": 1},
            "error": {"code": "InvalidSymbol", "message": "Invalid symbol."},
        }
    )
    ws = _FakeWebSocket([error_message])
    client = _make_client()

    # Must complete without raising (previously: KeyError on message["tick"]).
    await client._listen(ws)


@pytest.mark.asyncio
async def test_listen_processes_valid_ticks_after_an_error_message():
    error_message = json.dumps(
        {
            "msg_type": "tick",
            "echo_req": {"ticks": "invalidSymbol", "subscribe": 1},
            "error": {"code": "InvalidSymbol", "message": "Invalid symbol."},
        }
    )
    valid_tick_message = json.dumps(
        {
            "msg_type": "tick",
            "tick": {"symbol": "stpRNG", "epoch": 1_700_000_000, "quote": 1234.56},
        }
    )
    ws = _FakeWebSocket([error_message, valid_tick_message])
    received_ticks = []

    async def on_tick(tick: Tick) -> None:
        received_ticks.append(tick)

    client = _make_client(on_tick=on_tick)
    await client._listen(ws)

    assert len(received_ticks) == 1
    assert received_ticks[0].symbol == "stpRNG"
    assert received_ticks[0].quote == 1234.56


@pytest.mark.asyncio
async def test_handle_tick_message_ignores_tick_typed_message_with_no_payload():
    """Defense-in-depth: even if a tick-typed, error-free message somehow
    arrives with no "tick" key, _handle_tick_message must not crash."""
    client = _make_client()
    await client._handle_tick_message({"msg_type": "tick"})
