"""
Tests for `data.deriv_client.resolve_account_id` / `ensure_account_id` —
auto-discovering a Deriv account_id from just an api_token via
GET /trading/v1/options/accounts, so operators don't have to look it up
by hand. Mirrors the mocking pattern already used in test_deriv_client.py
for the (POST) OTP endpoint, adapted for a GET request.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from configs.schema import DerivConnectionConfig
from data.deriv_client import DerivClientError, ensure_account_id, resolve_account_id


def make_config_no_account_id(**overrides) -> DerivConnectionConfig:
    defaults = dict(app_id="12345", api_token="test_token_abc", ws_account_type="demo")
    defaults.update(overrides)
    return DerivConnectionConfig(**defaults)


def _mock_session_for_get(status: int, payload: dict):
    mock_response = MagicMock()
    mock_response.status = status
    mock_response.json = AsyncMock(return_value=payload)

    mock_get_cm = MagicMock()
    mock_get_cm.__aenter__ = AsyncMock(return_value=mock_response)
    mock_get_cm.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_get_cm)

    mock_session_cm = MagicMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_cm.__aexit__ = AsyncMock(return_value=False)

    return mock_session_cm, mock_session


@pytest.mark.asyncio
async def test_resolve_account_id_picks_the_single_active_matching_account():
    config = make_config_no_account_id()
    payload = {
        "data": [
            {"account_id": "VRTC1234567", "account_type": "demo", "status": "active"},
            {"account_id": "CR9000001", "account_type": "real", "status": "active"},
        ]
    }
    session_cm, _ = _mock_session_for_get(200, payload)
    with patch("aiohttp.ClientSession", return_value=session_cm):
        account_id = await resolve_account_id(config)

    assert account_id == "VRTC1234567"


@pytest.mark.asyncio
async def test_resolve_account_id_sends_correct_headers_and_url():
    config = make_config_no_account_id()
    session_cm, mock_session = _mock_session_for_get(
        200, {"data": [{"account_id": "VRTC1", "account_type": "demo", "status": "active"}]}
    )
    with patch("aiohttp.ClientSession", return_value=session_cm):
        await resolve_account_id(config)

    called_args, called_kwargs = mock_session.get.call_args
    assert called_args[0] == "https://api.derivws.com/trading/v1/options/accounts"
    assert called_kwargs["headers"]["Deriv-App-ID"] == "12345"
    assert called_kwargs["headers"]["Authorization"] == "Bearer test_token_abc"


@pytest.mark.asyncio
async def test_resolve_account_id_raises_when_no_active_match():
    config = make_config_no_account_id()
    payload = {"data": [{"account_id": "CR9000001", "account_type": "real", "status": "active"}]}
    session_cm, _ = _mock_session_for_get(200, payload)
    with patch("aiohttp.ClientSession", return_value=session_cm):
        with pytest.raises(DerivClientError, match="No active 'demo' account"):
            await resolve_account_id(config)


@pytest.mark.asyncio
async def test_resolve_account_id_raises_when_ambiguous():
    config = make_config_no_account_id()
    payload = {
        "data": [
            {"account_id": "VRTC1", "account_type": "demo", "status": "active"},
            {"account_id": "VRTC2", "account_type": "demo", "status": "active"},
        ]
    }
    session_cm, _ = _mock_session_for_get(200, payload)
    with patch("aiohttp.ClientSession", return_value=session_cm):
        with pytest.raises(DerivClientError, match="Multiple active 'demo' accounts"):
            await resolve_account_id(config)


@pytest.mark.asyncio
async def test_resolve_account_id_ignores_inactive_accounts():
    config = make_config_no_account_id()
    payload = {
        "data": [
            {"account_id": "VRTC_OLD", "account_type": "demo", "status": "disabled"},
            {"account_id": "VRTC_NEW", "account_type": "demo", "status": "active"},
        ]
    }
    session_cm, _ = _mock_session_for_get(200, payload)
    with patch("aiohttp.ClientSession", return_value=session_cm):
        account_id = await resolve_account_id(config)
    assert account_id == "VRTC_NEW"


@pytest.mark.asyncio
async def test_resolve_account_id_raises_on_error_status():
    config = make_config_no_account_id()
    session_cm, _ = _mock_session_for_get(
        401, {"errors": [{"status": 401, "code": "Unauthorized", "message": "bad token"}]}
    )
    with patch("aiohttp.ClientSession", return_value=session_cm):
        with pytest.raises(DerivClientError, match="Account lookup failed"):
            await resolve_account_id(config)


@pytest.mark.asyncio
async def test_resolve_account_id_requires_api_token():
    config = DerivConnectionConfig(app_id="12345")  # no api_token at all
    with pytest.raises(DerivClientError, match="without api_token"):
        await resolve_account_id(config)


@pytest.mark.asyncio
async def test_ensure_account_id_skips_lookup_when_already_set():
    config = DerivConnectionConfig(app_id="12345", api_token="tok", account_id="EXPLICIT1")
    with patch("aiohttp.ClientSession") as mock_cls:
        result = await ensure_account_id(config)
    mock_cls.assert_not_called()
    assert result.account_id == "EXPLICIT1"
    assert result is config


@pytest.mark.asyncio
async def test_ensure_account_id_skips_lookup_in_public_mode():
    config = DerivConnectionConfig(app_id="12345")  # no api_token, public-data-only
    with patch("aiohttp.ClientSession") as mock_cls:
        result = await ensure_account_id(config)
    mock_cls.assert_not_called()
    assert result.account_id is None
    assert result is config


@pytest.mark.asyncio
async def test_ensure_account_id_resolves_and_fills_in_when_missing():
    config = make_config_no_account_id()
    payload = {"data": [{"account_id": "VRTC777", "account_type": "demo", "status": "active"}]}
    session_cm, _ = _mock_session_for_get(200, payload)
    with patch("aiohttp.ClientSession", return_value=session_cm):
        result = await ensure_account_id(config)

    assert result.account_id == "VRTC777"
    # Original config object is untouched (model_copy, not mutation).
    assert config.account_id is None


def test_config_no_longer_requires_account_id_with_api_token():
    """Constructing a config with api_token but no account_id must succeed now
    (auto-resolution happens later, async, at startup) — this used to raise."""
    config = DerivConnectionConfig(app_id="12345", api_token="tok")
    assert config.account_id is None
    assert config.is_authenticated_mode is True
