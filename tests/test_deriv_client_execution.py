import asyncio
import json

import pytest

from configs.schema import DataIntegrityConfig, DerivConnectionConfig, HistoricalDataConfig
from data.deriv_client import DerivClientError, DerivWebSocketClient
from data.integrity import IntegrityValidator


class FakeWebSocket:
    """
    Records sent messages and lets the test manually deliver a response
    by calling client._resolve_pending(...), the same way a real message
    arriving via _listen() would — tests the request-building/response-
    parsing logic without needing a real socket.
    """

    def __init__(self):
        self.sent_messages = []

    async def send(self, raw: str) -> None:
        self.sent_messages.append(json.loads(raw))


async def make_client_with_fake_ws():
    connection_config = DerivConnectionConfig(app_id="12345")
    historical_config = HistoricalDataConfig()
    validator = IntegrityValidator(DataIntegrityConfig())

    async def on_tick(tick):
        pass

    client = DerivWebSocketClient(
        connection_config=connection_config,
        historical_config=historical_config,
        integrity_validator=validator,
        on_tick=on_tick,
    )
    client._ws = FakeWebSocket()
    return client


@pytest.mark.asyncio
async def test_fetch_proposal_sends_correct_request_shape():
    client = await make_client_with_fake_ws()

    async def deliver_after_send():
        await asyncio.sleep(0)
        req_id = client._ws.sent_messages[0]["req_id"]
        client._resolve_pending({
            "msg_type": "proposal", "req_id": req_id,
            "proposal": {"id": "prop1", "ask_price": 10.0, "payout": 19.0},
        })

    result, _ = await asyncio.gather(
        client.fetch_proposal("STPRNG100", "CALL", stake=10.0, duration_ticks=5),
        deliver_after_send(),
    )

    sent = client._ws.sent_messages[0]
    assert sent["proposal"] == 1
    assert sent["amount"] == 10.0
    assert sent["basis"] == "stake"
    assert sent["contract_type"] == "CALL"
    assert sent["duration"] == 5
    assert sent["duration_unit"] == "t"
    assert sent["symbol"] == "STPRNG100"
    assert result == {"id": "prop1", "ask_price": 10.0, "payout": 19.0}


@pytest.mark.asyncio
async def test_fetch_proposal_raises_on_error_response():
    client = await make_client_with_fake_ws()

    async def deliver_error():
        await asyncio.sleep(0)
        req_id = client._ws.sent_messages[0]["req_id"]
        client._resolve_pending({
            "msg_type": "proposal", "req_id": req_id,
            "error": {"code": "InvalidSymbol", "message": "bad symbol"},
        })

    with pytest.raises(DerivClientError, match="Proposal request failed"):
        await asyncio.gather(
            client.fetch_proposal("BADSYM", "CALL", stake=10.0, duration_ticks=5),
            deliver_error(),
        )


@pytest.mark.asyncio
async def test_fetch_proposal_raises_when_not_connected():
    client = await make_client_with_fake_ws()
    client._ws = None
    with pytest.raises(DerivClientError, match="not connected"):
        await client.fetch_proposal("STPRNG100", "CALL", stake=10.0, duration_ticks=5)


@pytest.mark.asyncio
async def test_buy_sends_correct_request_shape():
    client = await make_client_with_fake_ws()

    async def deliver_after_send():
        await asyncio.sleep(0)
        req_id = client._ws.sent_messages[0]["req_id"]
        client._resolve_pending({
            "msg_type": "buy", "req_id": req_id,
            "buy": {"contract_id": "c1", "buy_price": 10.0, "payout": 19.0},
        })

    result, _ = await asyncio.gather(
        client.buy("prop1", 10.0),
        deliver_after_send(),
    )

    sent = client._ws.sent_messages[0]
    assert sent["buy"] == "prop1"
    assert sent["price"] == 10.0
    assert result == {"contract_id": "c1", "buy_price": 10.0, "payout": 19.0}


@pytest.mark.asyncio
async def test_buy_raises_on_error_response():
    client = await make_client_with_fake_ws()

    async def deliver_error():
        await asyncio.sleep(0)
        req_id = client._ws.sent_messages[0]["req_id"]
        client._resolve_pending({
            "msg_type": "buy", "req_id": req_id,
            "error": {"code": "InsufficientBalance", "message": "not enough funds"},
        })

    with pytest.raises(DerivClientError, match="Buy request failed"):
        await asyncio.gather(
            client.buy("prop1", 10.0),
            deliver_error(),
        )


@pytest.mark.asyncio
async def test_buy_raises_when_not_connected():
    client = await make_client_with_fake_ws()
    client._ws = None
    with pytest.raises(DerivClientError, match="not connected"):
        await client.buy("prop1", 10.0)


@pytest.mark.asyncio
async def test_check_contract_status_sends_correct_request_shape():
    client = await make_client_with_fake_ws()

    async def deliver_after_send():
        await asyncio.sleep(0)
        req_id = client._ws.sent_messages[0]["req_id"]
        client._resolve_pending({
            "msg_type": "proposal_open_contract", "req_id": req_id,
            "proposal_open_contract": {
                "contract_id": "c789", "is_sold": 1, "profit": 9.0,
                "payout": 19.0, "buy_price": 10.0, "sell_price": 19.0,
            },
        })

    result, _ = await asyncio.gather(
        client.check_contract_status("c789"),
        deliver_after_send(),
    )

    sent = client._ws.sent_messages[0]
    assert sent["proposal_open_contract"] == 1
    assert sent["contract_id"] == "c789"
    assert sent["subscribe"] == 0
    assert result == {
        "contract_id": "c789", "is_sold": 1, "profit": 9.0,
        "payout": 19.0, "buy_price": 10.0, "sell_price": 19.0,
    }


@pytest.mark.asyncio
async def test_check_contract_status_raises_on_error_response():
    client = await make_client_with_fake_ws()

    async def deliver_error():
        await asyncio.sleep(0)
        req_id = client._ws.sent_messages[0]["req_id"]
        client._resolve_pending({
            "msg_type": "proposal_open_contract", "req_id": req_id,
            "error": {"code": "InvalidContractId", "message": "unknown contract"},
        })

    with pytest.raises(DerivClientError, match="Contract status request failed"):
        await asyncio.gather(
            client.check_contract_status("bad-id"),
            deliver_error(),
        )


@pytest.mark.asyncio
async def test_check_contract_status_raises_when_not_connected():
    client = await make_client_with_fake_ws()
    client._ws = None
    with pytest.raises(DerivClientError, match="not connected"):
        await client.check_contract_status("c789")


async def _wait_for_sent_message(client, index: int, max_iterations: int = 100):
    """Poll until client._ws.sent_messages has at least `index + 1` entries.

    fetch_proposal()/buy() await their response via asyncio.wait_for(future,
    ...), which adds an extra event-loop hop beyond the future's own
    resolution before control actually returns to the caller and the next
    request gets sent. A single `await asyncio.sleep(0)` isn't reliably
    enough turns of the loop for that — this polls instead of assuming a
    fixed number of yields, so the test is robust to that internal
    implementation detail rather than coupled to it.
    """
    for _ in range(max_iterations):
        if len(client._ws.sent_messages) > index:
            return
        await asyncio.sleep(0)
    raise AssertionError(
        f"Timed out waiting for sent_messages[{index}]; only "
        f"{len(client._ws.sent_messages)} message(s) sent so far."
    )


@pytest.mark.asyncio
async def test_proposal_and_buy_use_distinct_req_ids():
    client = await make_client_with_fake_ws()

    async def deliver_both():
        await _wait_for_sent_message(client, 0)
        req_id_1 = client._ws.sent_messages[0]["req_id"]
        client._resolve_pending({
            "msg_type": "proposal", "req_id": req_id_1,
            "proposal": {"id": "prop1", "ask_price": 10.0, "payout": 19.0},
        })
        await _wait_for_sent_message(client, 1)
        req_id_2 = client._ws.sent_messages[1]["req_id"]
        client._resolve_pending({
            "msg_type": "buy", "req_id": req_id_2,
            "buy": {"contract_id": "c1", "buy_price": 10.0, "payout": 19.0},
        })

    async def do_both():
        await client.fetch_proposal("STPRNG100", "CALL", stake=10.0, duration_ticks=5)
        await client.buy("prop1", 10.0)

    await asyncio.gather(do_both(), deliver_both())
    req_ids = [m["req_id"] for m in client._ws.sent_messages]
    assert len(set(req_ids)) == len(req_ids)
