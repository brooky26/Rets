import pytest

from execution.outcome_tracker import ContractOutcomeTracker


class FakeBrokerClient:
    def __init__(self, status_response: dict | None = None, raise_on_status=False):
        self._status_response = status_response
        self._raise_on_status = raise_on_status
        self.status_calls = []

    async def fetch_proposal(self, *args, **kwargs):  # pragma: no cover — unused here
        raise NotImplementedError

    async def buy(self, *args, **kwargs):  # pragma: no cover — unused here
        raise NotImplementedError

    async def check_contract_status(self, contract_id: str) -> dict:
        self.status_calls.append(contract_id)
        if self._raise_on_status:
            raise RuntimeError("simulated network failure")
        return self._status_response


@pytest.mark.asyncio
async def test_poll_returns_not_sold_when_contract_still_open():
    broker = FakeBrokerClient(status_response={
        "contract_id": "c1", "is_sold": 0, "payout": 19.0, "buy_price": 10.0,
    })
    tracker = ContractOutcomeTracker(broker)
    outcome = await tracker.poll("c1")
    assert outcome.is_sold is False
    assert outcome.pnl == 0.0
    assert outcome.sell_price is None
    assert broker.status_calls == ["c1"]


@pytest.mark.asyncio
async def test_poll_returns_real_pnl_on_a_win():
    broker = FakeBrokerClient(status_response={
        "contract_id": "c1", "is_sold": 1, "profit": 9.0,
        "payout": 19.0, "buy_price": 10.0, "sell_price": 19.0,
    })
    tracker = ContractOutcomeTracker(broker)
    outcome = await tracker.poll("c1")
    assert outcome.is_sold is True
    assert outcome.pnl == pytest.approx(9.0)
    assert outcome.payout == pytest.approx(19.0)
    assert outcome.sell_price == pytest.approx(19.0)


@pytest.mark.asyncio
async def test_poll_returns_real_pnl_on_a_loss():
    broker = FakeBrokerClient(status_response={
        "contract_id": "c1", "is_sold": 1, "profit": -10.0,
        "payout": 19.0, "buy_price": 10.0, "sell_price": 0.0,
    })
    tracker = ContractOutcomeTracker(broker)
    outcome = await tracker.poll("c1")
    assert outcome.is_sold is True
    assert outcome.pnl == pytest.approx(-10.0)


@pytest.mark.asyncio
async def test_poll_never_recomputes_win_loss_from_price_fields():
    """The tracker must relay Deriv's own `profit` field, not derive its
    own win/loss from buy_price/sell_price/payout — Deriv is the sole
    authority on live outcomes."""
    broker = FakeBrokerClient(status_response={
        "contract_id": "c1", "is_sold": 1, "profit": -3.5,
        "payout": 19.0, "buy_price": 10.0, "sell_price": 6.5,
    })
    tracker = ContractOutcomeTracker(broker)
    outcome = await tracker.poll("c1")
    assert outcome.pnl == pytest.approx(-3.5)


@pytest.mark.asyncio
async def test_poll_propagates_broker_failures_to_caller():
    broker = FakeBrokerClient(raise_on_status=True)
    tracker = ContractOutcomeTracker(broker)
    with pytest.raises(RuntimeError, match="simulated network failure"):
        await tracker.poll("c1")
