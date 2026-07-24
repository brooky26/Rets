from datetime import datetime, timezone

import pytest

from configs.schema import DataIntegrityConfig
from data.integrity import IntegrityValidator
from data.types import DataQualityFlag, Tick


def make_tick(epoch: int, quote: float) -> Tick:
    return Tick(
        symbol="STPRNG100",
        epoch=epoch,
        quote=quote,
        received_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def config() -> DataIntegrityConfig:
    return DataIntegrityConfig(
        max_allowed_gap_seconds=10.0,
        max_price_jump_sigma=4.0,
        min_ticks_for_sigma_estimate=5,
        duplicate_timestamp_policy="drop_duplicate",
    )


def test_first_tick_is_ok(config):
    validator = IntegrityValidator(config)
    result = validator.validate(make_tick(1000, 100.0))
    assert result.quality == DataQualityFlag.OK


def test_normal_sequence_stays_ok(config):
    validator = IntegrityValidator(config)
    price = 100.0
    for i in range(20):
        price += 0.01
        result = validator.validate(make_tick(1000 + i, price))
        assert result.quality == DataQualityFlag.OK


def test_gap_detected(config):
    validator = IntegrityValidator(config)
    validator.validate(make_tick(1000, 100.0))
    result = validator.validate(make_tick(1000 + 50, 100.1))  # 50s gap > 10s threshold
    assert result.quality == DataQualityFlag.GAP_DETECTED


def test_duplicate_epoch_flagged(config):
    validator = IntegrityValidator(config)
    validator.validate(make_tick(1000, 100.0))
    result = validator.validate(make_tick(1000, 100.0))
    assert result.quality == DataQualityFlag.DUPLICATE


def test_out_of_order_flagged(config):
    validator = IntegrityValidator(config)
    validator.validate(make_tick(1000, 100.0))
    result = validator.validate(make_tick(999, 100.0))
    assert result.quality == DataQualityFlag.OUT_OF_ORDER


def test_price_jump_flagged_after_warmup(config):
    import random

    rng = random.Random(42)
    validator = IntegrityValidator(config)
    price = 100.0
    # Warm up with small, *varying* returns so std is well-defined and nonzero
    # (identical returns every tick would give std=0 and never trigger the check).
    for i in range(config.min_ticks_for_sigma_estimate + 5):
        price *= 1.0 + rng.uniform(-0.0002, 0.0002)
        validator.validate(make_tick(1000 + i, price))
    # Now inject a huge relative jump (epoch stays contiguous so this isn't
    # also flagged as a gap).
    last_epoch = 1000 + config.min_ticks_for_sigma_estimate + 5 - 1
    huge_jump_tick = make_tick(last_epoch + 1, price * 1.5)
    result = validator.validate(huge_jump_tick)
    assert result.quality == DataQualityFlag.PRICE_JUMP_SUSPECT


def test_duplicate_does_not_advance_state(config):
    validator = IntegrityValidator(config)
    validator.validate(make_tick(1000, 100.0))
    validator.validate(make_tick(1000, 999.0))  # duplicate epoch, garbage price
    # Next legitimate tick should compare against the ORIGINAL last_epoch/last_quote,
    # not the duplicate's garbage price.
    result = validator.validate(make_tick(1001, 100.01))
    assert result.quality == DataQualityFlag.OK


@pytest.mark.parametrize(
    "policy,expected",
    [
        ("drop_duplicate", None),
        ("keep_last", "incoming"),
        ("keep_first", None),
    ],
)
def test_resolve_duplicate_policies(config, policy, expected):
    config.duplicate_timestamp_policy = policy
    validator = IntegrityValidator(config)
    existing = make_tick(1000, 100.0)
    incoming = make_tick(1000, 101.0)
    result = validator.resolve_duplicate(incoming, existing)
    if expected == "incoming":
        assert result is incoming
    else:
        assert result is None


def test_resolve_duplicate_raise_policy(config):
    config.duplicate_timestamp_policy = "raise"
    validator = IntegrityValidator(config)
    existing = make_tick(1000, 100.0)
    incoming = make_tick(1000, 101.0)
    with pytest.raises(ValueError):
        validator.resolve_duplicate(incoming, existing)
