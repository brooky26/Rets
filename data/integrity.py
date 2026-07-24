"""
Data Integrity Validator.

Mathematical basis
-------------------
Price-jump detection uses a rolling z-score on tick-to-tick log returns:

    r_t = ln(P_t / P_{t-1})
    z_t = (r_t - mean(r)) / std(r)

computed over the trailing `min_ticks_for_sigma_estimate` returns. A tick
whose |z_t| exceeds `max_price_jump_sigma` is flagged as a suspect print
rather than silently trusted or silently dropped — flagging (not dropping)
is deliberate: downstream layers decide how to treat suspect data, the
integrity layer's job is only detection.

Gap detection is a simple wall-clock delta check: if the time between two
consecutive ticks for a symbol exceeds `max_allowed_gap_seconds`, the tick
is flagged GAP_DETECTED so the historical-backfill logic can be triggered
to fill the missing window.

Duplicate/out-of-order handling operates on `epoch` (exchange-reported
time), not `received_at` (local wall clock), since network jitter can
reorder delivery even when the exchange sequence is correct.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from configs.schema import DataIntegrityConfig
from data.types import DataQualityFlag, Tick


@dataclass
class _SymbolState:
    last_epoch: int | None = None
    last_quote: float | None = None
    log_returns: deque[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.log_returns is None:
            self.log_returns = deque(maxlen=2000)


class IntegrityValidator:
    """
    Stateful, per-symbol streaming validator. Feed it ticks in arrival
    order via `validate(tick)`; it returns a new Tick with `quality` set
    appropriately. It never mutates the input tick (ticks are frozen).
    """

    def __init__(self, config: DataIntegrityConfig) -> None:
        self._config = config
        self._state: dict[str, _SymbolState] = {}

    def _state_for(self, symbol: str) -> _SymbolState:
        if symbol not in self._state:
            self._state[symbol] = _SymbolState()
        return self._state[symbol]

    def validate(self, tick: Tick) -> Tick:
        state = self._state_for(tick.symbol)
        flag = DataQualityFlag.OK

        # --- duplicate / out-of-order check (on exchange epoch) ---
        if state.last_epoch is not None:
            if tick.epoch == state.last_epoch:
                flag = DataQualityFlag.DUPLICATE
            elif tick.epoch < state.last_epoch:
                flag = DataQualityFlag.OUT_OF_ORDER

        # --- gap check ---
        if flag == DataQualityFlag.OK and state.last_epoch is not None:
            gap = tick.epoch - state.last_epoch
            if gap > self._config.max_allowed_gap_seconds:
                flag = DataQualityFlag.GAP_DETECTED

        # --- price jump check (z-score on log returns) ---
        if (
            flag == DataQualityFlag.OK
            and state.last_quote is not None
            and state.last_quote > 0
            and tick.quote > 0
        ):
            log_return = math.log(tick.quote / state.last_quote)
            if len(state.log_returns) >= self._config.min_ticks_for_sigma_estimate:
                mean = sum(state.log_returns) / len(state.log_returns)
                variance = sum((r - mean) ** 2 for r in state.log_returns) / len(
                    state.log_returns
                )
                std = math.sqrt(variance)
                if std > 0:
                    z = (log_return - mean) / std
                    if abs(z) > self._config.max_price_jump_sigma:
                        flag = DataQualityFlag.PRICE_JUMP_SUSPECT
            state.log_returns.append(log_return)

        # --- update state (only advance "last" pointers on well-ordered data) ---
        if flag not in (DataQualityFlag.DUPLICATE, DataQualityFlag.OUT_OF_ORDER):
            state.last_epoch = tick.epoch
            state.last_quote = tick.quote

        if flag == DataQualityFlag.OK:
            return tick
        # Return a new Tick with the flag set (Tick is frozen/immutable).
        return Tick(
            symbol=tick.symbol,
            epoch=tick.epoch,
            quote=tick.quote,
            received_at=tick.received_at,
            quality=flag,
        )

    def resolve_duplicate(self, incoming: Tick, existing: Tick) -> Tick | None:
        """
        Apply `duplicate_timestamp_policy` when two ticks share an epoch.
        Returns the tick that should be kept, or None if the incoming one
        should be dropped entirely.
        """
        policy = self._config.duplicate_timestamp_policy
        if policy == "keep_first":
            return None
        if policy == "keep_last":
            return incoming
        if policy == "drop_duplicate":
            return None
        if policy == "raise":
            raise ValueError(
                f"Duplicate tick epoch={incoming.epoch} symbol={incoming.symbol}"
            )
        raise ValueError(f"Unknown duplicate_timestamp_policy: {policy}")
