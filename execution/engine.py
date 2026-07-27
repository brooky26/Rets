"""
Execution Engine — Level 6.

The first stage in the whole pipeline that can touch real money. Given
that, this module is built around three separate safety rails rather
than one:

  1. Paper mode is the config default (`ExecutionConfig.mode = "paper"`).
  2. Live mode requires BOTH `ExecutionConfig.mode == "live"` AND the
     top-level `PlatformConfig.environment == "live"` to agree — checked
     at construction time (fails loudly and immediately, not on the
     first trade attempt) so a single misconfigured flag can never
     accidentally enable real trading.
  3. Position sizing and per-trade risk limits (Risk gate, upstream of
     this module) are unaffected by anything below.

Contract type mapping
-----------------------
Currently only RISE_FALL is supported for execution (direction=+1 ->
Deriv's "CALL" code, direction=-1 -> "PUT"). Other ContractTypes
(HIGHER_LOWER, TOUCH_NO_TOUCH, IN_OUT) need barrier parameters that
ContractSpec doesn't carry yet — attempting to execute one raises
NotImplementedError rather than silently mis-mapping it to the wrong
Deriv contract code, which would be a much worse failure mode than an
explicit error.

One-step buy (default) — no pre-trade drift check anymore
------------------------------------------------------------
This used to be a two-step flow: fetch_proposal() for a live quote, check
it against the EV/Risk decision's assumptions, abort if drifted too far,
then buy(proposal_id, price). It is now `broker_client.buy_direct(...)` —
ONE request that quotes and buys atomically, adopted specifically to cut
the number of round-trips exposed to Deriv's response-timing behavior in
half (see data/deriv_client.py's buy_direct docstring for the production
evidence behind this).

This is a real, deliberate trade-off, not a strict improvement: the old
flow could see the live price BEFORE committing money and abort if it
had drifted too far from what the decision was based on. buy_direct has
no such checkpoint — you commit to whatever price Deriv fills at. What
replaces it is a POST-hoc check: `max_payout_drift_pct` is still read
from config and still compared against the ACTUAL fill, but only to log
a clear warning after the fact (`_log_drift_warning_if_needed`) — since
the trade has already happened by the time this runs, it can inform
future tuning but cannot abort anything. Adopted as the platform default
per an explicit, informed choice weighing round-trip risk against this
lost safety rail — not a silent regression.
"""

from __future__ import annotations

import asyncio
import logging

from configs.execution_schema import ExecutionConfig
from expected_value.types import ContractSpec, ContractType, EVEstimate
from execution.types import BrokerClient, ExecutionDecision, ExecutionMode
from opportunity.types import TradeOpportunity
from risk.types import RiskAssessment

logger = logging.getLogger(__name__)

_CONTRACT_TYPE_CODES = {1: "CALL", -1: "PUT"}


class ExecutionConfigurationError(Exception):
    """Raised at construction when the two independent live-mode safety rails disagree."""


class ExecutionEngine:
    def __init__(
        self,
        config: ExecutionConfig,
        platform_environment: str,
        broker_client: BrokerClient | None = None,
    ) -> None:
        self._config = config
        self._mode = ExecutionMode(config.mode)

        if self._mode == ExecutionMode.LIVE:
            if platform_environment != "live":
                raise ExecutionConfigurationError(
                    "ExecutionConfig.mode is 'live' but PlatformConfig.environment is "
                    f"'{platform_environment}', not 'live'. Both must agree before this "
                    "engine will construct in live mode — refusing to start rather than "
                    "risk a single misconfigured flag enabling real trading."
                )
            if broker_client is None:
                raise ExecutionConfigurationError(
                    "ExecutionConfig.mode is 'live' but no broker_client was provided."
                )

        self._broker_client = broker_client

    @property
    def mode(self) -> ExecutionMode:
        return self._mode

    async def execute(
        self,
        opportunity: TradeOpportunity,
        ev: EVEstimate,
        risk: RiskAssessment,
        contract: ContractSpec,
    ) -> ExecutionDecision:
        if not opportunity.approved:
            return self._skip(
                opportunity,
                risk.recommended_stake,
                "Upstream TradeOpportunity was not approved: " + "; ".join(opportunity.veto_reasons),
            )
        if risk.recommended_stake <= 0:
            return self._skip(opportunity, 0.0, "Risk-recommended stake is zero — nothing to execute.")
        if contract.contract_type != ContractType.RISE_FALL:
            return self._error(
                opportunity, risk.recommended_stake,
                f"Execution does not yet support contract type '{contract.contract_type.value}' "
                "(missing barrier parameters on ContractSpec) — refusing rather than guessing.",
            )
        if ev.direction not in _CONTRACT_TYPE_CODES:
            return self._error(
                opportunity, risk.recommended_stake,
                f"No Deriv contract_type code for direction={ev.direction}.",
            )

        if self._mode == ExecutionMode.PAPER:
            return self._simulate_paper_buy(opportunity, ev, risk, contract)
        return await self._execute_live(opportunity, ev, risk, contract)

    # ------------------------------------------------------------------ #
    # Paper mode
    # ------------------------------------------------------------------ #

    def _simulate_paper_buy(
        self, opportunity: TradeOpportunity, ev: EVEstimate, risk: RiskAssessment, contract: ContractSpec
    ) -> ExecutionDecision:
        simulated_payout = risk.recommended_stake * (1.0 + ev.reward_to_risk)
        return ExecutionDecision(
            symbol=opportunity.symbol,
            epoch=opportunity.epoch,
            mode=ExecutionMode.PAPER,
            action="buy",
            stake=risk.recommended_stake,
            payout=simulated_payout,
            contract_id=None,
            reason="Paper mode: simulated buy, no real order placed.",
        )

    # ------------------------------------------------------------------ #
    # Live mode
    # ------------------------------------------------------------------ #

    async def _execute_live(
        self, opportunity: TradeOpportunity, ev: EVEstimate, risk: RiskAssessment, contract: ContractSpec
    ) -> ExecutionDecision:
        assert self._broker_client is not None  # guaranteed by __init__ in live mode

        contract_type_code = _CONTRACT_TYPE_CODES[ev.direction]
        try:
            buy_result = await self._broker_client.buy_direct(
                symbol=opportunity.symbol,
                contract_type_code=contract_type_code,
                stake=risk.recommended_stake,
                duration_ticks=contract.duration_ticks,
                currency=self._config.currency,
            )
        except asyncio.TimeoutError:
            # asyncio.TimeoutError.__str__() is "" — without this branch it
            # gets logged as an unhelpful blank "Buy request failed: ".
            return self._error(
                opportunity, risk.recommended_stake,
                "Buy request timed out waiting for Deriv's response "
                "(no error from the API — the request just never came back "
                "within request_timeout_seconds). IMPORTANT: the contract may "
                "still have been bought on Deriv's side even though we never "
                "got confirmation — check the account before retrying.",
            )
        except Exception as exc:  # noqa: BLE001 — broker failures are reported, not propagated raw
            return self._error(
                opportunity, risk.recommended_stake,
                f"Buy request failed: {exc!r} (type={type(exc).__name__})",
            )

        buy_price = float(buy_result["buy_price"]) if "buy_price" in buy_result else risk.recommended_stake
        live_payout = float(buy_result.get("payout", 0.0))
        self._log_drift_warning_if_needed(opportunity, ev, buy_price, live_payout)

        return ExecutionDecision(
            symbol=opportunity.symbol,
            epoch=opportunity.epoch,
            mode=ExecutionMode.LIVE,
            action="buy",
            stake=buy_price,
            payout=live_payout,
            contract_id=str(buy_result["contract_id"]),
            reason="Live buy executed (one-step buy_direct).",
        )

    def _log_drift_warning_if_needed(
        self, opportunity: TradeOpportunity, ev: EVEstimate, buy_price: float, live_payout: float,
    ) -> None:
        """
        POST-hoc replacement for the old PRE-trade drift check — see this
        module's docstring for why buy_direct has no checkpoint to abort
        at. This can only inform (log a clear warning for anyone tuning
        max_payout_drift_pct or reviewing trade quality after the fact);
        it cannot undo a trade that's already happened. Never raises —
        a logging-only check must not be able to fail the trade it's
        reporting on.
        """
        if buy_price <= 0:
            return
        try:
            live_reward_to_risk = (live_payout - buy_price) / buy_price
            decision_reward_to_risk = ev.reward_to_risk
            drift = (
                abs(live_reward_to_risk - decision_reward_to_risk) / decision_reward_to_risk
                if decision_reward_to_risk > 0
                else float("inf")
            )
            if drift > self._config.max_payout_drift_pct:
                logger.warning(
                    "%s: filled reward-to-risk (%.4f) drifted %.2f%% from the decision basis "
                    "(%.4f), exceeding the %.2f%% tolerance — this already happened (buy_direct "
                    "has no pre-trade abort point, see execution/engine.py's docstring); logged "
                    "for review, not blocked.",
                    opportunity.symbol, live_reward_to_risk, drift * 100, decision_reward_to_risk,
                    self._config.max_payout_drift_pct * 100,
                )
        except Exception:  # noqa: BLE001 — this is an informational check only, never fatal
            logger.debug("%s: drift warning check itself failed — skipping.", opportunity.symbol, exc_info=True)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _skip(self, opportunity: TradeOpportunity, stake: float, reason: str) -> ExecutionDecision:
        return ExecutionDecision(
            symbol=opportunity.symbol, epoch=opportunity.epoch, mode=self._mode,
            action="skip", stake=stake, payout=0.0, contract_id=None, reason=reason,
        )

    def _error(self, opportunity: TradeOpportunity, stake: float, reason: str) -> ExecutionDecision:
        return ExecutionDecision(
            symbol=opportunity.symbol, epoch=opportunity.epoch, mode=self._mode,
            action="error", stake=stake, payout=0.0, contract_id=None, reason=reason,
        )
