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
  3. Even once live, every execution re-validates the live proposal
     against what the upstream decision (EV/Risk/Opportunity Scoring)
     was actually based on, and aborts rather than trades if the market
     has drifted too far in the time between scoring and execution.

Contract type mapping
-----------------------
Currently only RISE_FALL is supported for execution (direction=+1 ->
Deriv's "CALL" code, direction=-1 -> "PUT"). Other ContractTypes
(HIGHER_LOWER, TOUCH_NO_TOUCH, IN_OUT) need barrier parameters that
ContractSpec doesn't carry yet — attempting to execute one raises
NotImplementedError rather than silently mis-mapping it to the wrong
Deriv contract code, which would be a much worse failure mode than an
explicit error.

Staleness check
------------------
The EV/Risk/Opportunity decision chain was computed against a
*hypothetical* ContractSpec's reward_to_risk. By the time execution
actually runs, real market conditions (and Deriv's own live pricing)
may have moved. Before buying, the engine recomputes reward_to_risk from
the live proposal and compares:

    live_reward_to_risk = (proposal.payout - proposal.ask_price) / proposal.ask_price
    drift = |live_reward_to_risk - decision_reward_to_risk| / decision_reward_to_risk

If `drift > max_payout_drift_pct`, the trade is aborted — trading on a
stale assumption is exactly the kind of thing a research platform should
refuse to do silently.
"""

from __future__ import annotations

from configs.execution_schema import ExecutionConfig
from expected_value.types import ContractSpec, ContractType, EVEstimate
from execution.types import BrokerClient, ExecutionDecision, ExecutionMode
from opportunity.types import TradeOpportunity
from risk.types import RiskAssessment

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
            proposal = await self._broker_client.fetch_proposal(
                symbol=opportunity.symbol,
                contract_type_code=contract_type_code,
                stake=risk.recommended_stake,
                duration_ticks=contract.duration_ticks,
                currency=self._config.currency,
            )
        except Exception as exc:  # noqa: BLE001 — broker failures are reported, not propagated raw
            return self._error(opportunity, risk.recommended_stake, f"Proposal request failed: {exc}")

        ask_price = float(proposal["ask_price"])
        live_payout = float(proposal["payout"])

        if ask_price <= 0:
            return self._error(opportunity, risk.recommended_stake, "Live proposal returned a non-positive ask_price.")

        live_reward_to_risk = (live_payout - ask_price) / ask_price
        decision_reward_to_risk = ev.reward_to_risk
        drift = (
            abs(live_reward_to_risk - decision_reward_to_risk) / decision_reward_to_risk
            if decision_reward_to_risk > 0
            else float("inf")
        )

        if drift > self._config.max_payout_drift_pct:
            return self._skip(
                opportunity, risk.recommended_stake,
                f"Live proposal reward-to-risk ({live_reward_to_risk:.4f}) drifted "
                f"{drift:.2%} from the decision basis ({decision_reward_to_risk:.4f}), "
                f"exceeding the {self._config.max_payout_drift_pct:.2%} tolerance — aborting "
                "rather than trading on stale numbers.",
            )

        max_acceptable_price = risk.recommended_stake * (1.0 + self._config.price_slippage_tolerance_pct)
        if ask_price > max_acceptable_price:
            return self._skip(
                opportunity, risk.recommended_stake,
                f"Live ask_price {ask_price:.2f} exceeds the maximum acceptable "
                f"{max_acceptable_price:.2f} given the slippage tolerance — aborting.",
            )

        try:
            buy_result = await self._broker_client.buy(proposal["id"], ask_price)
        except Exception as exc:  # noqa: BLE001
            return self._error(opportunity, risk.recommended_stake, f"Buy request failed: {exc}")

        return ExecutionDecision(
            symbol=opportunity.symbol,
            epoch=opportunity.epoch,
            mode=ExecutionMode.LIVE,
            action="buy",
            stake=ask_price,
            payout=float(buy_result.get("payout", live_payout)),
            contract_id=str(buy_result["contract_id"]),
            reason="Live buy executed.",
        )

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
