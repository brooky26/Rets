"""
Config for the live paper-trading pipeline (v1).

Scope, stated up front: this wires Regime -> Probability -> EV -> Risk ->
Opportunity Scoring -> Execution (paper mode) -> Post-Trade together on
live Deriv candles, the live equivalent of what `WalkForwardBacktester`
already does against historical data. Two deliberate simplifications
versus a full live system, both chosen explicitly rather than faked:

1. **In PAPER mode, settlement is next-candle-close direction, not real
   duration_ticks expiry.** This matches the exact label definition the
   probability model is trained on (see backtesting/walk_forward.py's
   `_fit_model` and Stage 5's README recap) and keeps one consistent
   definition of "the real realized outcome" for paper trades, which
   never touch the broker (so this gap has zero effect on capital; it
   only means the *decision* framing is slightly simplified from a
   genuine multi-tick contract). This does NOT apply once execution is
   in LIVE mode: a live buy gets a real `contract_id`, and
   `PaperTradingOrchestrator` polls `ContractOutcomeTracker` for Deriv's
   own real settlement (win/loss/payout on its own duration_ticks clock)
   instead of ever applying this fictional rule to it — see
   `paper_trading/orchestrator.py`'s module docstring.
2. **`assumed_payout_ratio` stands in for a live Deriv proposal.** A real
   proposal (`DerivWebSocketClient.fetch_proposal`) costs nothing and
   touches no money, so wiring it in for realistic live pricing is a
   natural v1.1 improvement — deferred here to keep v1 fully testable
   without a network dependency in the decision loop itself.

Level 7 (RL Trade Management) is deliberately NOT wired into v1: every
paper trade is held to its one-candle settlement rather than managed
with HOLD/SELL decisions. That's a separate, explicitly agreed v2 once
real paper-trading data exists to train the Q-table on.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PaperTradingConfig(BaseModel):
    enabled: bool = Field(
        default=True,
        description="If False, main.py stops at regime classification only (pre-Stage-5 "
        "behavior) — useful for running the data-collection-only mode this platform "
        "started with, without touching the decision pipeline at all.",
    )
    stake: float = Field(default=10.0, description="Fixed paper stake per trade, in account currency units.")
    assumed_payout_ratio: float = Field(
        default=1.90,
        description="payout = stake * assumed_payout_ratio. Stands in for a live Deriv "
        "proposal's real payout — see module docstring simplification #2.",
    )
    duration_ticks: int = Field(
        default=5,
        description="Nominal duration for ContractSpec validity/EV calc — paper-mode "
        "settlement does not actually wait this many ticks (see simplification #1).",
    )
    starting_equity: float = Field(default=1000.0, description="Initial virtual account balance.")
    min_bootstrap_candles: int = Field(
        default=200,
        description="Minimum completed historical candles required to fit a symbol's "
        "initial probability model before it starts making live decisions. Symbols with "
        "fewer available historical candles are skipped (logged, not silently ignored) "
        "until enough live candles accumulate to retry.",
    )
    use_ensemble_fusion: bool = Field(
        default=False,
        description="If True, main.py wires `PaperTradingOrchestrator` with "
        "`fusion_config`/`bagged_gbm_config`/`monte_carlo_config` from the rest of "
        "`PlatformConfig`, so live decisions combine Bayesian Logistic + Bagged GBM + "
        "Monte Carlo GBM evidence via `EnsembleFusionEngine` instead of Bayesian Logistic "
        "alone. False (default) preserves the original single-model behavior exactly — "
        "see `paper_trading/orchestrator.py`'s module docstring, 'Optional Ensemble Fusion'.",
    )
    bagged_gbm_target_samples: int = Field(
        default=500,
        description="Sample count at which the Bagged GBM's fusion weight is treated as "
        "fully trusted (sufficiency ratio = 1.0). Below this (but still >= "
        "min_bootstrap_candles, the hard floor to fit at all), its fusion weight is "
        "shrunk proportionally rather than either sidelining it entirely or trusting it "
        "at full strength on too little data — see meta_learning/sufficiency.py. "
        "Bayesian Logistic has no separate target: min_bootstrap_candles IS its target, "
        "since that floor already represents 'enough for this simple model' by design.",
    )
    use_duration_selection: bool = Field(
        default=False,
        description="If True, main.py wires PaperTradingOrchestrator with "
        "duration_selection_config from the rest of PlatformConfig, so live decisions pick "
        "the candidate duration_ticks maximizing risk-adjusted EV each candle (Monte Carlo "
        "primary, Hurst-exponent-scaling fallback) instead of a single fixed duration_ticks. "
        "False (default) preserves the original static-contract behavior exactly — see "
        "monte_carlo/duration_selector.py's module docstring.",
    )

    @field_validator("stake", "assumed_payout_ratio", "starting_equity")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be positive")
        return v

    @field_validator("duration_ticks", "min_bootstrap_candles", "bagged_gbm_target_samples")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be positive")
        return v
