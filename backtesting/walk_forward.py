"""
Walk-Forward Backtester.

Rolling train/test evaluation: fit the probability model on a window of
history, evaluate the full Regime -> Probability -> EV -> Risk ->
Opportunity Scoring chain out-of-sample on the window immediately after
it, record what actually happened, then slide both windows forward and
repeat. This is standard walk-forward analysis — the point is that every
trade's outcome is evaluated using a model that never saw that trade's
own future data during training, unlike a single global train/test split
which only checks this once.

A deliberate change from every earlier end-to-end demo script in this
project: those used `rng.uniform(0,1) < probability_used` to simulate a
win/loss, which is appropriate for illustrating the pipeline's mechanics
but is NOT a real backtest — it manufactures outcomes from the model's
own belief rather than checking that belief against what actually
happened. This module uses the REAL realized direction (whether the next
candle actually closed higher or lower) to settle every trade, which is
what a genuine backtest requires.

Continuity across windows
-----------------------------
The RiskEngine (equity curve, circuit breakers) and TradeOpportunityScorer
(adaptive threshold) carry state continuously across all windows, since
they're modeling one continuous account operating through the whole
backtest period. Only the probability model is refit fresh each window —
that's the one component this analysis is actually testing the
out-of-sample validity of.
"""

from __future__ import annotations

import numpy as np

from configs.backtest_schema import WalkForwardConfig
from configs.ev_schema import ExpectedValueConfig
from configs.opportunity_schema import OpportunityScoringConfig
from configs.post_trade_schema import PostTradeAnalysisConfig
from configs.probability_schema import BayesianLogisticConfig
from configs.risk_schema import RiskConfig
from backtesting.types import WalkForwardReport, WalkForwardWindowResult
from expected_value.engine import ExpectedValueEngine
from expected_value.types import ContractSpec
from opportunity.scorer import TradeOpportunityScorer
from post_trade.analyzer import PostTradeAnalyzer
from post_trade.types import CompletedTrade
from probability.bayesian_logistic import BayesianLogisticRegression
from regime.rule_based import RuleBasedRegimeDetector
from risk.engine import RiskEngine
from risk.types import TradeOutcome
from state_encoder.types import MarketState


class WalkForwardBacktester:
    def __init__(
        self,
        walk_forward_config: WalkForwardConfig,
        probability_config: BayesianLogisticConfig,
        ev_config: ExpectedValueConfig,
        risk_config: RiskConfig,
        opportunity_config: OpportunityScoringConfig,
        post_trade_config: PostTradeAnalysisConfig,
        contract: ContractSpec,
        starting_equity: float,
    ) -> None:
        self._wf_config = walk_forward_config
        self._probability_config = probability_config
        self._ev_engine = ExpectedValueEngine(ev_config)
        self._contract = contract
        self._risk_config = risk_config
        self._opportunity_config = opportunity_config
        self._post_trade_config = post_trade_config
        self._starting_equity = starting_equity

    def run(
        self,
        states: list[MarketState],
        closes: np.ndarray,
        regime_detector: RuleBasedRegimeDetector,
    ) -> WalkForwardReport:
        if len(states) != len(closes):
            raise ValueError("states and closes must be the same length")
        if len(states) < 2:
            raise ValueError("Need at least 2 aligned (state, close) pairs")

        usable_states = states[:-1]
        labels = (np.diff(closes) > 0).astype(int)
        n = len(usable_states)

        wf = self._wf_config
        risk_engine = RiskEngine(self._risk_config, starting_equity=self._starting_equity)
        scorer = TradeOpportunityScorer(self._opportunity_config)
        global_analyzer = PostTradeAnalyzer(self._post_trade_config)

        window_results = []
        window_index = 0
        test_start = wf.train_window_trades

        while test_start + wf.test_window_trades <= n:
            train_start = max(0, test_start - wf.train_window_trades)
            train_end = test_start
            test_end = test_start + wf.test_window_trades

            model = self._fit_model(usable_states[train_start:train_end], labels[train_start:train_end])
            window_analyzer = PostTradeAnalyzer(self._post_trade_config)

            for i in range(test_start, test_end):
                state = usable_states[i]
                prob = model.predict(state)
                regime = regime_detector.classify(state)
                ev = self._ev_engine.evaluate(prob, self._contract)
                risk = risk_engine.assess(ev)
                opp = scorer.evaluate(ev, risk, regime, prob)

                if opp.approved:
                    actual_stake = risk.recommended_stake
                    actual_payout = actual_stake * (1.0 + ev.reward_to_risk)
                    actual_direction = 1 if labels[i] == 1 else -1
                    won = actual_direction == ev.direction
                    pnl = (actual_payout - actual_stake) if won else -actual_stake

                    new_equity = risk_engine.equity + pnl
                    risk_engine.record_trade_result(
                        TradeOutcome(epoch=state.epoch, pnl=pnl, equity_after=new_equity)
                    )
                    trade = CompletedTrade(
                        symbol=state.symbol, entry_epoch=state.epoch, exit_epoch=state.epoch,
                        direction=ev.direction, stake=actual_stake, pnl=pnl,
                        predicted_probability=ev.probability_used, regime_at_entry=opp.regime,
                        quality_score_at_entry=opp.quality_score, exit_reason="expired",
                    )
                    global_analyzer.record_trade(trade)
                    window_analyzer.record_trade(trade)

            window_metrics = window_analyzer.compute_metrics(window=wf.test_window_trades)
            window_results.append(
                WalkForwardWindowResult(
                    window_index=window_index,
                    train_start_index=train_start, train_end_index=train_end,
                    test_start_index=test_start, test_end_index=test_end,
                    metrics=window_metrics,
                )
            )

            window_index += 1
            test_start += wf.step_trades

        aggregate_metrics = global_analyzer.compute_metrics(window=n + 1)
        return WalkForwardReport(windows=tuple(window_results), aggregate_metrics=aggregate_metrics)

    def _fit_model(self, train_states: list[MarketState], train_labels: np.ndarray) -> BayesianLogisticRegression:
        X = np.array(
            [[getattr(s, dim) for dim in self._probability_config.feature_dims] for s in train_states]
        )
        model = BayesianLogisticRegression(self._probability_config)
        model.fit(X, train_labels)
        return model
