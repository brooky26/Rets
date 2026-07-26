"""
Continuous Learning Orchestrator.

The daily cycle, in the order the spec names it
-----------------------------------------------------
    collect/validate data
        -> feature engineering
            -> drift detection
                -> train candidates (Bayesian, GBM, sequence models)
                    -> walk-forward-style holdout evaluation
                        -> champion-challenger comparison
                            -> promote/rollback via ModelRegistry
                                -> Bayesian ensemble-weight optimization

Every step is a method on this class so a caller can also run pieces of
the cycle individually (e.g. just drift detection, for a dashboard) —
`run_daily_cycle` is the composition of all of them, not the only entry
point.

What this orchestrator does NOT own
----------------------------------------
- Collecting raw ticks/candles from Deriv — that's `data/deriv_client.py`
  and `data/storage.py`. This orchestrator receives already-encoded
  `MarketState` history (+ aligned closes) from its caller, exactly like
  `WalkForwardBacktester.run` does — keeping this module about
  training/promotion/optimization logic, not data plumbing.
- Feature engineering itself — the `MarketState`s it receives are
  assumed already built by `features/pipeline.py` + `state_encoder`. The
  "feature engineering" step named in the spec's cycle is therefore a
  no-op HERE by design (it already happened upstream, in the same
  MarketState objects) — restated explicitly rather than silently
  skipped, so the daily cycle's step list matches the spec 1:1.
- Persisting fitted model OBJECTS across process restarts.
  `ModelRegistry` (per its own docstring) only tracks metadata; this
  orchestrator keeps fitted model objects in an in-memory
  `_artifacts: dict[model_id, object]`. This is a DEFERRED item,
  consistent with this project's other documented deferred items
  (env-var config overrides, Supabase migration, Railway deployment
  config) — see the README's "Known Limitations" section. In production
  this dict should be backed by real artifact storage (pickled model +
  object store, or a framework-specific serialization) keyed by the
  same `model_id` the registry already uses.

Scheduling
--------------
Two supported ways to actually run this daily, matching the spec's
"Add scheduling (use APScheduler for in-process or document external
Railway cron)":

1. In-process: `start_scheduler()` uses APScheduler's
   `BackgroundScheduler` with a daily cron trigger at
   `config.schedule_cron_hour_utc`. Simple, but the schedule is only
   alive while this process is running — a Railway restart/redeploy
   resets the timer to the next scheduled hour, not "whatever cycles
   were missed." Suitable for a single long-lived Railway service.

2. External cron: configure a second Railway service (Cron Jobs feature,
   or `railway.json`'s `cronSchedule`) to run
   `python -m continuous_learning.orchestrator --run-once` once daily.
   `_run_once_async` (bottom of this file) fetches fresh history for
   `market_data.connection.symbols[0]` via a short-lived Deriv
   connection, runs one `run_daily_cycle`, and persists the model
   registry (`JSONFileModelRegistryStore`), fitted model artifacts
   (`save_artifacts`/`load_artifacts`), and ensemble weights
   (`WeightLearner.save_to_file`) to `CL_DATA_DIR` so the next day's
   fresh process (Railway cron containers are ephemeral — nothing
   in-memory survives between runs) picks up where the last one left
   off. This is more robust against worker redeploys (Railway's cron
   scheduler is external to the main app) at the cost of needing a
   second Railway service with a Volume mounted at `CL_DATA_DIR` — see
   the README's Railway env-var table.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from anomaly.anomaly_detector import AnomalyDetector
from champion_challenger.comparator import ChampionChallengerComparator
from configs.continuous_learning_schema import ContinuousLearningConfig
from configs.schema import PlatformConfig
from continuous_learning.types import CandidateTrainingResult, DailyCycleReport
from drift_detection.detector import DriftDetector
from ensemble.bayesian_weight_optimizer import BayesianWeightOptimizer
from ensemble.fusion_engine import EnsembleFusionEngine
from ensemble.types import WeightOptimizationRecord
from expected_value.types import ContractSpec
from meta_learning.weight_learner import WeightLearner
from model_registry.registry import ModelRegistry
from probability.bayesian_logistic import BayesianLogisticRegression
from probability.gbm import BaggedGBMEstimator
from regime.rule_based import RuleBasedRegimeDetector
from sequence_models._torch_optional import TORCH_AVAILABLE
from sequence_models.dataset import build_sequences_and_labels
from state_encoder.types import MarketState

logger = logging.getLogger("continuous_learning.orchestrator")


class ContinuousLearningOrchestrator:
    def __init__(
        self,
        config: ContinuousLearningConfig,
        platform_config: PlatformConfig,
        model_registry: ModelRegistry,
        contract: ContractSpec,
    ) -> None:
        self._config = config
        self._platform_config = platform_config
        self._registry = model_registry
        self._contract = contract

        self._anomaly_detector = AnomalyDetector(platform_config.anomaly_detection)
        self._drift_detector = DriftDetector(platform_config.drift_detection)
        self._comparator = ChampionChallengerComparator(platform_config.champion_challenger)
        self._regime_detector = RuleBasedRegimeDetector(platform_config.regime_detection.rule_based)
        self._weight_learner = WeightLearner(platform_config.meta_learning)
        self._weight_optimizer = BayesianWeightOptimizer(
            platform_config.bayesian_weight_optimizer, model_names=platform_config.meta_learning.model_names
        )
        self._fusion_engine = EnsembleFusionEngine(platform_config.ensemble_fusion)

        # See module docstring's "What this orchestrator does NOT own" — deferred to real
        # artifact storage in production.
        self._artifacts: dict[str, object] = {}

        if "lstm_sequence" in config.model_types or "transformer_sequence" in config.model_types:
            if not TORCH_AVAILABLE:
                logger.warning(
                    "config.model_types includes sequence models but PyTorch is not installed — "
                    "those model types will be skipped every cycle until torch is installed."
                )

    @property
    def weight_learner(self) -> WeightLearner:
        return self._weight_learner

    @property
    def fusion_engine(self) -> EnsembleFusionEngine:
        return self._fusion_engine

    # ------------------------------------------------------------------ #
    # Artifact persistence (opt-in; used by the external-cron CLI path so
    # a fresh process each day doesn't lose every fitted model — see
    # module docstring's "What this orchestrator does NOT own" and the
    # README's "Known limitations". In-process usage (Option B) can also
    # call these around `run_daily_cycle` if desired; nothing calls them
    # automatically, keeping this fully opt-in like everything else here.
    # ------------------------------------------------------------------ #

    def save_artifacts(self, path: str) -> None:
        """Pickles `self._artifacts` (model_id -> fitted model object) to
        `path`. Raises on failure so the caller can decide whether that's
        fatal."""
        import pickle
        from pathlib import Path as _Path

        p = _Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self._artifacts, f)
        logger.info("Saved %d model artifact(s) to %s.", len(self._artifacts), p)

    def load_artifacts(self, path: str) -> None:
        """Loads a previously `save_artifacts`-written pickle into
        `self._artifacts`, merging with (and overwriting on collision)
        whatever's already there. No-op, logged, if `path` doesn't exist
        yet (e.g. the very first cron run)."""
        import pickle
        from pathlib import Path as _Path

        p = _Path(path)
        if not p.exists():
            logger.info("No artifact file at %s yet — starting with an empty artifact set.", p)
            return
        with open(p, "rb") as f:
            loaded = pickle.load(f)
        self._artifacts.update(loaded)
        logger.info("Loaded %d model artifact(s) from %s.", len(loaded), p)

    # ------------------------------------------------------------------ #
    # Step 1: collect/validate data
    # ------------------------------------------------------------------ #

    def _clean_data(
        self, states: list[MarketState], closes: np.ndarray
    ) -> tuple[list[MarketState], np.ndarray, int]:
        """Drops invalid (NaN) states and, if enough valid history exists
        to fit one, anomalous states (via `AnomalyDetector`) — both
        `states` and `closes` are filtered with the SAME mask so
        alignment is preserved. Returns (clean_states, clean_closes,
        n_anomalies_flagged)."""
        valid_mask = np.array([s.is_valid for s in states])
        n_valid = int(valid_mask.sum())

        anomaly_mask = np.zeros(len(states), dtype=bool)  # True = anomalous, to be dropped
        n_anomalies = 0
        if n_valid >= self._platform_config.anomaly_detection.min_training_samples:
            valid_states = [s for s, v in zip(states, valid_mask) if v]
            self._anomaly_detector.fit(valid_states)
            report = self._anomaly_detector.score_batch(states)
            anomaly_mask = np.array([sc.is_anomaly for sc in report.scores])
            n_anomalies = report.n_flagged
        else:
            logger.info(
                "Only %d valid states available (< %d needed) — skipping anomaly filtering this cycle.",
                n_valid, self._platform_config.anomaly_detection.min_training_samples,
            )

        keep_mask = valid_mask & ~anomaly_mask
        clean_states = [s for s, k in zip(states, keep_mask) if k]
        clean_closes = closes[keep_mask]
        return clean_states, clean_closes, n_anomalies

    # ------------------------------------------------------------------ #
    # Step 3: drift detection (compares the CURRENT champion's behavior
    # across a reference/current split of this cycle's cleaned data)
    # ------------------------------------------------------------------ #

    def _run_drift_detection(self, clean_states: list[MarketState], clean_closes: np.ndarray):
        champion_version = self._registry.get_champion("bayesian_logistic")
        if champion_version is None or champion_version.model_id not in self._artifacts:
            logger.info("No champion model with a live artifact available yet — skipping drift detection this cycle.")
            return None

        champion_model = self._artifacts[champion_version.model_id]
        usable_states = clean_states[:-1]
        labels = (np.diff(clean_closes) > 0).astype(int)
        split = int(len(usable_states) * self._config.drift_reference_fraction)
        if split < 2 or len(usable_states) - split < 2:
            return None

        reference_states, current_states = usable_states[:split], usable_states[split:]
        reference_labels, current_labels = labels[:split], labels[split:]

        reference_predicted = [champion_model.predict(s).prob_up for s in reference_states]
        current_predicted = [champion_model.predict(s).prob_up for s in current_states]
        reference_outcomes = [bool(v) for v in reference_labels]
        current_outcomes = [bool(v) for v in current_labels]
        # Simplified binary "return": +1 if the model's implied direction matched the
        # realized outcome, else -1. This is a proxy for realized trading performance
        # used ONLY to feed the performance-drift check with something return-shaped,
        # not an actual P&L simulation (that already happens more faithfully in
        # `_simulate_holdout_returns` below, which champion-challenger promotion uses).
        reference_returns = [1.0 if (p > 0.5) == o else -1.0 for p, o in zip(reference_predicted, reference_outcomes)]
        current_returns = [1.0 if (p > 0.5) == o else -1.0 for p, o in zip(current_predicted, current_outcomes)]

        return self._drift_detector.run_full_report(
            reference_states, current_states, reference_predicted, current_predicted,
            reference_outcomes, current_outcomes, reference_returns, current_returns,
        )

    # ------------------------------------------------------------------ #
    # Step 4: train candidates
    # ------------------------------------------------------------------ #

    def _fit_candidate(self, model_type: str, train_states: list[MarketState], train_labels: np.ndarray):
        if model_type == "bayesian_logistic":
            cfg = self._platform_config.probability_estimation.bayesian_logistic
            X = np.array([[getattr(s, d) for d in cfg.feature_dims] for s in train_states])
            return BayesianLogisticRegression(cfg).fit(X, train_labels)

        if model_type == "bagged_gbm":
            cfg = self._platform_config.probability_estimation.bagged_gbm
            X = np.array([[getattr(s, d) for d in cfg.feature_dims] for s in train_states])
            return BaggedGBMEstimator(cfg).fit(X, train_labels)

        if model_type == "lstm_sequence":
            if not TORCH_AVAILABLE:
                return None
            from sequence_models.lstm_model import LSTMSequenceEstimator

            cfg = self._platform_config.sequence_models.lstm
            if len(train_states) < cfg.sequence_length + 11:
                return None
            closes_proxy = np.concatenate([[100.0], 100.0 + np.cumsum(np.where(train_labels == 1, 1.0, -1.0))])
            sequences, labels = build_sequences_and_labels(
                train_states + [train_states[-1]], closes_proxy, cfg.feature_dims, cfg.sequence_length
            )
            if len(sequences) < 10:
                return None
            return LSTMSequenceEstimator(cfg).fit(sequences, labels)

        if model_type == "transformer_sequence":
            if not TORCH_AVAILABLE:
                return None
            from sequence_models.transformer_model import TransformerSequenceEstimator

            cfg = self._platform_config.sequence_models.transformer
            if len(train_states) < cfg.sequence_length + 11:
                return None
            closes_proxy = np.concatenate([[100.0], 100.0 + np.cumsum(np.where(train_labels == 1, 1.0, -1.0))])
            sequences, labels = build_sequences_and_labels(
                train_states + [train_states[-1]], closes_proxy, cfg.feature_dims, cfg.sequence_length
            )
            if len(sequences) < 10:
                return None
            return TransformerSequenceEstimator(cfg).fit(sequences, labels)

        raise ValueError(f"Unknown model_type '{model_type}'")

    @staticmethod
    def _holdout_accuracy(model, holdout_states: list[MarketState], holdout_labels: np.ndarray) -> float:
        correct = 0
        n = 0
        for state, label in zip(holdout_states, holdout_labels):
            pe = model.predict(state)
            if not pe.is_valid:
                continue
            predicted_up = pe.prob_up > 0.5
            correct += int(predicted_up == bool(label))
            n += 1
        return correct / n if n > 0 else float("nan")

    def _simulate_holdout_returns(self, model, holdout_states: list[MarketState], holdout_labels: np.ndarray) -> list[float]:
        """
        A trade is 'taken' on every holdout state where the model
        produces a valid, non-zero-direction prediction; the realized
        return_pct uses `self._contract`'s payout structure (win =
        `+profit_if_win/stake`, loss = `-1.0`) against whether the
        realized close direction matched the model's implied direction.
        This is the same trade-settlement logic
        `backtesting.walk_forward.WalkForwardBacktester.run` uses (settle
        against the REAL realized direction, not a probability-sampled
        simulated one), simplified to skip the EV/Risk/Opportunity gates
        entirely — champion-challenger comparison is specifically about
        the PROBABILITY MODEL's quality, not the full pipeline's trading
        decisions, so gating here would conflate the two.
        """
        reward_to_risk = self._contract.profit_if_win / self._contract.stake
        returns = []
        for state, label in zip(holdout_states, holdout_labels):
            pe = model.predict(state)
            if not pe.is_valid or pe.expected_direction == 0:
                continue
            actual_direction = 1 if label == 1 else -1
            won = pe.expected_direction == actual_direction
            returns.append(reward_to_risk if won else -1.0)
        return returns

    def _train_and_promote_candidates(
        self, train_states: list[MarketState], train_labels: np.ndarray,
        holdout_states: list[MarketState], holdout_labels: np.ndarray, cycle_epoch: int,
    ) -> list[CandidateTrainingResult]:
        results = []
        for model_type in self._config.model_types:
            model = self._fit_candidate(model_type, train_states, train_labels)
            if model is None:
                logger.info("Skipping candidate '%s' this cycle (insufficient data or torch unavailable).", model_type)
                continue

            holdout_accuracy = self._holdout_accuracy(model, holdout_states, holdout_labels)
            candidate_version = self._registry.register(
                model_type=model_type, model_name=model_type,
                hyperparameters={}, training_start_epoch=train_states[0].epoch,
                training_end_epoch=train_states[-1].epoch,
                validation_metrics={"holdout_accuracy": holdout_accuracy},
                artifact_reference=f"{model_type}-cycle{cycle_epoch}",
                created_at=cycle_epoch,
            )
            self._artifacts[candidate_version.model_id] = model

            champion_version = self._registry.get_champion(model_type)
            if champion_version is None:
                self._registry.promote(
                    candidate_version.model_id, cycle_epoch,
                    "No existing champion for this model_type; promoting first trained candidate.",
                )
                results.append(CandidateTrainingResult(
                    model_type=model_type, candidate_version=candidate_version,
                    n_train_samples=len(train_states), n_holdout_samples=len(holdout_states),
                    holdout_accuracy=holdout_accuracy, promotion_decision=None, promoted=True,
                ))
                continue

            champion_model = self._artifacts.get(champion_version.model_id)
            if champion_model is None:
                # Deferred artifact-persistence limitation (see module docstring) — the champion's
                # fitted object isn't available in this process (e.g. after a restart). Promote the
                # freshly-trained candidate directly rather than blocking forever on a comparison
                # that can never happen without the missing artifact.
                logger.warning(
                    "Champion '%s' has no live artifact in this process (likely a restart) — "
                    "promoting new candidate '%s' without a champion-challenger comparison.",
                    champion_version.model_id, candidate_version.model_id,
                )
                self._registry.promote(
                    candidate_version.model_id, cycle_epoch,
                    "Champion artifact unavailable in this process; promoting candidate directly.",
                )
                results.append(CandidateTrainingResult(
                    model_type=model_type, candidate_version=candidate_version,
                    n_train_samples=len(train_states), n_holdout_samples=len(holdout_states),
                    holdout_accuracy=holdout_accuracy, promotion_decision=None, promoted=True,
                ))
                continue

            champion_returns = self._simulate_holdout_returns(champion_model, holdout_states, holdout_labels)
            candidate_returns = self._simulate_holdout_returns(model, holdout_states, holdout_labels)
            decision = self._comparator.compare(
                champion_version.model_id, candidate_version.model_id, champion_returns, candidate_returns
            )
            if decision.promote:
                self._registry.promote(candidate_version.model_id, cycle_epoch, decision.reason)
            else:
                self._registry.reject(candidate_version.model_id)

            results.append(CandidateTrainingResult(
                model_type=model_type, candidate_version=candidate_version,
                n_train_samples=len(train_states), n_holdout_samples=len(holdout_states),
                holdout_accuracy=holdout_accuracy, promotion_decision=decision, promoted=decision.promote,
            ))
        return results

    # ------------------------------------------------------------------ #
    # Step: Bayesian ensemble-weight optimization
    # ------------------------------------------------------------------ #

    def _build_weight_optimization_records(
        self, holdout_states: list[MarketState], holdout_labels: np.ndarray
    ) -> list[WeightOptimizationRecord]:
        champions = {mt: self._registry.get_champion(mt) for mt in self._config.model_types}
        current_weights = self._weight_learner.get_weights()
        reward_to_risk = self._contract.profit_if_win / self._contract.stake

        records = []
        for state, label in zip(holdout_states, holdout_labels):
            if not state.is_valid:
                continue
            model_probs: dict[str, float] = {}
            for mt, champ in champions.items():
                model = self._artifacts.get(champ.model_id) if champ else None
                if model is None:
                    continue
                pe = model.predict(state)
                if pe.is_valid:
                    model_probs[mt] = pe.prob_up
            if not model_probs:
                continue

            available_weight = {n: current_weights.weights.get(n, 0.0) for n in model_probs}
            total = sum(available_weight.values())
            if total <= 0:
                continue
            fused_prob_up = sum(w * model_probs[n] for n, w in available_weight.items()) / total
            direction = 1 if fused_prob_up >= 0.5 else -1
            won = direction == (1 if label == 1 else -1)
            realized_return_pct = reward_to_risk if won else -1.0

            regime = self._regime_detector.classify(state).regime if state.is_valid else None
            records.append(WeightOptimizationRecord(
                symbol=state.symbol, epoch=state.epoch, model_probabilities=model_probs,
                direction=direction, realized_return_pct=realized_return_pct, regime=regime,
            ))
        return records

    def _run_weight_optimization(self, holdout_states: list[MarketState], holdout_labels: np.ndarray):
        records = self._build_weight_optimization_records(holdout_states, holdout_labels)
        if len(records) < self._platform_config.bayesian_weight_optimizer.min_trades_per_regime:
            logger.info("Too few weight-optimization records this cycle (%d) — skipping weight optimization.", len(records))
            return None, {}

        global_weights, regime_weights = self._weight_optimizer.optimize(records)
        self._weight_learner.set_global_weights(global_weights)
        for regime, weights in regime_weights.items():
            self._weight_learner.set_regime_weights(regime, weights)
        return global_weights, regime_weights

    # ------------------------------------------------------------------ #
    # Full cycle
    # ------------------------------------------------------------------ #

    def run_daily_cycle(self, states: list[MarketState], closes: np.ndarray, cycle_epoch: int) -> DailyCycleReport:
        if len(states) != len(closes):
            raise ValueError("states and closes must be the same length")

        clean_states, clean_closes, n_anomalies = self._clean_data(states, closes)

        if len(clean_states) < self._config.min_states_for_cycle:
            return DailyCycleReport(
                cycle_epoch=cycle_epoch, skipped=True,
                skip_reason=f"Only {len(clean_states)} clean states available, need at least "
                f"{self._config.min_states_for_cycle}.",
                n_states_collected=len(states), n_states_after_cleaning=len(clean_states),
                n_anomalies_flagged=n_anomalies, drift_report=None,
            )

        drift_report = self._run_drift_detection(clean_states, clean_closes)
        if drift_report is not None and drift_report.should_trigger_retraining:
            logger.warning(
                "Drift detection triggered retraining this cycle: %s",
                [a.detail for a in drift_report.alerts if a.severity.value != "none"],
            )

        usable_states = clean_states[:-1]
        labels = (np.diff(clean_closes) > 0).astype(int)
        train_end = int(len(usable_states) * self._config.train_fraction)
        train_states, train_labels = usable_states[:train_end], labels[:train_end]
        holdout_states, holdout_labels = usable_states[train_end:], labels[train_end:]

        candidate_results = self._train_and_promote_candidates(
            train_states, train_labels, holdout_states, holdout_labels, cycle_epoch
        )

        global_weights, regime_weights = (None, {})
        if self._config.run_weight_optimization and len(holdout_states) > 0:
            global_weights, regime_weights = self._run_weight_optimization(holdout_states, holdout_labels)

        return DailyCycleReport(
            cycle_epoch=cycle_epoch, skipped=False, skip_reason=None,
            n_states_collected=len(states), n_states_after_cleaning=len(clean_states),
            n_anomalies_flagged=n_anomalies, drift_report=drift_report,
            candidate_results=tuple(candidate_results),
            global_weights=global_weights, regime_weights=regime_weights,
        )

    # ------------------------------------------------------------------ #
    # Scheduling — Option 1: in-process APScheduler
    # ------------------------------------------------------------------ #

    def start_scheduler(self, data_provider) -> "object":
        """
        `data_provider` is a zero-argument callable returning
        `(states, closes, cycle_epoch)` for "the latest window of history
        as of right now" — e.g. wrapping
        `data.deriv_client.DerivWebSocketClient`'s bootstrap-history fetch
        plus `features/pipeline.py` + `state_encoder`. Kept as an
        injected callable (rather than this orchestrator owning a
        DerivWebSocketClient directly) so it stays testable without a
        live connection and reusable from both live and paper-trading
        contexts, matching this project's consistent
        dependency-injection style elsewhere (e.g.
        `paper_trading.orchestrator.PaperTradingOrchestrator`'s injected
        `BrokerClient`).

        Returns the running `BackgroundScheduler` — call `.shutdown()`
        on it to stop.
        """
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger

        def _job() -> None:
            try:
                states, closes, cycle_epoch = data_provider()
                report = self.run_daily_cycle(states, closes, cycle_epoch)
                logger.info(report.summary)
            except Exception:
                logger.exception("Continuous learning daily cycle failed.")

        scheduler = BackgroundScheduler()
        scheduler.add_job(_job, CronTrigger(hour=self._config.schedule_cron_hour_utc, minute=0))
        scheduler.start()
        logger.info(
            "Continuous learning scheduler started — daily cycle runs at %02d:00 UTC.",
            self._config.schedule_cron_hour_utc,
        )
        return scheduler


def _refit_and_persist_hmm_challenger_if_enabled(config, states: list[MarketState], data_dir: str) -> None:
    """
    This is what actually makes the regime-consensus HMM keep learning
    over time, rather than only ever being fit once per main-worker
    restart against a single snapshot: called once per cron invocation
    (see `_run_once_async`), refits a fresh Gaussian HMM against
    whatever history is available *that day*, and pickles it to
    `CL_HMM_PATH` (default `<CL_DATA_DIR>/hmm_regime_detector.pkl`) — the
    exact path `main.py`'s `fit_hmm_challenger_if_enabled` checks for and
    loads at its own next startup, in preference to fitting fresh itself.

    Best-effort and independent of the probability-model daily cycle
    around it: a failure here (insufficient valid states, a fitting
    error) is logged and does not fail the cron run or block the rest of
    `_run_once_async` — the main worker just keeps using whatever HMM
    (or none) it last loaded.
    """
    import os
    import pickle
    from pathlib import Path

    from regime.hmm_detector import GaussianHMMRegimeDetector

    regime_cfg = config.regime_detection
    if not regime_cfg.enable_hmm_promotion:
        return

    valid_states = [s for s in states if s.is_valid]
    if len(valid_states) < regime_cfg.hmm.n_states * 2:
        logger.info(
            "Only %d valid states available — too few to refit the HMM regime-consensus "
            "challenger this cycle (need >= %d) — skipping.",
            len(valid_states), regime_cfg.hmm.n_states * 2,
        )
        return

    hmm_path = os.environ.get("CL_HMM_PATH", str(Path(data_dir) / "hmm_regime_detector.pkl"))
    try:
        hmm_detector = GaussianHMMRegimeDetector(regime_cfg.hmm).fit(valid_states)
        Path(hmm_path).parent.mkdir(parents=True, exist_ok=True)
        with open(hmm_path, "wb") as f:
            pickle.dump(hmm_detector, f)
        logger.info(
            "Refit and persisted the HMM regime-consensus challenger to %s (%d valid states).",
            hmm_path, len(valid_states),
        )
    except Exception as exc:  # noqa: BLE001 — this cycle's own retrain must not be blocked by this
        logger.warning(
            "HMM regime-consensus challenger refit failed this cycle (%s) — the main worker "
            "keeps using whatever it last loaded (or none).", exc,
        )


async def _run_once_async(config_path: str) -> "DailyCycleReport":
    """
    Real implementation behind `--run-once` (Option C / external Railway
    cron). Mirrors what `main.py`'s `bootstrap_paper_trading` does for the
    live worker — a short-lived Deriv connection fetches this cycle's
    history for `market_data.connection.symbols[0]` (same single-symbol
    wiring scope as Option B/in-process, see `ContinuousLearningConfig
    .enabled`'s docstring), replays it through a FRESH
    FeatureEngineeringPipeline/MarketStateEncoder pair (fresh because this
    is a brand-new process each cron run, unlike the long-lived worker's
    continuously-warmed instances), then runs one daily cycle and exits.

    Persistence across cron runs (each invocation is a brand-new process,
    so nothing survives in memory):
      - Model registry: `JSONFileModelRegistryStore` at `CL_REGISTRY_PATH`
        (env var, default `./data_store/model_registry.json`) instead of
        `InMemoryModelRegistryStore` — round-trips versions + promotion
        history to disk.
      - Fitted model objects: `save_artifacts`/`load_artifacts` (see
        above) at `CL_ARTIFACT_PATH` (default
        `./data_store/model_artifacts.pkl`) — otherwise drift detection
        and champion-challenger comparison would have no champion
        artifact to compare against on every single run.
      - Ensemble weights: `WeightLearner.load_from_file`/`save_to_file`
        at `continuous_learning.weight_store_path`.
    All three paths are under one directory so a single Railway Volume
    mounted at that directory covers everything — see the README/env-var
    list for `CL_DATA_DIR`.
    """
    import os
    from pathlib import Path

    from configs.loader import load_config
    from data.candle_aggregator import CandleAggregator  # noqa: F401 (not needed here; history arrives as candles already)
    from data.deriv_client import DerivWebSocketClient, ensure_account_id
    from data.integrity import IntegrityValidator
    from expected_value.types import ContractSpec, ContractType
    from features.pipeline import FeatureEngineeringPipeline
    from model_registry.registry import ModelRegistry
    from model_registry.store import JSONFileModelRegistryStore
    from state_encoder.encoder import MarketStateEncoder

    config = load_config(config_path)
    md_cfg = config.market_data
    md_cfg = md_cfg.model_copy(update={"connection": await ensure_account_id(md_cfg.connection)})
    paper_cfg = config.paper_trading
    cl_cfg = config.continuous_learning

    symbols = md_cfg.connection.symbols
    if not symbols:
        raise RuntimeError("market_data.connection.symbols is empty — nothing to run a cycle for.")
    symbol = symbols[0]  # same single-symbol scope as main.py's in-process wiring (Option B)

    data_dir = os.environ.get("CL_DATA_DIR", "./data_store")
    registry_path = os.environ.get("CL_REGISTRY_PATH", str(Path(data_dir) / "model_registry.json"))
    artifact_path = os.environ.get("CL_ARTIFACT_PATH", str(Path(data_dir) / "model_artifacts.pkl"))
    weight_path = os.environ.get("CL_WEIGHT_STORE_PATH", str(Path(data_dir) / "ensemble_weights.json"))

    # Target candle count: enough for min_states_for_cycle plus the
    # train/holdout split, with the same generous margin
    # `main.py.compute_bootstrap_target_candle_count` uses for the paper
    # -trading bootstrap — reusing whichever is larger so this cycle isn't
    # starved of history relative to what the live worker itself expects.
    target_candle_count = max(
        int(cl_cfg.min_states_for_cycle / min(cl_cfg.train_fraction, 1.0 - cl_cfg.train_fraction)) + 50,
        paper_cfg.min_bootstrap_candles,
    )

    async def _noop_on_tick(tick) -> None:
        return None

    client = DerivWebSocketClient(
        connection_config=md_cfg.connection,
        historical_config=md_cfg.historical,
        integrity_validator=IntegrityValidator(md_cfg.integrity),
        on_tick=_noop_on_tick,
    )

    feature_pipeline = FeatureEngineeringPipeline(config.feature_engineering)
    state_encoder = MarketStateEncoder(config.state_encoder)

    logger.info(
        "Option C run-once: fetching %d candles for %s (granularity=%ds).",
        target_candle_count, symbol, md_cfg.historical.candle_granularity_seconds,
    )
    historical = await client.fetch_bootstrap_history(
        [symbol], md_cfg.historical.candle_granularity_seconds, target_candle_count,
    )
    candles = historical.get(symbol, [])

    states: list[MarketState] = []
    closes: list[float] = []
    for candle in candles:
        vector = feature_pipeline.on_candle(candle)
        if vector is not None:
            state = state_encoder.encode(vector)
            states.append(state)
            closes.append(candle.close)

    if not states:
        raise RuntimeError(
            f"{symbol}: no usable states produced from {len(candles)} fetched candles — "
            "cannot run a daily cycle this invocation."
        )

    _refit_and_persist_hmm_challenger_if_enabled(config, states, data_dir)

    registry = ModelRegistry(JSONFileModelRegistryStore(registry_path))
    contract = ContractSpec(
        contract_type=ContractType.RISE_FALL,
        stake=paper_cfg.stake,
        payout=paper_cfg.stake * paper_cfg.assumed_payout_ratio,
        duration_ticks=paper_cfg.duration_ticks,
    )

    orchestrator = ContinuousLearningOrchestrator(cl_cfg, config, registry, contract)
    orchestrator.load_artifacts(artifact_path)
    if Path(weight_path).exists():
        orchestrator.weight_learner.load_from_file(weight_path)

    report = orchestrator.run_daily_cycle(states, np.array(closes), cycle_epoch=states[-1].epoch)
    logger.info(report.summary)

    orchestrator.save_artifacts(artifact_path)
    orchestrator.weight_learner.save_to_file(weight_path)

    return report


def _cli_main() -> None:
    """
    See the module docstring's "Scheduling" section, option 2 (external
    Railway cron): `python -m continuous_learning.orchestrator --run-once`
    is the single command an external scheduler needs to invoke. Fetches
    fresh history for `market_data.connection.symbols[0]`, runs one daily
    cycle, persists registry/artifacts/weights to `CL_DATA_DIR` (see
    `_run_once_async`'s docstring), and exits — designed to be invoked by
    Railway's Cron Jobs feature against a service with a Volume mounted at
    `CL_DATA_DIR`.
    """
    import asyncio

    parser = argparse.ArgumentParser(description="Continuous Learning Orchestrator CLI")
    parser.add_argument("--run-once", action="store_true", help="Run a single daily cycle and exit (for external cron).")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the platform YAML config.")
    args = parser.parse_args()

    if args.run_once:
        asyncio.run(_run_once_async(args.config))
        return
    parser.print_help()


if __name__ == "__main__":
    _cli_main()
