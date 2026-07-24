"""
Meta Learner — regime-aware dynamic ensemble weights.

Role in the platform
------------------------
`EnsembleFusionEngine` (see `ensemble/fusion_engine.py`) needs a weight
per member model to combine their `ProbabilityEstimate`s. Those weights
are not static: `ensemble/bayesian_weight_optimizer.py` periodically
re-optimizes them (globally, or per-regime) against realized trade
outcomes as part of the Continuous Learning daily cycle. `WeightLearner`
is the thin stateful layer between "an optimizer produced a new weight
vector" and "the fusion engine, on the very next prediction, uses it" —
it is intentionally NOT the optimizer itself (that's a much heavier,
occasional batch job) and NOT the fusion engine itself (that's a
lightweight per-prediction combiner) — just the shared piece of state
both read/write, exactly analogous to how `ModelRegistry` sits between
"a candidate model finished training" and "the live prediction path
uses whichever model is currently champion."

Fallback behavior
----------------------
`get_weights(regime)` returns the regime-specific weights ONLY if that
regime has been explicitly set with enough supporting optimization
trials (tracked via each `EnsembleWeights.n_trials` and gated by
`MetaLearningConfig.min_regime_samples_before_specific_weights`);
otherwise it falls back to the global weights — mirroring the same
"don't trust a sparse regime-specific slice over the pooled estimate"
caution `regime/rule_based.py` and `champion_challenger` apply
elsewhere in this codebase.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from configs.meta_learning_schema import MetaLearningConfig
from meta_learning.types import EnsembleWeights, WeightStore
from regime.types import RegimeLabel

logger = logging.getLogger("meta_learning.weight_learner")


def _equal_weights(model_names: list[str], source: str = "default") -> EnsembleWeights:
    n = len(model_names)
    return EnsembleWeights(weights={name: 1.0 / n for name in model_names}, source=source, n_trials=0)


class WeightLearner:
    def __init__(self, config: MetaLearningConfig) -> None:
        self._config = config
        self._store = WeightStore(global_weights=_equal_weights(config.model_names))

    @property
    def store(self) -> WeightStore:
        return self._store

    def get_weights(self, regime: RegimeLabel | None = None) -> EnsembleWeights:
        """Returns the best weights available for `regime` — regime-
        specific if sufficiently supported, else the global fallback.
        `regime=None` always returns the global weights directly (used
        when no regime classification is available for this prediction)."""
        if regime is None:
            return self._store.global_weights

        regime_specific = self._store.regime_weights.get(regime)
        if regime_specific is not None and regime_specific.n_trials >= self._config.min_regime_samples_before_specific_weights:
            return regime_specific

        logger.debug(
            "No sufficiently-supported regime-specific weights for %s (need >= %d trials) — using global weights.",
            regime.value, self._config.min_regime_samples_before_specific_weights,
        )
        return self._store.global_weights

    def set_global_weights(self, weights: EnsembleWeights) -> None:
        self._validate_model_names(weights)
        self._store = WeightStore(global_weights=weights, regime_weights=dict(self._store.regime_weights))
        logger.info("Global ensemble weights updated (source=%s, n_trials=%d): %s", weights.source, weights.n_trials, weights.weights)

    def set_regime_weights(self, regime: RegimeLabel, weights: EnsembleWeights) -> None:
        self._validate_model_names(weights)
        updated = dict(self._store.regime_weights)
        updated[regime] = weights
        self._store = WeightStore(global_weights=self._store.global_weights, regime_weights=updated)
        logger.info(
            "Regime-specific ensemble weights updated for %s (source=%s, n_trials=%d): %s",
            regime.value, weights.source, weights.n_trials, weights.weights,
        )

    def _validate_model_names(self, weights: EnsembleWeights) -> None:
        expected = set(self._config.model_names)
        got = set(weights.weights.keys())
        if got != expected:
            raise ValueError(
                f"EnsembleWeights keys {sorted(got)} do not match configured model_names {sorted(expected)}"
            )

    # --- persistence -----------------------------------------------------
    # Simple JSON round-trip so the Continuous Learning Orchestrator can
    # persist learned weights across process restarts/redeploys — same
    # "documented deferred item, simplest thing that works now" spirit
    # as `model_registry/store.py`'s JSON-file backend, ahead of the
    # documented (still-deferred) move to Supabase.

    def to_dict(self) -> dict:
        def weights_to_dict(w: EnsembleWeights) -> dict:
            return {"weights": w.weights, "source": w.source, "n_trials": w.n_trials}

        return {
            "global_weights": weights_to_dict(self._store.global_weights),
            "regime_weights": {
                regime.value: weights_to_dict(w) for regime, w in self._store.regime_weights.items()
            },
        }

    def save_to_file(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    def load_from_file(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text())

        def weights_from_dict(d: dict) -> EnsembleWeights:
            return EnsembleWeights(weights=d["weights"], source=d["source"], n_trials=d["n_trials"])

        self._store = WeightStore(
            global_weights=weights_from_dict(payload["global_weights"]),
            regime_weights={
                RegimeLabel(regime_str): weights_from_dict(w)
                for regime_str, w in payload.get("regime_weights", {}).items()
            },
        )
