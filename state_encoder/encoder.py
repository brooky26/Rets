"""
Market State Encoder.

Transforms a Stage-2 FeatureVector (~30 raw features on incompatible
scales) into a compact MarketState — the fixed-dimensionality,
comparable-scale "universal language" every downstream model reads.

Two building blocks do essentially all the work:

  1. OnlineNormalizer (state_encoder/normalizer.py) — puts unbounded
     features on a common z-score scale via Welford's running mean/std.
  2. tanh squashing — maps an unbounded weighted-average z-score into
     (-1, 1): tanh(x) is the natural choice here because it's smooth,
     monotonic, symmetric, and saturates gracefully rather than clipping
     hard at extremes (a z-score of 3 and a z-score of 8 both mean "very
     unusual", and tanh reflects that without letting either blow up the
     scale for everything else).

Dimension-specific notes:

  - trend / momentum / acceleration / volatility / noise / uncertainty:
    generic weighted-zscore-then-tanh, per the FeatureMapping config.
  - persistence: mapped directly from the Hurst exponent via an affine
    transform centered at 0.5 (random walk) rather than z-scored — the
    Hurst exponent is already meaningful on its own absolute scale, and
    z-scoring it against a running mean would erase that meaning.
  - complexity: same reasoning, affine-mapped from the Higuchi fractal
    dimension (natural range ~[1, 2], centered at 1.5).
  - compression_expansion: log-ratio of a short-window volatility to a
    long-window volatility,
        compression_expansion = tanh( ln(std_short / std_long) )
    Positive = short-term vol exceeds long-term vol (expansion/breakout-like
    conditions); negative = short-term vol is subdued relative to longer
    history (compression).
  - liquidity: **explicit placeholder**, always 0.0. Deriv's synthetic
    indices (Step Index included) are continuously generated at a fixed
    synthetic volatility with no real order book, bid/ask spread, or
    market depth — there is no genuine liquidity signal to encode here.
    The dimension is kept in the vector for interface compatibility with
    the spec (and in case non-synthetic instruments are added later);
    encoding a fake liquidity number would be worse than being honest
    that this dimension carries no information for this instrument class.
  - market_phase: a **rough continuous proxy** built from a weighted
    blend of the trend and compression_expansion dimensions:
        market_phase = tanh(w_trend * trend + w_compression * compression_expansion)
    This is explicitly NOT the categorical regime classification the
    spec calls for at Level 1 (Strong Trend / Range / Breakout / etc.) —
    that requires the HMM/GMM/clustering machinery of the next stage.
    This is a cheap, always-available scalar for models that want a
    phase-ish signal before Level 1 regime detection exists.
"""

from __future__ import annotations

import math

from configs.state_encoder_schema import FeatureMapping, StateEncoderConfig
from features.types import FeatureVector
from state_encoder.normalizer import OnlineNormalizer
from state_encoder.types import MarketState

NAN = float("nan")


class MarketStateEncoder:
    def __init__(
        self,
        config: StateEncoderConfig,
        normalizer: OnlineNormalizer | None = None,
    ) -> None:
        self._config = config
        self._normalizer = normalizer or OnlineNormalizer()

    @property
    def normalizer(self) -> OnlineNormalizer:
        """Exposed for persistence (save/restore running stats across restarts)."""
        return self._normalizer

    def encode(self, feature_vector: FeatureVector, update_normalizer: bool = True) -> MarketState:
        """
        Encode one FeatureVector into a MarketState.

        `update_normalizer=True` (default, for live streaming and
        chronological backtesting) folds this observation into the
        running stats before computing its z-score. Set False only for
        genuinely out-of-order inspection/debugging use where you want a
        read-only snapshot against already-accumulated stats.
        """
        values = feature_vector.values

        trend = self._combine(self._config.trend, values, update_normalizer)
        momentum = self._combine(self._config.momentum, values, update_normalizer)
        acceleration = self._combine(self._config.acceleration, values, update_normalizer)
        volatility = self._combine(self._config.volatility, values, update_normalizer)
        noise = self._combine(self._config.noise, values, update_normalizer)
        uncertainty = self._combine(self._config.uncertainty, values, update_normalizer)

        persistence = self._affine_direct(
            values.get(self._config.hurst_feature_key), center=0.5, scale=0.5
        )
        complexity = self._affine_direct(
            values.get(self._config.fractal_feature_key), center=1.5, scale=0.5
        )
        compression_expansion = self._compression_expansion(values)

        market_phase = self._market_phase(trend, compression_expansion)

        return MarketState(
            symbol=feature_vector.symbol,
            epoch=feature_vector.epoch,
            trend=trend,
            momentum=momentum,
            acceleration=acceleration,
            volatility=volatility,
            noise=noise,
            persistence=persistence,
            compression_expansion=compression_expansion,
            complexity=complexity,
            uncertainty=uncertainty,
            liquidity=0.0,  # see class docstring: no genuine liquidity signal for synthetic indices
            market_phase=market_phase,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _combine(
        self,
        mappings: list[FeatureMapping],
        values: dict[str, float],
        update_normalizer: bool,
    ) -> float:
        if not mappings:
            return NAN

        weighted_sum = 0.0
        total_weight = 0.0
        for mapping in mappings:
            if mapping.feature_key not in values:
                raise KeyError(
                    f"State encoder config references feature '{mapping.feature_key}' "
                    f"which is not present in the incoming FeatureVector. "
                    f"Check that configs/state_encoder_schema.py matches the "
                    f"FeatureEngineeringConfig windows actually in use."
                )
            raw = values[mapping.feature_key]
            if raw != raw:  # NaN — insufficient history upstream
                return NAN

            if mapping.transform == "zscore":
                if update_normalizer:
                    transformed = self._normalizer.update_and_zscore(mapping.feature_key, raw)
                else:
                    transformed = self._normalizer.zscore(mapping.feature_key, raw)
            else:  # affine
                transformed = self._clip(
                    (raw - mapping.center) / mapping.scale, -1.0, 1.0
                )

            weighted_sum += transformed * mapping.weight
            total_weight += mapping.weight

        if total_weight == 0:
            return NAN
        return math.tanh(weighted_sum / total_weight)

    @staticmethod
    def _affine_direct(raw: float | None, center: float, scale: float) -> float:
        if raw is None or raw != raw:
            return NAN
        return MarketStateEncoder._clip((raw - center) / scale, -1.0, 1.0)

    def _compression_expansion(self, values: dict[str, float]) -> float:
        short_key = self._config.compression_short_std_key
        long_key = self._config.compression_long_std_key
        if short_key not in values or long_key not in values:
            raise KeyError(
                f"State encoder needs '{short_key}' and '{long_key}' for "
                f"compression_expansion — check VolatilityFeatureConfig.std_windows."
            )
        short_std = values[short_key]
        long_std = values[long_key]
        if short_std != short_std or long_std != long_std:
            return NAN
        if short_std <= 0 or long_std <= 0:
            return 0.0  # no meaningful ratio when either side is degenerate/flat
        return math.tanh(math.log(short_std / long_std))

    def _market_phase(self, trend: float, compression_expansion: float) -> float:
        if trend != trend or compression_expansion != compression_expansion:
            return NAN
        w_t = self._config.market_phase_trend_weight
        w_c = self._config.market_phase_compression_weight
        return math.tanh(w_t * trend + w_c * compression_expansion)

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))
