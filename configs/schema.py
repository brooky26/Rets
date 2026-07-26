"""
Central configuration schema for the AI Trading Research Platform.

Design principle: every tunable parameter in the system must be expressed
here (or in a nested model referenced here) and loaded from YAML — nothing
hardcoded in module logic. This file currently defines the Market Data
Layer configuration. Later modules (features, regime, probability, risk,
RL, etc.) will each add their own Pydantic models here as they are built,
without touching already-built modules.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from configs.backtest_schema import MonteCarloStressConfig, WalkForwardConfig
from configs.feature_schema import FeatureEngineeringConfig
from configs.ev_schema import ExpectedValueConfig
from configs.execution_schema import ExecutionConfig
from configs.opportunity_schema import OpportunityScoringConfig
from configs.paper_trading_schema import PaperTradingConfig
from configs.post_trade_schema import PostTradeAnalysisConfig
from configs.probability_schema import ProbabilityEstimationConfig
from configs.champion_challenger_schema import ChampionChallengerConfig
from configs.drift_detection_schema import DriftDetectionConfig
from configs.duration_selection_schema import DurationSelectionConfig
from configs.regime_schema import RegimeDetectionConfig
from configs.risk_schema import RiskConfig
from configs.state_encoder_schema import StateEncoderConfig
from configs.trade_management_schema import QLearningConfig
from configs.monte_carlo_schema import MonteCarloPricePathConfig
from configs.sequence_models_schema import SequenceModelsConfig
from configs.anomaly_schema import AnomalyDetectionConfig
from configs.meta_learning_schema import MetaLearningConfig
from configs.ensemble_schema import EnsembleFusionConfig, BayesianWeightOptimizerConfig
from configs.continuous_learning_schema import ContinuousLearningConfig


class DerivConnectionConfig(BaseModel):
    """
    Connection parameters for Deriv's current Options API transport.

    Deriv migrated away from the legacy single WebSocket URL
    (wss://ws.derivws.com/websockets/v3?app_id=...) to a REST-issued OTP
    (one-time-password) bootstrap flow:

      1. POST {rest_base_url}/trading/v1/options/accounts/{account_id}/otp
         with headers `Deriv-App-ID` and `Authorization: Bearer <api_token>`
      2. Response contains a ready-to-use WebSocket URL with the OTP
         already embedded, e.g.
         wss://api.derivws.com/trading/v1/options/ws/demo?otp=abc123
      3. Connect directly to that URL — no further auth handshake needed.

    OTP tokens are short-lived, so a fresh OTP must be requested on every
    reconnect, not just on first connect.

    For read-only public market data (ticks/candles on symbols like the
    Step Indices), no auth is required at all: connect straight to
    `ws_public_url`. This client uses the public endpoint automatically
    whenever `api_token`/`account_id` are not provided, and the OTP flow
    whenever they are.
    """

    rest_base_url: str = Field(
        default="https://api.derivws.com",
        description="Base URL for REST endpoints (OTP issuance, account info).",
    )
    ws_public_url: str = Field(
        default="wss://api.derivws.com/trading/v1/options/ws/public",
        description="Unauthenticated WebSocket endpoint for public market data.",
    )
    ws_account_type: str = Field(
        default="demo",
        description="Which authenticated WS endpoint to use once OTP is issued: demo|real.",
    )
    app_id: str = Field(
        ..., description="Deriv application ID, sent as the Deriv-App-ID header."
    )
    api_token: str | None = Field(
        default=None,
        description="Bearer token for the OTP endpoint. None = unauthenticated, public-data-only mode.",
    )
    account_id: str | None = Field(
        default=None,
        description=(
            "Deriv account ID (e.g. 'DOT90004580') used in the OTP request path. If api_token is "
            "set and this is left None, it is auto-resolved at startup via GET "
            "/trading/v1/options/accounts (see data/deriv_client.py's resolve_account_id) — picking "
            "the active account matching ws_account_type. Set this explicitly only if you have "
            "multiple accounts of the same type and want a specific one, or want to skip the extra "
            "startup REST call."
        ),
    )
    symbols: list[str] = Field(
        default=["stpRNG", "stpRNG2", "stpRNG3", "stpRNG4", "stpRNG5"],
        description="Deriv symbol codes for Step Index 100/200/300/400/500.",
    )
    ping_interval_seconds: float = 20.0
    reconnect_initial_backoff_seconds: float = 1.0
    reconnect_max_backoff_seconds: float = 60.0
    reconnect_backoff_multiplier: float = 2.0
    max_reconnect_attempts: int | None = Field(
        default=None, description="None = retry forever."
    )
    request_timeout_seconds: float = Field(
        default=25.0,
        description="Was 15.0 — raised after measuring, across 90 real production timeouts, that "
        "Deriv's actual proposal round-trip time for this account sits at a median of just "
        "0.031s PAST the old 15.0s cutoff — i.e. nearly every proposal request was 'timing out' "
        "locally by milliseconds despite succeeding server-side moments later (caught by the "
        "shield()+forget mechanism in fetch_proposal, but too late to ever actually place a live "
        "trade in time). 25.0 comfortably clears that typical case with margin, while the "
        "shield()+forget safety net still handles the rare genuine outlier (one observed case "
        "arrived a full 60s late).",
    )

    @field_validator("symbols")
    @classmethod
    def _non_empty_symbols(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("symbols list must not be empty")
        return v

    @field_validator("ws_account_type")
    @classmethod
    def _valid_account_type(cls, v: str) -> str:
        allowed = {"demo", "real"}
        if v not in allowed:
            raise ValueError(f"ws_account_type must be one of {allowed}")
        return v

    @property
    def is_authenticated_mode(self) -> bool:
        return self.api_token is not None


class HistoricalDataConfig(BaseModel):
    """Parameters for backfilling historical ticks/candles on startup."""

    lookback_days: int = 30
    candle_granularity_seconds: int = 60
    request_count_max: int = 5000
    pagination_delay_seconds: float = Field(
        default=0.3,
        description="Sleep between successive paginated fetch_historical_candles calls for the "
        "same symbol, so bootstrapping several data-hungry model families across several symbols "
        "doesn't hammer Deriv's API with back-to-back requests right at startup.",
    )


class DataIntegrityConfig(BaseModel):
    """Validation thresholds for the data integrity layer."""

    max_allowed_gap_seconds: float = Field(
        default=30.0,
        description="Ticks gaps larger than this are flagged and trigger a resync.",
    )
    max_price_jump_sigma: float = Field(
        default=12.0,
        description=(
            "A single-tick move beyond this many rolling std-devs is flagged as a "
            "likely bad print rather than a genuine market move."
        ),
    )
    min_ticks_for_sigma_estimate: int = 200
    duplicate_timestamp_policy: str = Field(
        default="drop_duplicate",
        description="One of: drop_duplicate, keep_last, keep_first, raise.",
    )

    @field_validator("duplicate_timestamp_policy")
    @classmethod
    def _valid_policy(cls, v: str) -> str:
        allowed = {"drop_duplicate", "keep_last", "keep_first", "raise"}
        if v not in allowed:
            raise ValueError(f"duplicate_timestamp_policy must be one of {allowed}")
        return v


class StorageConfig(BaseModel):
    """Persistence for the data layer.

    sqlite: local file, zero-ops, fine for research/single-instance dev.
    supabase: Postgres-backed via Supabase's REST API — survives Railway
    redeploys/restarts, since sqlite's local file does not (ephemeral
    filesystem). supabase_url/supabase_key are meant to come from
    environment variables (SUPABASE_URL/SUPABASE_KEY), not committed YAML.
    """

    backend: str = Field(default="sqlite", description="One of: sqlite, supabase.")
    sqlite_path: str = "./data_store/ticks.db"
    supabase_url: str | None = None
    supabase_key: str | None = None
    supabase_table: str = "ticks"
    write_batch_size: int = 500
    flush_interval_seconds: float = 2.0

    @field_validator("backend")
    @classmethod
    def _valid_backend(cls, v: str) -> str:
        allowed = {"sqlite", "supabase"}
        if v not in allowed:
            raise ValueError(f"backend must be one of {allowed}")
        return v

    @model_validator(mode="after")
    def _supabase_credentials_required(self) -> "StorageConfig":
        if self.backend == "supabase" and not (self.supabase_url and self.supabase_key):
            raise ValueError(
                "backend='supabase' requires both supabase_url and supabase_key "
                "(set via SUPABASE_URL/SUPABASE_KEY environment variables)."
            )
        return self


class MarketDataLayerConfig(BaseModel):
    connection: DerivConnectionConfig
    historical: HistoricalDataConfig = HistoricalDataConfig()
    integrity: DataIntegrityConfig = DataIntegrityConfig()
    storage: StorageConfig = StorageConfig()


class PlatformConfig(BaseModel):
    """Top-level config. Later stages attach their own sub-configs here."""

    market_data: MarketDataLayerConfig
    environment: str = Field(default="development", description="development|paper|live")
    feature_engineering: FeatureEngineeringConfig = FeatureEngineeringConfig()
    state_encoder: StateEncoderConfig = StateEncoderConfig()
    regime_detection: RegimeDetectionConfig = RegimeDetectionConfig()
    probability_estimation: ProbabilityEstimationConfig = ProbabilityEstimationConfig()
    expected_value: ExpectedValueConfig = ExpectedValueConfig()
    risk: RiskConfig = RiskConfig()
    opportunity_scoring: OpportunityScoringConfig = OpportunityScoringConfig()
    execution: ExecutionConfig = ExecutionConfig()
    trade_management: QLearningConfig = QLearningConfig()
    post_trade_analysis: PostTradeAnalysisConfig = PostTradeAnalysisConfig()
    paper_trading: PaperTradingConfig = PaperTradingConfig()
    champion_challenger: ChampionChallengerConfig = ChampionChallengerConfig()
    drift_detection: DriftDetectionConfig = DriftDetectionConfig()
    duration_selection: DurationSelectionConfig = DurationSelectionConfig()
    monte_carlo_stress: MonteCarloStressConfig = MonteCarloStressConfig()
    walk_forward: WalkForwardConfig = WalkForwardConfig()
    monte_carlo_price_paths: MonteCarloPricePathConfig = MonteCarloPricePathConfig()
    sequence_models: SequenceModelsConfig = SequenceModelsConfig()
    anomaly_detection: AnomalyDetectionConfig = AnomalyDetectionConfig()
    meta_learning: MetaLearningConfig = MetaLearningConfig()
    ensemble_fusion: EnsembleFusionConfig = EnsembleFusionConfig()
    bayesian_weight_optimizer: BayesianWeightOptimizerConfig = BayesianWeightOptimizerConfig()
    continuous_learning: ContinuousLearningConfig = ContinuousLearningConfig()

    @field_validator("environment")
    @classmethod
    def _valid_env(cls, v: str) -> str:
        allowed = {"development", "paper", "live"}
        if v not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v
