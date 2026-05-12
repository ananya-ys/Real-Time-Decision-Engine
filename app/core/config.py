"""
Application configuration — single source of truth.
Handles Render's DATABASE_URL format (postgres:// → postgresql+asyncpg://).
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ─────────────────────────────────────────────
    app_name: str = "rtde-backend"
    app_env: str = "development"

    @field_validator("app_env")
    @classmethod
    def validate_app_env(cls, v: str) -> str:
        """
        Reject unknown environment names. A typo like "developement" would
        silently bypass auth (treated as non-production) while secret_key
        validator also skips (expects exactly "production"). Defense in depth.
        """
        allowed = {"development", "staging", "production", "test"}
        if v not in allowed:
            raise ValueError(
                f"APP_ENV must be one of {sorted(allowed)}, got {v!r}. "
                "Check for typos — an unknown env name bypasses safety checks."
            )
        return v

    debug: bool = False
    log_level: str = "INFO"
    secret_key: str = Field(default="change-me-in-production")

    # ── Database ────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://rtde_user:rtde_pass@localhost:5432/rtde_db"
    )
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_timeout: int = 30
    db_echo: bool = False

    # ── Redis ───────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    redis_state_ttl_seconds: int = 10

    # ── Celery ──────────────────────────────────────────────────
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # ── Baseline Policy ─────────────────────────────────────────
    baseline_cpu_high: float = 0.80
    baseline_cpu_critical: float = 0.90
    baseline_cpu_low: float = 0.30
    baseline_latency_high_ms: float = 500.0
    baseline_latency_critical_ms: float = 800.0
    baseline_latency_low_ms: float = 100.0
    min_instances: int = 1
    max_instances: int = 20

    # ── Reward Weights ──────────────────────────────────────────
    reward_alpha_latency: float = 1.0
    reward_beta_cost: float = 0.5
    reward_gamma_sla: float = 2.0
    reward_delta_instability: float = 0.3

    # ── Exploration Guard ───────────────────────────────────────
    exploration_latency_warning_ms: float = 400.0
    exploration_sla_warning_rate: float = 0.03
    exploration_high_load_rps: float = 5000.0
    exploration_max_consecutive_violations: int = 3

    # ── Drift Detection ────────────────────────────────────────
    drift_window_size: int = 100
    drift_hysteresis_k: int = 3
    drift_significance_alpha: float = 0.05
    drift_psi_threshold: float = 0.2
    drift_eval_interval_seconds: int = 60

    # ── RL / Bandit ─────────────────────────────────────────────
    bandit_epsilon_start: float = 1.0
    bandit_epsilon_floor: float = 0.05
    bandit_epsilon_decay: float = 0.995
    rl_learning_rate: float = 0.001
    rl_gamma: float = 0.99
    rl_batch_size: int = 64
    rl_buffer_capacity: int = 50000
    rl_checkpoint_interval: int = 500
    rl_warm_start_min_decisions: int = 1000
    provisioning_delay_ticks: int = 5

    # ── Inference ───────────────────────────────────────────────
    inference_timeout_ms: int = 300
    shadow_mode_enabled: bool = True

    @field_validator("database_url", mode="before")
    @classmethod
    def fix_database_url(cls, v: str) -> str:
        """
        Render (and other hosts) provide postgres:// URLs.
        asyncpg requires postgresql+asyncpg://.
        """
        if v.startswith("postgres://"):
            v = v.replace("postgres://", "postgresql+asyncpg://", 1)
        elif v.startswith("postgresql://") and "+asyncpg" not in v:
            v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    @field_validator("baseline_cpu_high", "baseline_cpu_critical", "baseline_cpu_low")
    @classmethod
    def validate_cpu_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"CPU threshold must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("min_instances")
    @classmethod
    def validate_min_instances(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"min_instances must be >= 1, got {v}")
        return v

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str, info) -> str:
        """
        Prevent accidental use of the default placeholder key in production.
        A forged JWT signed with the default key could impersonate any user.
        """
        unsafe_defaults = {"change-me-in-production", "secret", "changeme", ""}
        app_env = (info.data or {}).get("app_env", "development")
        if app_env == "production" and v in unsafe_defaults:
            raise ValueError(
                "SECRET_KEY must be set to a secure random value in production. "
                "Generate one with: openssl rand -hex 32"
            )
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def sync_database_url(self) -> str:
        """Sync URL for Alembic — removes asyncpg driver."""
        url = self.database_url
        url = url.replace("+asyncpg", "")
        url = url.replace("postgresql://", "postgresql://")
        return url


def get_settings() -> Settings:
    return Settings()
