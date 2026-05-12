"""Initial schema - 8 tables

Revision ID: 001_initial_schema
Revises: None
Create Date: 2025-01-01 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ENUM, JSONB, UUID

# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _create_enum_type(name: str, labels: list[str]) -> None:
    """
    Create a PostgreSQL enum type only if it does not already exist.
    This avoids duplicate-object errors during migration re-runs.
    """
    labels_sql = ", ".join(f"'{label}'" for label in labels)
    op.execute(
        sa.text(
            f"""
DO $$
BEGIN
    CREATE TYPE {name} AS ENUM ({labels_sql});
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;
"""
        )
    )


def upgrade() -> None:
    # ── Enum labels ──────────────────────────────────────────────
    traffic_regime_labels = ["STEADY", "BURST", "PERIODIC", "UNKNOWN"]
    state_source_labels = ["REAL", "SIMULATED"]
    action_type_labels = ["SCALE_UP_1", "SCALE_UP_3", "SCALE_DOWN_1", "SCALE_DOWN_3", "HOLD"]
    policy_type_labels = ["BASELINE", "BANDIT", "RL"]
    policy_mode_labels = ["ACTIVE", "SHADOW"]
    policy_status_labels = ["TRAINING", "SHADOW", "ACTIVE", "RETIRED"]
    drift_signal_labels = ["REWARD_DEGRADATION", "INPUT_DRIFT", "BOTH"]
    suppression_reason_labels = ["HIGH_LATENCY", "HIGH_LOAD", "SLA_VIOLATION_STREAK", "MANUAL"]

    # ── Create enum types once ──────────────────────────────────
    _create_enum_type("traffic_regime_enum", traffic_regime_labels)
    _create_enum_type("state_source_enum", state_source_labels)
    _create_enum_type("action_type_enum", action_type_labels)
    _create_enum_type("policy_type_enum", policy_type_labels)
    _create_enum_type("policy_mode_enum", policy_mode_labels)
    _create_enum_type("policy_status_enum", policy_status_labels)
    _create_enum_type("drift_signal_enum", drift_signal_labels)
    _create_enum_type("suppression_reason_enum", suppression_reason_labels)

    # ── SQLAlchemy enum objects (do NOT auto-create types) ───────
    traffic_regime_enum = ENUM(
        *traffic_regime_labels,
        name="traffic_regime_enum",
        create_type=False,
    )
    state_source_enum = ENUM(
        *state_source_labels,
        name="state_source_enum",
        create_type=False,
    )
    action_type_enum = ENUM(
        *action_type_labels,
        name="action_type_enum",
        create_type=False,
    )
    policy_type_enum = ENUM(
        *policy_type_labels,
        name="policy_type_enum",
        create_type=False,
    )
    policy_mode_enum = ENUM(
        *policy_mode_labels,
        name="policy_mode_enum",
        create_type=False,
    )
    policy_status_enum = ENUM(
        *policy_status_labels,
        name="policy_status_enum",
        create_type=False,
    )
    drift_signal_enum = ENUM(
        *drift_signal_labels,
        name="drift_signal_enum",
        create_type=False,
    )
    suppression_reason_enum = ENUM(
        *suppression_reason_labels,
        name="suppression_reason_enum",
        create_type=False,
    )

    # ── 1. environment_states ───────────────────────────────────
    op.create_table(
        "environment_states",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("cpu_utilization", sa.Float, nullable=False),
        sa.Column("request_rate", sa.Float, nullable=False),
        sa.Column("p99_latency_ms", sa.Float, nullable=False),
        sa.Column("instance_count", sa.Integer, nullable=False),
        sa.Column("hour_of_day", sa.Integer, nullable=False, server_default="0"),
        sa.Column("day_of_week", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "traffic_regime",
            traffic_regime_enum,
            nullable=False,
            server_default="UNKNOWN",
        ),
        sa.Column(
            "source",
            state_source_enum,
            nullable=False,
            server_default="SIMULATED",
        ),
        sa.Column("version", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "cpu_utilization >= 0.0 AND cpu_utilization <= 1.0",
            name="ck_env_state_cpu_range",
        ),
        sa.CheckConstraint(
            "request_rate >= 0.0",
            name="ck_env_state_rps_positive",
        ),
        sa.CheckConstraint(
            "p99_latency_ms >= 0.0",
            name="ck_env_state_latency_positive",
        ),
        sa.CheckConstraint(
            "instance_count >= 1",
            name="ck_env_state_instances_min",
        ),
        sa.CheckConstraint(
            "hour_of_day >= 0 AND hour_of_day <= 23",
            name="ck_env_state_hour_range",
        ),
        sa.CheckConstraint(
            "day_of_week >= 0 AND day_of_week <= 6",
            name="ck_env_state_dow_range",
        ),
    )
    op.create_index("ix_env_state_timestamp", "environment_states", ["timestamp"])

    # ── 2. policy_versions (must exist before decision_logs FK) ─
    op.create_table(
        "policy_versions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("policy_type", policy_type_enum, nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("algorithm", sa.String(50), nullable=True),
        sa.Column("training_run_id", UUID(as_uuid=True), nullable=True),
        sa.Column("weights_path", sa.String(500), nullable=True),
        sa.Column("normalizer_path", sa.String(500), nullable=True),
        sa.Column("eval_reward_mean", sa.Float, nullable=True),
        sa.Column("eval_reward_std", sa.Float, nullable=True),
        sa.Column("eval_seeds", sa.Integer, nullable=True),
        sa.Column("status", policy_status_enum, nullable=False, server_default="TRAINING"),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("demoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # ── 3. scaling_actions ──────────────────────────────────────
    op.create_table(
        "scaling_actions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("action_type", action_type_enum, nullable=False),
        sa.Column("instances_before", sa.Integer, nullable=False),
        sa.Column("instances_after", sa.Integer, nullable=False),
        sa.Column("policy_type", policy_type_enum, nullable=False),
        sa.Column(
            "policy_mode",
            policy_mode_enum,
            nullable=False,
            server_default="ACTIVE",
        ),
        sa.Column(
            "state_id",
            UUID(as_uuid=True),
            sa.ForeignKey("environment_states.id"),
            nullable=False,
        ),
        sa.Column("success_flag", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("rollback_trigger", sa.Boolean, server_default="false"),
        sa.Column(
            "committed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # ── 4. decision_logs ────────────────────────────────────────
    op.create_table(
        "decision_logs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("trace_id", UUID(as_uuid=True), nullable=False),
        sa.Column("policy_type", policy_type_enum, nullable=False),
        sa.Column(
            "policy_version_id",
            UUID(as_uuid=True),
            sa.ForeignKey("policy_versions.id"),
            nullable=True,
        ),
        sa.Column("state_snapshot", JSONB, nullable=False),
        sa.Column("action", action_type_enum, nullable=False),
        sa.Column("q_values", JSONB, nullable=True),
        sa.Column("confidence_spread", sa.Float, nullable=True),
        sa.Column("reward", sa.Float, nullable=True),
        sa.Column("latency_ms", sa.Float, nullable=True),
        sa.Column("fallback_flag", sa.Boolean, server_default="false"),
        sa.Column("shadow_flag", sa.Boolean, server_default="false"),
        sa.Column("drift_flag", sa.Boolean, server_default="false"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_decision_log_trace_id", "decision_logs", ["trace_id"])
    op.create_index("ix_decision_log_created_at", "decision_logs", ["created_at"])

    # ── 5. reward_logs ──────────────────────────────────────────
    op.create_table(
        "reward_logs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "decision_log_id",
            UUID(as_uuid=True),
            sa.ForeignKey("decision_logs.id"),
            nullable=False,
        ),
        sa.Column("reward", sa.Float, nullable=False),
        sa.Column("n_step_reward", sa.Float, nullable=True),
        sa.Column("cumulative_reward", sa.Float, nullable=True),
        sa.Column("cumulative_regret", sa.Float, nullable=True),
        sa.Column("baseline_reward", sa.Float, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_reward_log_decision_id", "reward_logs", ["decision_log_id"])

    # ── 6. policy_checkpoints ───────────────────────────────────
    op.create_table(
        "policy_checkpoints",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "policy_version_id",
            UUID(as_uuid=True),
            sa.ForeignKey("policy_versions.id"),
            nullable=False,
        ),
        sa.Column("weights", JSONB, nullable=True),
        sa.Column("step_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("performance_metric", sa.Float, nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="false"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_checkpoint_version_id", "policy_checkpoints", ["policy_version_id"])

    # ── 7. drift_events ─────────────────────────────────────────
    op.create_table(
        "drift_events",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "policy_version_id",
            UUID(as_uuid=True),
            sa.ForeignKey("policy_versions.id"),
            nullable=True,
        ),
        sa.Column("drift_signal", drift_signal_enum, nullable=False),
        sa.Column("psi_score", sa.Float, nullable=True),
        sa.Column("reward_delta", sa.Float, nullable=True),
        sa.Column("window_count", sa.Integer, nullable=True),
        sa.Column("policy_from", policy_type_enum, nullable=False),
        sa.Column("policy_to", policy_type_enum, nullable=False),
        sa.Column("retraining_job_id", UUID(as_uuid=True), nullable=True),
        sa.Column(
            "triggered_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_drift_event_triggered_at", "drift_events", ["triggered_at"])

    # ── 8. exploration_guard_logs ───────────────────────────────
    op.create_table(
        "exploration_guard_logs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "decision_log_id",
            UUID(as_uuid=True),
            sa.ForeignKey("decision_logs.id"),
            nullable=True,
        ),
        sa.Column(
            "exploration_suppressed",
            sa.Boolean,
            nullable=False,
            server_default="true",
        ),
        sa.Column("suppression_reason", suppression_reason_enum, nullable=False),
        sa.Column("state_snapshot", JSONB, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_guard_log_decision_id", "exploration_guard_logs", ["decision_log_id"])


def downgrade() -> None:
    op.drop_table("exploration_guard_logs")
    op.drop_table("drift_events")
    op.drop_table("policy_checkpoints")
    op.drop_table("reward_logs")
    op.drop_table("decision_logs")
    op.drop_table("scaling_actions")
    op.drop_table("policy_versions")
    op.drop_table("environment_states")

    # Drop enums
    for enum_name in [
        "suppression_reason_enum",
        "drift_signal_enum",
        "policy_status_enum",
        "policy_mode_enum",
        "policy_type_enum",
        "action_type_enum",
        "state_source_enum",
        "traffic_regime_enum",
    ]:
        op.execute(sa.text(f'DROP TYPE IF EXISTS "{enum_name}"'))