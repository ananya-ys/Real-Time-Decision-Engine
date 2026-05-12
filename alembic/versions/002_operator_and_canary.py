"""Add operator_events and canary_configs tables

Revision ID: 002_operator_and_canary
Revises: 001_initial_schema
Create Date: 2025-01-02 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "002_operator_and_canary"
down_revision: Union[str, None] = "001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── operator_events ──────────────────────────────────────────
    op.create_table(
        "operator_events",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("actor", sa.String(255), nullable=False),
        sa.Column("actor_role", sa.String(50), nullable=False),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("target", sa.String(100), nullable=True),
        sa.Column("reason", sa.String(1000), nullable=False),
        sa.Column("state_before", JSONB, nullable=True),
        sa.Column("state_after", JSONB, nullable=True),
        sa.Column("success", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("error_detail", sa.String(1000), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_operator_event_created_at", "operator_events", ["created_at"])
    op.create_index("ix_operator_event_actor", "operator_events", ["actor"])
    op.create_index("ix_operator_event_action", "operator_events", ["action"])

    # ── canary_configs ───────────────────────────────────────────
    # Note: Canary state is Redis-backed for instant propagation.
    # This table stores historical canary run records for audit.
    op.create_table(
        "canary_run_logs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "policy_version_id",
            UUID(as_uuid=True),
            sa.ForeignKey("policy_versions.id"),
            nullable=True,
        ),
        sa.Column("policy_type", sa.String(50), nullable=False),
        sa.Column("traffic_pct_start", sa.Integer, nullable=False),
        sa.Column("traffic_pct_end", sa.Integer, nullable=True),
        sa.Column("stage_reached", sa.String(50), nullable=True),
        sa.Column("outcome", sa.String(50), nullable=True),  # PROMOTED | ABORTED | IN_PROGRESS
        sa.Column("abort_reason", sa.String(500), nullable=True),
        sa.Column("started_by", sa.String(255), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metrics_snapshot", JSONB, nullable=True),
    )
    op.create_index("ix_canary_run_started_at", "canary_run_logs", ["started_at"])
    op.create_index("ix_canary_run_policy_type", "canary_run_logs", ["policy_type"])

    # ── cost_logs ────────────────────────────────────────────────
    op.create_table(
        "cost_logs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "decision_log_id",
            UUID(as_uuid=True),
            sa.ForeignKey("decision_logs.id"),
            nullable=True,
        ),
        sa.Column("inference_ms", sa.Float, nullable=False),
        sa.Column("instance_count", sa.Integer, nullable=False),
        sa.Column("compute_cost_usd", sa.Float, nullable=False),
        sa.Column("instance_cost_usd", sa.Float, nullable=False),
        sa.Column("total_cost_usd", sa.Float, nullable=False),
        sa.Column("policy_type", sa.String(50), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_cost_log_created_at", "cost_logs", ["created_at"])


def downgrade() -> None:
    op.drop_table("cost_logs")
    op.drop_table("canary_run_logs")
    op.drop_table("operator_events")
