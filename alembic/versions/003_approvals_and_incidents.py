"""Add approval_requests and incidents tables

Revision ID: 003_approvals_and_incidents
Revises: 002_operator_and_canary
Create Date: 2025-01-03 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "003_approvals_and_incidents"
down_revision: Union[str, None] = "002_operator_and_canary"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── approval_requests ────────────────────────────────────────
    op.create_table(
        "approval_requests",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("requester_id", sa.String(255), nullable=False),
        sa.Column("requester_role", sa.String(50), nullable=False),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("action_target", sa.String(255), nullable=True),
        sa.Column("reason", sa.Text, nullable=False),
        sa.Column("parameters", JSONB, nullable=True),
        sa.Column("blast_radius", sa.Text, nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="PENDING"),
        sa.Column("reviewer_id", sa.String(255), nullable=True),
        sa.Column("reviewer_role", sa.String(50), nullable=True),
        sa.Column("review_comment", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("execution_result", JSONB, nullable=True),
    )
    op.create_index("ix_approval_requests_status", "approval_requests", ["status"])
    op.create_index("ix_approval_requests_created_at", "approval_requests", ["created_at"])
    op.create_index("ix_approval_requests_requester_id", "approval_requests", ["requester_id"])

    # ── incidents ────────────────────────────────────────────────
    op.create_table(
        "incidents",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("severity", sa.String(10), nullable=False, server_default="P2"),
        sa.Column("status", sa.String(30), nullable=False, server_default="OPEN"),
        sa.Column("trigger_type", sa.String(50), nullable=False),
        sa.Column("trigger_entity_id", sa.String(255), nullable=True),
        sa.Column("trigger_detail", JSONB, nullable=True),
        sa.Column("acknowledged_by", sa.String(255), nullable=True),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("mitigation_action", sa.String(100), nullable=True),
        sa.Column("mitigation_detail", sa.Text, nullable=True),
        sa.Column("mitigated_by", sa.String(255), nullable=True),
        sa.Column("mitigated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_by", sa.String(255), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolution_note", sa.Text, nullable=True),
        sa.Column("postmortem_id", sa.String(255), nullable=True),
        sa.Column("postmortem_complete", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("timeline", JSONB, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_incidents_status", "incidents", ["status"])
    op.create_index("ix_incidents_severity", "incidents", ["severity"])
    op.create_index("ix_incidents_created_at", "incidents", ["created_at"])


def downgrade() -> None:
    op.drop_table("incidents")
    op.drop_table("approval_requests")
