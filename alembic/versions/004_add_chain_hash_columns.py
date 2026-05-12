"""Add chain_hash columns to operator_events

These columns were added to the OperatorEvent model for cryptographic
audit trail immutability but were never added to any migration.
Without this migration, any code that writes chain_hash will fail with:
  ProgrammingError: column "chain_hash" of relation "operator_events" does not exist

Revision ID: 004_add_chain_hash_columns
Revises: 003_approvals_and_incidents
Create Date: 2025-01-04 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "004_add_chain_hash_columns"
down_revision: Union[str, None] = "003_approvals_and_incidents"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "operator_events",
        sa.Column(
            "chain_hash",
            sa.String(64),
            nullable=True,
            comment="SHA-256 hash of this event's content + prev_hash",
        ),
    )
    op.add_column(
        "operator_events",
        sa.Column(
            "chain_prev_hash",
            sa.String(64),
            nullable=True,
            comment="SHA-256 hash of the previous event (genesis=64 zeros)",
        ),
    )
    # Index for fast chain traversal during verification
    op.create_index(
        "ix_operator_events_chain_hash",
        "operator_events",
        ["chain_hash"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_operator_events_chain_hash", table_name="operator_events")
    op.drop_column("operator_events", "chain_prev_hash")
    op.drop_column("operator_events", "chain_hash")
