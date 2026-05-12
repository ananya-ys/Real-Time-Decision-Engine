"""
OperatorEvent — immutable audit log for all operator actions.

WHY THIS EXISTS:
- Every kill switch activation, override, freeze, approval decision
  must be permanently recorded with WHO, WHEN, WHY.
- This is the evidence chain for post-incident review.
- Immutable: no UPDATE allowed on this table. Only INSERT.

QUERIED BY:
- Incident timeline builder (Phase 11)
- Auto-generated postmortem (Phase 16)
- Compliance/audit export
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class OperatorEvent(Base):
    """Immutable record of every operator action."""

    __tablename__ = "operator_events"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Who performed the action
    actor: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="User ID or service name that performed the action",
    )
    actor_role: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Role at time of action",
    )

    # What action was taken
    action: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="e.g. KILL_SWITCH_GLOBAL, FORCE_BASELINE, FREEZE_EXPLORATION",
    )
    target: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="e.g. policy_type=RL, version_id=abc",
    )

    # Why
    reason: Mapped[str] = mapped_column(
        String(1000),
        nullable=False,
        comment="Operator-provided reason for the action",
    )

    # State snapshot at time of action
    state_before: Mapped[dict | None] = mapped_column(  # type: ignore[type-arg]
        JSONB,
        nullable=True,
        comment="System state before this action",
    )
    state_after: Mapped[dict | None] = mapped_column(  # type: ignore[type-arg]
        JSONB,
        nullable=True,
        comment="System state after this action",
    )

    # Outcome
    success: Mapped[bool] = mapped_column(nullable=False, default=True)
    error_detail: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    # Immutable timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        index=True,
    )

    # Hash chain fields — cryptographic immutability
    chain_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="SHA-256 hash of this event's content + prev_hash",
    )
    chain_prev_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="SHA-256 hash of the previous event (genesis=64 zeros)",
    )

    def __repr__(self) -> str:
        return (
            f"<OperatorEvent {self.action} by={self.actor} "
            f"target={self.target} success={self.success}>"
        )
