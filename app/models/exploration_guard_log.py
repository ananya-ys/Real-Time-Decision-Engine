"""
ExplorationGuardLog — audit trail for exploration suppression events.

WHY THIS EXISTS:
- Every suppression must be logged with the reason and state at suppression time.
- Without this log, you can't tell if the guard is firing too aggressively or not enough.
- Prometheus counter tracks rate; this table provides the forensic detail.

WHAT BREAKS IF WRONG:
- No log = can't tune guard thresholds. Too aggressive = policy never learns.
  Too permissive = SLA breaches during exploration.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.schemas.common import SuppressionReason


class ExplorationGuardLog(Base):
    """Record of an ExplorationGuard suppression event."""

    __tablename__ = "exploration_guard_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    decision_log_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("decision_logs.id"),
        nullable=True,
        index=True,
    )

    exploration_suppressed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    suppression_reason: Mapped[str] = mapped_column(
        Enum(SuppressionReason, name="suppression_reason_enum", create_constraint=True),
        nullable=False,
    )

    state_snapshot: Mapped[dict | None] = mapped_column(  # type: ignore[type-arg]
        JSONB,
        nullable=True,
        comment="State at suppression time for post-analysis",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    def __repr__(self) -> str:
        return (
            f"<ExplorationGuardLog suppressed={self.exploration_suppressed} "
            f"reason={self.suppression_reason}>"
        )
