"""
Incident — structured incident lifecycle tracking.

STATES: OPEN → ACKNOWLEDGED → MITIGATING → RESOLVED → POST_MORTEM_COMPLETE
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class IncidentSeverity(str, enum.Enum):
    P0 = "P0"  # Total outage / safety system failure
    P1 = "P1"  # Major degradation, SLA breach
    P2 = "P2"  # Moderate impact, drift detected
    P3 = "P3"  # Minor anomaly, informational


class IncidentStatus(str, enum.Enum):
    OPEN = "OPEN"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    MITIGATING = "MITIGATING"
    RESOLVED = "RESOLVED"
    POST_MORTEM_COMPLETE = "POST_MORTEM_COMPLETE"


class Incident(Base):
    """Structured incident with full lifecycle tracking."""

    __tablename__ = "incidents"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Classification
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    severity: Mapped[str] = mapped_column(
        String(10), nullable=False, default=IncidentSeverity.P2.value, index=True
    )
    status: Mapped[str] = mapped_column(
        String(30), nullable=False, default=IncidentStatus.OPEN.value, index=True
    )

    # Trigger
    trigger_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # DRIFT | SLA_BREACH | FALLBACK_SPIKE | MANUAL
    trigger_entity_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    trigger_detail: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # type: ignore[type-arg]

    # Assignee
    acknowledged_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Mitigation
    mitigation_action: Mapped[str | None] = mapped_column(String(100), nullable=True)
    mitigation_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    mitigated_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    mitigated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Resolution
    resolved_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolution_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Postmortem
    postmortem_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    postmortem_complete: Mapped[bool] = mapped_column(nullable=False, default=False)

    # Timeline (append-only JSON array of {ts, actor, action, note})
    timeline: Mapped[list | None] = mapped_column(JSONB, nullable=True, default=list)  # type: ignore[type-arg]

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        index=True,
    )

    def __repr__(self) -> str:
        return f"<Incident {self.severity} {self.title[:40]} status={self.status}>"
