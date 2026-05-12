"""
ApprovalRequest — two-person approval workflow for high-risk actions.

WHY THIS EXISTS:
The review: "anything affecting live traffic needs human governance:
request → review → approve → execute."

HIGH-RISK ACTIONS require a second person to approve before execution.
The requester cannot approve their own request.

STATES:
  PENDING  → request submitted, waiting for approver
  APPROVED → second person approved, can now execute
  REJECTED → second person rejected the request
  EXECUTED → action has been executed (terminal)
  EXPIRED  → approval window passed (15 minutes)
  CANCELLED → requester withdrew the request
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


class ApprovalStatus(str, enum.Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXECUTED = "EXECUTED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class ApprovalRequest(Base):
    """Two-person approval request for high-risk operator actions."""

    __tablename__ = "approval_requests"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Who requested
    requester_id: Mapped[str] = mapped_column(String(255), nullable=False)
    requester_role: Mapped[str] = mapped_column(String(50), nullable=False)

    # What action is requested
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    action_target: Mapped[str | None] = mapped_column(String(255), nullable=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    parameters: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # type: ignore[type-arg]
    blast_radius: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Current state
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=ApprovalStatus.PENDING.value, index=True
    )

    # Who approved/rejected
    reviewer_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    reviewer_role: Mapped[str | None] = mapped_column(String(50), nullable=True)
    review_comment: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        index=True,
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Execution trace
    execution_result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # type: ignore[type-arg]

    def __repr__(self) -> str:
        return f"<ApprovalRequest {self.action} by={self.requester_id} status={self.status}>"
