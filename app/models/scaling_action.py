"""
ScalingAction — immutable record of every scaling decision committed.

WHY THIS EXISTS:
- Equivalent to the Hospital appointment state machine. Every state transition is recorded.
- FK to EnvironmentState = forensic traceability: which state produced which action.
- policy_mode distinguishes ACTIVE (committed) from SHADOW (logged only).

WHAT BREAKS IF WRONG:
- No instances_before/after = can't detect invalid transitions.
- No policy_mode = shadow decisions mixed with real decisions in analysis.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.schemas.common import ActionType, PolicyMode, PolicyType


class ScalingAction(Base):
    """Record of a scaling action taken by a policy."""

    __tablename__ = "scaling_actions"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    action_type: Mapped[str] = mapped_column(
        Enum(ActionType, name="action_type_enum", create_constraint=True),
        nullable=False,
    )
    instances_before: Mapped[int] = mapped_column(Integer, nullable=False)
    instances_after: Mapped[int] = mapped_column(Integer, nullable=False)

    policy_type: Mapped[str] = mapped_column(
        Enum(PolicyType, name="policy_type_enum", create_constraint=True),
        nullable=False,
    )
    policy_mode: Mapped[str] = mapped_column(
        Enum(PolicyMode, name="policy_mode_enum", create_constraint=True),
        nullable=False,
        default=PolicyMode.ACTIVE,
    )

    state_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("environment_states.id"),
        nullable=False,
    )

    success_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rollback_trigger: Mapped[bool] = mapped_column(Boolean, default=False)

    committed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    def __repr__(self) -> str:
        return (
            f"<ScalingAction {self.action_type} "
            f"{self.instances_before}->{self.instances_after} "
            f"by={self.policy_type} mode={self.policy_mode}>"
        )
