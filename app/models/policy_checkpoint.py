"""
PolicyCheckpoint — serialized policy weights and state.

WHY THIS EXISTS:
- Kill and restart app → Q-values resume from last checkpoint. Zero data loss.
- is_active invariant: exactly one active per policy_type at any time.
- JSONB weights for bandit Q-values; file path for DQN weights (too large for DB).

WHAT BREAKS IF WRONG:
- No checkpoint persistence = training lost on every restart.
- Two is_active=True for same policy_type = split-brain = undefined behavior.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class PolicyCheckpoint(Base):
    """Serialized policy state for persistence across restarts."""

    __tablename__ = "policy_checkpoints"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    policy_version_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("policy_versions.id"),
        nullable=False,
        index=True,
    )

    weights: Mapped[dict | None] = mapped_column(  # type: ignore[type-arg]
        JSONB,
        nullable=True,
        comment="Serialized Q-values (bandit) or model config (RL)",
    )
    step_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Training steps completed at checkpoint time",
    )
    performance_metric: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Latest eval reward for promotion decision",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Invariant: exactly one active per policy_type",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    def __repr__(self) -> str:
        return (
            f"<PolicyCheckpoint step={self.step_count} "
            f"active={self.is_active} metric={self.performance_metric}>"
        )
