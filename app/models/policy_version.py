"""
PolicyVersion — the model registry.

WHY THIS EXISTS:
- A checkpoint file answers "what are the weights?"
- A PolicyVersion answers "which model made that decision, when was it trained,
  what did it score in eval, when was it promoted, and can we roll back to it?"
- Without this, rollback = binary (RL or Baseline). With this = any specific version.
- normalizer_path is REQUIRED — mismatch with weights = silent wrong inference.

WHAT BREAKS IF WRONG:
- No registry = no model version linked to SLA breach = no accountability.
- No normalizer_path = new weights load with old normalizer = silent wrong inference.
- No eval_seeds tracking = single-seed eval cherry-picking goes undetected.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Enum, Float, Integer, String
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.schemas.common import PolicyStatus, PolicyType


class PolicyVersion(Base):
    """Versioned registry entry for a trained policy."""

    __tablename__ = "policy_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    policy_type: Mapped[str] = mapped_column(
        Enum(PolicyType, name="policy_type_enum", create_constraint=True, create_type=False),
        nullable=False,
    )
    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Monotonically increasing per policy_type",
    )
    algorithm: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="e.g. 'DQN', 'UCB', 'threshold_v2'",
    )

    # ── Training Provenance ─────────────────────────────────────
    training_run_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        comment="Celery task ID that trained this version",
    )

    # ── Artifact Paths ──────────────────────────────────────────
    weights_path: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="Object store path for model weights",
    )
    normalizer_path: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="Object store path for versioned StateNormalizer — MUST match weights",
    )

    # ── Evaluation Metrics ──────────────────────────────────────
    eval_reward_mean: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Mean cumulative reward across eval seeds",
    )
    eval_reward_std: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Std dev across eval seeds (statistical validity)",
    )
    eval_seeds: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of seeds used — must be >= 5 for promotion",
    )

    # ── Lifecycle Status ────────────────────────────────────────
    status: Mapped[str] = mapped_column(
        Enum(PolicyStatus, name="policy_status_enum", create_constraint=True),
        nullable=False,
        default=PolicyStatus.TRAINING,
    )
    promoted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When status changed to ACTIVE",
    )
    demoted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When status changed to RETIRED",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    def __repr__(self) -> str:
        return (
            f"<PolicyVersion {self.policy_type} v{self.version} "
            f"status={self.status} eval={self.eval_reward_mean}>"
        )
