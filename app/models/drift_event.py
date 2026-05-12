"""
DriftEvent — audit trail for every drift detection + rollback event.

WHY THIS EXISTS:
- Every rollback must be fully auditable: which signal, which version, what stats.
- drift_signal (REWARD_DEGRADATION | INPUT_DRIFT | BOTH) documents the trigger type.
- psi_score + reward_delta provide the quantitative evidence for the rollback decision.
- retraining_job_id links to the Celery task for end-to-end forensic trace.

WHAT BREAKS IF WRONG:
- No drift_signal = can't tell if rollback was from data drift or model decay.
- No psi_score = can't verify the drift detector was correct post-incident.
- No retraining_job_id = can't trace from rollback to retrain to new model version.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.schemas.common import DriftSignal, PolicyType


class DriftEvent(Base):
    """Audit record for a drift detection and rollback event."""

    __tablename__ = "drift_events"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    policy_version_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("policy_versions.id"),
        nullable=True,
        comment="Which policy version was rolled back",
    )

    # ── Drift Detection Results ─────────────────────────────────
    drift_signal: Mapped[str] = mapped_column(
        Enum(DriftSignal, name="drift_signal_enum", create_constraint=True),
        nullable=False,
        comment="Which signal(s) triggered the rollback",
    )
    psi_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Population Stability Index on input features",
    )
    reward_delta: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="active_reward - baseline_reward at trigger point",
    )
    window_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="How many consecutive windows were degraded",
    )

    # ── Policy Transition ───────────────────────────────────────
    policy_from: Mapped[str] = mapped_column(
        Enum(PolicyType, name="policy_type_enum", create_constraint=True, create_type=False),
        nullable=False,
    )
    policy_to: Mapped[str] = mapped_column(
        Enum(PolicyType, name="policy_type_enum", create_constraint=True, create_type=False),
        nullable=False,
    )

    # ── Recovery ────────────────────────────────────────────────
    retraining_job_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        comment="Celery task ID for forensic trace to retrain job",
    )

    triggered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        index=True,
    )

    def __repr__(self) -> str:
        return (
            f"<DriftEvent signal={self.drift_signal} "
            f"{self.policy_from}->{self.policy_to} "
            f"psi={self.psi_score} delta={self.reward_delta}>"
        )
