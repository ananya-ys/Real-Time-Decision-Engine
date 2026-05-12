"""
DecisionLog — mandatory audit log for every decision.

WHY THIS EXISTS:
- No decision commits without a corresponding log entry. This is the forensic record.
- JSONB state_snapshot preserves state as-seen by policy (FK alone loses data if state mutates).
- q_values + confidence_spread enable Decision Explainer: "why did it do that?"
- trace_id threads through all layers for end-to-end request correlation.

WHAT BREAKS IF WRONG:
- No state_snapshot = can't reconstruct what the policy saw at decision time.
- No trace_id = debugging requires log archaeology instead of one grep.
- No shadow_flag = shadow decisions mixed with real decisions in analysis.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.schemas.common import ActionType, PolicyType


class DecisionLog(Base):
    """Audit trail for every scaling decision made by any policy."""

    __tablename__ = "decision_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # ── Request Correlation ─────────────────────────────────────
    trace_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="UUID threaded through all layers for tracing",
    )

    # ── Decision Context ────────────────────────────────────────
    policy_type: Mapped[str] = mapped_column(
        Enum(PolicyType, name="policy_type_enum", create_constraint=True, create_type=False),
        nullable=False,
    )
    policy_version_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("policy_versions.id"),
        nullable=True,
        comment="Links decision to exact model version in registry",
    )

    # ── State Snapshot (JSONB) ──────────────────────────────────
    state_snapshot: Mapped[dict] = mapped_column(  # type: ignore[type-arg]
        JSONB,
        nullable=False,
        comment="Full state as-seen by policy at decision time",
    )

    # ── Action Taken ────────────────────────────────────────────
    action: Mapped[str] = mapped_column(
        Enum(ActionType, name="action_type_enum", create_constraint=True, create_type=False),
        nullable=False,
    )

    # ── Decision Explainer Fields ───────────────────────────────
    q_values: Mapped[dict | None] = mapped_column(  # type: ignore[type-arg]
        JSONB,
        nullable=True,
        comment="Q-value vector for all actions (DQN explainer)",
    )
    confidence_spread: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="max_Q - second_Q: how certain was the policy?",
    )

    # ── Outcome (populated async) ───────────────────────────────
    reward: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Populated after environment feedback",
    )

    # ── Performance ─────────────────────────────────────────────
    latency_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Full decision + commit latency",
    )

    # ── Flags ───────────────────────────────────────────────────
    fallback_flag: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        comment="True = policy raised exception, baseline used",
    )
    shadow_flag: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        comment="True = shadow decision, not committed to environment",
    )
    drift_flag: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        comment="True = drift detected at this decision",
    )

    # ── Timestamps ──────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        index=True,
    )

    def __repr__(self) -> str:
        return (
            f"<DecisionLog policy={self.policy_type} action={self.action} "
            f"reward={self.reward} fallback={self.fallback_flag}>"
        )
