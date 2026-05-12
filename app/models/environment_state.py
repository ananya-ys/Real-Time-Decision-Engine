"""
EnvironmentState — the core observable state at every decision tick.

WHY THIS EXISTS:
- Every policy reads this to decide. It is the S in the MDP.
- version field enables optimistic concurrency (UPDATE WHERE version = expected).
- source enum prevents training on wrong data (REAL vs SIMULATED).
- traffic_regime feeds the two-signal drift detector.

WHAT BREAKS IF WRONG:
- No version field = no optimistic concurrency guard = silent overwrites under load.
- No source field = policy trained on simulated data deployed on real traffic without knowing.
- No traffic_regime = drift detector cannot condition on traffic pattern = false positives.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Enum, Float, Integer
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.schemas.common import StateSource, TrafficRegime


class EnvironmentState(Base):
    """Observable system state at a single decision tick."""

    __tablename__ = "environment_states"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # ── Observable Features (MDP State Space) ───────────────────
    cpu_utilization: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="CPU utilization ratio [0.0, 1.0]",
    )
    request_rate: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Requests per second",
    )
    p99_latency_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="99th percentile latency in milliseconds",
    )
    instance_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Current number of compute instances",
    )
    hour_of_day: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Hour of day [0-23] for temporal features",
    )
    day_of_week: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Day of week [0-6] for temporal features",
    )

    # ── Classification & Metadata ───────────────────────────────
    traffic_regime: Mapped[str] = mapped_column(
        Enum(TrafficRegime, name="traffic_regime_enum", create_constraint=True),
        nullable=False,
        default=TrafficRegime.UNKNOWN,
        comment="Traffic pattern classification for drift detector",
    )
    source: Mapped[str] = mapped_column(
        Enum(StateSource, name="state_source_enum", create_constraint=True),
        nullable=False,
        default=StateSource.SIMULATED,
        comment="Origin of this state data",
    )

    # ── Concurrency Control ─────────────────────────────────────
    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Optimistic concurrency version — UPDATE WHERE version = expected",
    )

    # ── Timestamps ──────────────────────────────────────────────
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        index=True,
        comment="When this state was observed",
    )

    def __repr__(self) -> str:
        return (
            f"<EnvironmentState cpu={self.cpu_utilization:.2f} "
            f"rps={self.request_rate:.0f} p99={self.p99_latency_ms:.0f}ms "
            f"instances={self.instance_count} v={self.version}>"
        )
