"""
RewardLog — tracks reward outcomes for every decision.

WHY THIS EXISTS:
- Reward arrives AFTER the decision (async feedback loop).
- n_step_reward stores the discounted n-step return (v3 temporal credit fix).
- baseline_reward enables drift comparison: active vs shadow baseline.
- cumulative_regret tracks bandit performance vs oracle.

WHAT BREAKS IF WRONG:
- Reward in DecisionLog only = no separate analysis of reward distributions.
- No baseline_reward = drift detector has no reference to compare against.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class RewardLog(Base):
    """Reward outcome for a single decision."""

    __tablename__ = "reward_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    decision_log_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("decision_logs.id"),
        nullable=False,
        index=True,
    )

    reward: Mapped[float] = mapped_column(Float, nullable=False)
    n_step_reward: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="n-step discounted return (v3 temporal credit fix)",
    )
    cumulative_reward: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Running total since policy activation",
    )
    cumulative_regret: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="vs oracle baseline for bandit evaluation",
    )
    baseline_reward: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Shadow baseline reward at same timestep for drift comparison",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    def __repr__(self) -> str:
        return f"<RewardLog r={self.reward:.3f} n_step={self.n_step_reward}>"
