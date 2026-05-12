"""
DecisionRepository — all DecisionLog DB access in one place.

WHY THIS EXISTS:
- Services should not contain raw SQLAlchemy queries.
- Repository pattern: one file owns all DB queries for one model.
- Easier to test, cache, and optimize queries in a single location.
- Prevents duplicate query logic across services.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.decision_log import DecisionLog


class DecisionRepository:
    """All database access for DecisionLog model."""

    async def create(
        self,
        db: AsyncSession,
        *,
        trace_id: uuid.UUID,
        policy_type: str,
        state_snapshot: dict,  # type: ignore[type-arg]
        action: str,
        q_values: dict | None = None,  # type: ignore[type-arg]
        confidence_spread: float | None = None,
        latency_ms: float | None = None,
        fallback_flag: bool = False,
        shadow_flag: bool = False,
        drift_flag: bool = False,
        policy_version_id: uuid.UUID | None = None,
    ) -> DecisionLog:
        """Persist a new DecisionLog entry."""
        log = DecisionLog(
            trace_id=trace_id,
            policy_type=policy_type,
            policy_version_id=policy_version_id,
            state_snapshot=state_snapshot,
            action=action,
            q_values=q_values,
            confidence_spread=confidence_spread,
            latency_ms=latency_ms,
            fallback_flag=fallback_flag,
            shadow_flag=shadow_flag,
            drift_flag=drift_flag,
        )
        db.add(log)
        await db.flush()
        return log

    async def get_by_id(self, db: AsyncSession, decision_id: uuid.UUID) -> DecisionLog | None:
        result = await db.execute(select(DecisionLog).where(DecisionLog.id == decision_id))
        return result.scalar_one_or_none()

    async def get_recent(
        self,
        db: AsyncSession,
        *,
        limit: int = 50,
        policy_type: str | None = None,
        fallback_only: bool = False,
    ) -> list[DecisionLog]:
        q = select(DecisionLog).order_by(DecisionLog.created_at.desc()).limit(limit)
        if policy_type:
            q = q.where(DecisionLog.policy_type == policy_type)
        if fallback_only:
            q = q.where(DecisionLog.fallback_flag.is_(True))
        result = await db.execute(q)
        return list(result.scalars().all())

    async def count_in_window(
        self,
        db: AsyncSession,
        *,
        window_hours: int = 1,
        fallback_only: bool = False,
        slo_breach_ms: float = 300.0,
    ) -> dict[str, int]:
        since = datetime.now(UTC) - timedelta(hours=window_hours)

        total_q = select(func.count(DecisionLog.id)).where(DecisionLog.created_at >= since)
        fallback_q = select(func.count(DecisionLog.id)).where(
            DecisionLog.created_at >= since,
            DecisionLog.fallback_flag.is_(True),
        )
        breach_q = select(func.count(DecisionLog.id)).where(
            DecisionLog.created_at >= since,
            DecisionLog.latency_ms > slo_breach_ms,
        )

        total = (await db.execute(total_q)).scalar() or 0
        fallbacks = (await db.execute(fallback_q)).scalar() or 0
        breaches = (await db.execute(breach_q)).scalar() or 0

        return {"total": total, "fallbacks": fallbacks, "slo_breaches": breaches}

    async def get_avg_latency(self, db: AsyncSession, *, window_hours: int = 1) -> float | None:
        since = datetime.now(UTC) - timedelta(hours=window_hours)
        result = await db.execute(
            select(func.avg(DecisionLog.latency_ms)).where(DecisionLog.created_at >= since)
        )
        val = result.scalar()
        return float(val) if val is not None else None

    async def update_reward(
        self,
        db: AsyncSession,
        decision_id: uuid.UUID,
        reward: float,
    ) -> None:
        """Populate reward field after environment feedback arrives."""
        result = await db.execute(select(DecisionLog).where(DecisionLog.id == decision_id))
        log = result.scalar_one_or_none()
        if log:
            log.reward = reward
            await db.flush()
