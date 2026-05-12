"""
PolicyRepository — DB access for PolicyVersion and PolicyCheckpoint.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.policy_checkpoint import PolicyCheckpoint
from app.models.policy_version import PolicyVersion
from app.schemas.common import PolicyStatus, PolicyType


class PolicyRepository:
    """All database access for policy registry models."""

    async def get_active(self, db: AsyncSession, policy_type: PolicyType) -> PolicyVersion | None:
        result = await db.execute(
            select(PolicyVersion)
            .where(
                PolicyVersion.policy_type == policy_type.value,
                PolicyVersion.status == PolicyStatus.ACTIVE.value,
            )
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def list_all(
        self,
        db: AsyncSession,
        *,
        policy_type: PolicyType | None = None,
        limit: int = 50,
    ) -> list[PolicyVersion]:
        q = select(PolicyVersion).order_by(PolicyVersion.created_at.desc()).limit(limit)
        if policy_type:
            q = q.where(PolicyVersion.policy_type == policy_type.value)
        result = await db.execute(q)
        return list(result.scalars().all())

    async def get_by_id(self, db: AsyncSession, version_id: uuid.UUID) -> PolicyVersion | None:
        result = await db.execute(select(PolicyVersion).where(PolicyVersion.id == version_id))
        return result.scalar_one_or_none()

    async def get_active_checkpoint(
        self, db: AsyncSession, policy_version_id: uuid.UUID
    ) -> PolicyCheckpoint | None:
        result = await db.execute(
            select(PolicyCheckpoint)
            .where(
                PolicyCheckpoint.policy_version_id == policy_version_id,
                PolicyCheckpoint.is_active.is_(True),
            )
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def upsert_checkpoint(
        self,
        db: AsyncSession,
        *,
        policy_version_id: uuid.UUID,
        weights: dict,  # type: ignore[type-arg]
        step_count: int,
        performance_metric: float | None,
    ) -> PolicyCheckpoint:
        # Deactivate existing
        await db.execute(
            update(PolicyCheckpoint)
            .where(
                PolicyCheckpoint.policy_version_id == policy_version_id,
                PolicyCheckpoint.is_active.is_(True),
            )
            .values(is_active=False)
        )
        # Create new
        cp = PolicyCheckpoint(
            policy_version_id=policy_version_id,
            weights=weights,
            step_count=step_count,
            performance_metric=performance_metric,
            is_active=True,
        )
        db.add(cp)
        await db.flush()
        return cp
