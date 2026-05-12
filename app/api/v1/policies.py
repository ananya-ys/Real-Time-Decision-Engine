"""
Policy management API — Phase 4.

Endpoints:
  GET  /api/v1/policies/active       — currently active policy info
  GET  /api/v1/policies/versions     — all versions in registry
  POST /api/v1/policies/checkpoint   — trigger checkpoint save

Router ONLY. All logic in PolicyService. Zero business logic here.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies.db import get_db
from app.models.policy_version import PolicyVersion
from app.schemas.common import PolicyStatus, PolicyType

router = APIRouter(prefix="/api/v1/policies", tags=["policies"])
logger = structlog.get_logger(__name__)


@router.get("/active")
async def get_active_policy(
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Return the currently active policy version."""
    result = await db.execute(
        select(PolicyVersion)
        .where(PolicyVersion.status == PolicyStatus.ACTIVE.value)
        .order_by(PolicyVersion.promoted_at.desc())
        .limit(5)
    )
    versions = result.scalars().all()

    return {
        "active_policies": [
            {
                "id": str(v.id),
                "policy_type": v.policy_type,
                "version": v.version,
                "algorithm": v.algorithm,
                "status": v.status,
                "eval_reward_mean": v.eval_reward_mean,
                "promoted_at": v.promoted_at.isoformat() if v.promoted_at else None,
            }
            for v in versions
        ]
    }


@router.get("/versions")
async def list_policy_versions(
    policy_type: PolicyType | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """List all policy versions in the registry."""
    query = select(PolicyVersion).order_by(PolicyVersion.created_at.desc()).limit(50)
    if policy_type is not None:
        query = query.where(PolicyVersion.policy_type == policy_type.value)

    result = await db.execute(query)
    versions = result.scalars().all()

    return {
        "total": len(versions),
        "versions": [
            {
                "id": str(v.id),
                "policy_type": v.policy_type,
                "version": v.version,
                "algorithm": v.algorithm,
                "status": v.status,
                "eval_reward_mean": v.eval_reward_mean,
                "eval_seeds": v.eval_seeds,
                "created_at": v.created_at.isoformat(),
            }
            for v in versions
        ],
    }
