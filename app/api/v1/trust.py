"""
Trust Score API — policy health and trust metrics.

ENDPOINTS:
  GET /api/v1/trust/{policy_type}          → current trust score (cached)
  POST /api/v1/trust/{policy_type}/refresh → recompute trust score now
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import CurrentUser, Role
from app.dependencies.auth import require_role
from app.dependencies.db import get_db
from app.schemas.common import PolicyType
from app.trust.policy_trust_score import TrustScoreComputer

router = APIRouter(prefix="/api/v1/trust", tags=["trust"])
logger = structlog.get_logger(__name__)

_computer = TrustScoreComputer(window_hours=2)


@router.get("/{policy_type}")
async def get_trust_score(
    policy_type: PolicyType,
    user: CurrentUser = Depends(require_role(Role.VIEWER)),
) -> dict[str, Any]:
    """Return cached trust score for a policy. Updated every 60s by Celery beat."""
    score = await _computer.get_cached_score(policy_type)
    if score is None:
        return {
            "policy_type": policy_type.value,
            "status": "not_computed_yet",
            "message": "Trust score is computed every 60s. Check back shortly.",
        }
    return score.to_dict()


@router.post("/{policy_type}/refresh")
async def refresh_trust_score(
    policy_type: PolicyType,
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> dict[str, Any]:
    """Force recompute trust score now (bypasses 60s cache)."""
    score = await _computer.compute(policy_type=policy_type, db=db)
    return score.to_dict()
