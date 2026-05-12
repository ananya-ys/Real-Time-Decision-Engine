"""
Canary API — progressive traffic splitting management.

ENDPOINTS:
  POST /api/v1/canary/{policy_type}/start    → start canary at 10%
  GET  /api/v1/canary/{policy_type}/status   → current canary state + metrics
  POST /api/v1/canary/{policy_type}/advance  → advance to next stage (10→25→50→100)
  POST /api/v1/canary/{policy_type}/abort    → abort canary, 0% traffic
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Body, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.canary.canary_router import CanaryRouter
from app.core.auth import CurrentUser, Role
from app.dependencies.auth import require_role
from app.dependencies.db import get_db
from app.models.operator_event import OperatorEvent
from app.schemas.common import PolicyType

router = APIRouter(prefix="/api/v1/canary", tags=["canary"])
logger = structlog.get_logger(__name__)

_canary = CanaryRouter()


@router.post("/{policy_type}/start")
async def start_canary(
    policy_type: PolicyType,
    version_id: str = Body(..., embed=True),
    initial_pct: int = Body(default=10, embed=True),
    reason: str = Body(default="", embed=True),
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> dict[str, Any]:
    """Start canary rollout for a policy at the given traffic percentage."""
    config = await _canary.start_canary(
        policy_type=policy_type,
        version_id=version_id,
        initial_pct=initial_pct,
        actor=user.user_id,
    )

    event = OperatorEvent(
        actor=user.user_id,
        actor_role=user.role.value,
        action="CANARY_START",
        target=f"policy_type={policy_type.value},version={version_id}",
        reason=reason or f"Canary started at {initial_pct}%",
        state_after=config.to_dict(),
        success=True,
    )
    db.add(event)
    await db.commit()

    return config.to_dict()


@router.get("/{policy_type}/status")
async def get_canary_status(
    policy_type: PolicyType,
    user: CurrentUser = Depends(require_role(Role.VIEWER)),
) -> dict[str, Any]:
    """Return current canary configuration and live metrics."""
    return await _canary.get_metrics(policy_type)


@router.post("/{policy_type}/advance")
async def advance_canary(
    policy_type: PolicyType,
    reason: str = Body(default="", embed=True),
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> dict[str, Any]:
    """Advance canary to next traffic stage (10→25→50→100)."""
    # Check auto-abort conditions first
    should_abort, abort_reason = await _canary.should_auto_abort(policy_type)
    if should_abort:
        config = await _canary.abort_canary(
            policy_type=policy_type,
            reason=f"Auto-abort blocked advance: {abort_reason}",
            actor="auto_abort",
        )
        event = OperatorEvent(
            actor="auto_abort",
            actor_role="system",
            action="CANARY_AUTO_ABORT",
            target=f"policy_type={policy_type.value}",
            reason=abort_reason or "",
            success=True,
        )
        db.add(event)
        await db.commit()
        return {"status": "auto_aborted", "reason": abort_reason, "config": config.to_dict()}

    config = await _canary.advance_stage(policy_type=policy_type, actor=user.user_id)
    event = OperatorEvent(
        actor=user.user_id,
        actor_role=user.role.value,
        action="CANARY_ADVANCE",
        target=f"policy_type={policy_type.value}",
        reason=reason or f"Advanced to {config.traffic_pct}%",
        success=True,
    )
    db.add(event)
    await db.commit()
    return config.to_dict()


@router.post("/{policy_type}/abort")
async def abort_canary(
    policy_type: PolicyType,
    reason: str = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> dict[str, Any]:
    """Abort canary — immediately return all traffic to main active policy."""
    config = await _canary.abort_canary(
        policy_type=policy_type,
        reason=reason,
        actor=user.user_id,
    )
    event = OperatorEvent(
        actor=user.user_id,
        actor_role=user.role.value,
        action="CANARY_ABORT",
        target=f"policy_type={policy_type.value}",
        reason=reason,
        success=True,
    )
    db.add(event)
    await db.commit()
    return config.to_dict()
