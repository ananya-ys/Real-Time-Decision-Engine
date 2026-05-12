"""
Operator Control API — kill switch, overrides, freeze controls.

RBAC: All endpoints require OPERATOR or ADMIN role.
Every action writes an OperatorEvent audit entry.

ENDPOINTS:
  POST /api/v1/operator/kill-switch/activate       → global kill
  POST /api/v1/operator/kill-switch/deactivate     → restore
  POST /api/v1/operator/kill-switch/policy/{type}  → kill specific policy
  POST /api/v1/operator/override/force-baseline    → force baseline
  POST /api/v1/operator/override/release-baseline  → restore ML
  POST /api/v1/operator/maintenance/enter          → maintenance mode
  POST /api/v1/operator/maintenance/exit           → exit maintenance
  POST /api/v1/operator/exploration/freeze         → freeze exploration
  POST /api/v1/operator/exploration/unfreeze       → restore exploration
  GET  /api/v1/operator/status                     → full operator status
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Body, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.hash_chain import AuditHashChain
from app.core.auth import CurrentUser
from app.core.rbac import Role
from app.dependencies.auth import require_role
from app.dependencies.db import get_db
from app.models.operator_event import OperatorEvent
from app.operator.kill_switch import KillSwitch
from app.operator.manual_override import ManualOverride
from app.schemas.common import PolicyType

router = APIRouter(prefix="/api/v1/operator", tags=["operator"])
logger = structlog.get_logger(__name__)

_kill_switch = KillSwitch()
_override = ManualOverride()

# ── Auth: real JWT, Role.OPERATOR minimum ────────────────────────────────────
# Dev mode (APP_ENV=development) returns ADMIN — see app/dependencies/auth.py


# ── Kill Switch ───────────────────────────────────────────────────────────────


@router.post("/kill-switch/activate")
async def activate_global_kill_switch(
    reason: str = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> dict[str, Any]:
    """
    EMERGENCY: Disable all ML policies. Baseline serves all traffic.
    Requires: OPERATOR role minimum.
    """
    actor = current_user.user_id
    await _kill_switch.activate_global(actor=actor, reason=reason)

    await _log_operator_event(
        db=db,
        actor=actor,
        actor_role=current_user.role.value,
        action="KILL_SWITCH_GLOBAL_ACTIVATE",
        reason=reason,
    )

    logger.critical("operator_kill_switch_global", actor=actor, reason=reason)
    return {
        "status": "kill_switch_activated",
        "effect": "All ML policies disabled. Baseline serves all traffic.",
        "actor": actor,
        "reason": reason,
    }


@router.post("/kill-switch/deactivate")
async def deactivate_global_kill_switch(
    reason: str = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> dict[str, Any]:
    """Re-enable ML policies after incident resolved."""
    actor = current_user.user_id
    await _kill_switch.deactivate_global(actor=actor, reason=reason)

    await _log_operator_event(
        db=db,
        actor=actor,
        actor_role=current_user.role.value,
        action="KILL_SWITCH_GLOBAL_DEACTIVATE",
        reason=reason,
    )

    return {"status": "kill_switch_deactivated", "actor": actor}


@router.post("/kill-switch/policy/{policy_type}")
async def kill_policy(
    policy_type: PolicyType,
    reason: str = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> dict[str, Any]:
    """Disable a specific policy type (RL, BANDIT, or BASELINE)."""
    actor = current_user.user_id
    await _kill_switch.kill_policy(policy_type=policy_type, actor=actor, reason=reason)

    await _log_operator_event(
        db=db,
        actor=actor,
        actor_role=current_user.role.value,
        action="KILL_SWITCH_POLICY_ACTIVATE",
        target=f"policy_type={policy_type.value}",
        reason=reason,
    )

    return {
        "status": f"{policy_type.value}_policy_killed",
        "actor": actor,
        "reason": reason,
    }


@router.post("/kill-switch/policy/{policy_type}/restore")
async def restore_policy(
    policy_type: PolicyType,
    reason: str = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> dict[str, Any]:
    """Re-enable a specific policy type."""
    actor = current_user.user_id
    await _kill_switch.restore_policy(policy_type=policy_type, actor=actor, reason=reason)

    await _log_operator_event(
        db=db,
        actor=actor,
        actor_role=current_user.role.value,
        action="KILL_SWITCH_POLICY_RESTORE",
        target=f"policy_type={policy_type.value}",
        reason=reason,
    )

    return {"status": f"{policy_type.value}_policy_restored", "actor": actor}


# ── Manual Override ───────────────────────────────────────────────────────────


@router.post("/override/force-baseline")
async def force_baseline(
    reason: str = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> dict[str, Any]:
    """Force baseline serving. ML policy preserves training state."""
    actor = current_user.user_id
    result = await _override.force_baseline(actor=actor, reason=reason)

    await _log_operator_event(
        db=db,
        actor=actor,
        actor_role=current_user.role.value,
        action="OVERRIDE_FORCE_BASELINE",
        reason=reason,
    )

    return result


@router.post("/override/release-baseline")
async def release_baseline(
    reason: str = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> dict[str, Any]:
    """Return to normal ML policy serving."""
    actor = current_user.user_id
    result = await _override.release_baseline(actor=actor, reason=reason)

    await _log_operator_event(
        db=db,
        actor=actor,
        actor_role=current_user.role.value,
        action="OVERRIDE_RELEASE_BASELINE",
        reason=reason,
    )

    return result


# ── Maintenance Mode ──────────────────────────────────────────────────────────


@router.post("/maintenance/enter")
async def enter_maintenance(
    reason: str = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> dict[str, Any]:
    """Enter maintenance mode: force baseline + freeze exploration + freeze promotion."""
    actor = current_user.user_id
    result = await _override.enter_maintenance_mode(actor=actor, reason=reason)

    await _log_operator_event(
        db=db,
        actor=actor,
        actor_role=current_user.role.value,
        action="MAINTENANCE_MODE_ENTER",
        reason=reason,
        state_after=result,
    )

    return result


@router.post("/maintenance/exit")
async def exit_maintenance(
    reason: str = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> dict[str, Any]:
    """Exit maintenance mode — restore full ML operations."""
    actor = current_user.user_id
    result = await _override.exit_maintenance_mode(actor=actor, reason=reason)

    await _log_operator_event(
        db=db,
        actor=actor,
        actor_role=current_user.role.value,
        action="MAINTENANCE_MODE_EXIT",
        reason=reason,
    )

    return result


# ── Exploration Controls ──────────────────────────────────────────────────────


@router.post("/exploration/freeze")
async def freeze_exploration(
    reason: str = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> dict[str, Any]:
    """Freeze all exploration immediately. Policies exploit only."""
    actor = current_user.user_id
    kill_switch = KillSwitch()
    await kill_switch.freeze_exploration(actor=actor, reason=reason)

    await _log_operator_event(
        db=db,
        actor=actor,
        actor_role=current_user.role.value,
        action="EXPLORATION_FREEZE",
        reason=reason,
    )

    return {"status": "exploration_frozen", "actor": actor, "reason": reason}


@router.post("/exploration/unfreeze")
async def unfreeze_exploration(
    reason: str = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> dict[str, Any]:
    """Re-enable exploration."""
    actor = current_user.user_id
    kill_switch = KillSwitch()
    await kill_switch.unfreeze_exploration(actor=actor, reason=reason)

    await _log_operator_event(
        db=db,
        actor=actor,
        actor_role=current_user.role.value,
        action="EXPLORATION_UNFREEZE",
        reason=reason,
    )

    return {"status": "exploration_unfrozen", "actor": actor}


# ── Status ────────────────────────────────────────────────────────────────────


@router.get("/status")
async def get_operator_status() -> dict[str, Any]:
    """Return full operator control status for dashboard."""
    override_status = await _override.get_override_status()
    return {
        "timestamp": __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .isoformat(),
        **override_status,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _log_operator_event(
    db: AsyncSession,
    actor: str,
    actor_role: str,
    action: str,
    reason: str,
    target: str | None = None,
    state_before: dict | None = None,  # type: ignore[type-arg]
    state_after: dict | None = None,  # type: ignore[type-arg]
) -> None:
    """Write immutable operator action to audit log."""
    event = OperatorEvent(
        actor=actor,
        actor_role=actor_role,
        action=action,
        target=target,
        reason=reason,
        state_before=state_before,
        state_after=state_after,
        success=True,
    )
    db.add(event)
    await db.flush()  # get ID + created_at set by Python default

    # Stamp with cryptographic hash chain — links this event to all previous events.
    # Without this, chain_hash = NULL and the immutable audit trail is broken.
    chain = AuditHashChain()
    await chain.stamp(event, db)

    await db.commit()
