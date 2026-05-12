"""
Approval Queue API — two-person authorization workflow.

ENDPOINTS:
  POST /api/v1/approvals/request         → submit a high-risk action for approval
  GET  /api/v1/approvals/pending         → list pending requests (for reviewers)
  POST /api/v1/approvals/{id}/approve    → approve a pending request (ADMIN, different person)
  POST /api/v1/approvals/{id}/reject     → reject a pending request
  POST /api/v1/approvals/{id}/execute    → execute an approved request (original requester)
  POST /api/v1/approvals/{id}/cancel     → cancel a pending request (original requester)
  GET  /api/v1/approvals/{id}            → get request details

CONFIRMATION GATE:
  POST /api/v1/operator/confirm/challenge  → issue typed confirmation challenge
  POST /api/v1/operator/confirm/validate   → validate typed challenge
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog
from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import CurrentUser, Role
from app.dependencies.auth import require_role
from app.dependencies.db import get_db
from app.operator.confirmation_gate import HIGH_RISK_ACTIONS, ConfirmationGate
from app.services.approval_service import ApprovalService

router = APIRouter(prefix="/api/v1/approvals", tags=["approvals"])
logger = structlog.get_logger(__name__)

_gate = ConfirmationGate()
_approval_svc = ApprovalService()


# ── Confirmation Gate ──────────────────────────────────────────────────────


@router.post("/confirm/challenge")
async def issue_confirmation_challenge(
    action: str = Body(..., embed=True),
    reason: str = Body(..., embed=True),
    metadata: dict | None = Body(default=None, embed=True),  # type: ignore[type-arg]
    user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> dict[str, Any]:
    """
    Issue a typed confirmation challenge for a destructive action.

    For HIGH_RISK actions, returns challenge + creates an approval request.
    For MEDIUM_RISK, returns challenge only (no approval queue).
    """
    try:
        challenge = await _gate.issue_challenge(
            action=action,
            actor=user.user_id,
            metadata=metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc

    logger.warning(
        "confirmation_challenge_requested",
        action=action,
        actor=user.user_id,
        risk_level=challenge.risk_level,
    )

    return challenge.to_dict()


@router.post("/confirm/validate")
async def validate_confirmation(
    token: str = Body(..., embed=True),
    typed_string: str = Body(..., embed=True),
    user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> dict[str, Any]:
    """Validate a typed confirmation challenge."""
    valid, action, metadata = await _gate.validate_and_consume(
        token=token,
        actor=user.user_id,
        typed_string=typed_string,
    )
    if not valid:
        raise HTTPException(
            status_code=400,
            detail="Invalid confirmation token or challenge string. "
            "Challenges are case-sensitive and expire in 5 minutes.",
        )
    return {
        "confirmed": True,
        "action": action,
        "metadata": metadata,
        "message": (
            "Confirmation validated. "
            + (
                "Submit an approval request to proceed (this action requires second-person authorization)."
                if action in HIGH_RISK_ACTIONS
                else "You may now submit the action with this confirmation."
            )
        ),
    }


# ── Approval Workflow ──────────────────────────────────────────────────────


@router.post("/request")
async def submit_approval_request(
    action: str = Body(..., embed=True),
    reason: str = Body(..., embed=True),
    action_target: str | None = Body(default=None, embed=True),
    parameters: dict | None = Body(default=None, embed=True),  # type: ignore[type-arg]
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> dict[str, Any]:
    """Submit a high-risk action for second-person approval."""
    blast = _gate._blast_radius(action)

    req = await _approval_svc.submit_request(
        action=action,
        requester_id=user.user_id,
        requester_role=user.role.value,
        reason=reason,
        parameters=parameters,
        blast_radius=blast,
        action_target=action_target,
        db=db,
    )
    await db.commit()

    return {
        "request_id": str(req.id),
        "status": req.status,
        "action": req.action,
        "blast_radius": req.blast_radius,
        "expires_at": req.expires_at.isoformat() if req.expires_at else None,
        "next_step": "A different operator with ADMIN role must approve this request.",
    }


@router.get("/pending")
async def list_pending_requests(
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> dict[str, Any]:
    """List all pending approval requests (for the reviewer)."""
    pending = await _approval_svc.get_pending(db)
    return {
        "total": len(pending),
        "requests": [
            {
                "id": str(r.id),
                "action": r.action,
                "action_target": r.action_target,
                "requester": r.requester_id,
                "requester_role": r.requester_role,
                "reason": r.reason,
                "blast_radius": r.blast_radius,
                "created_at": r.created_at.isoformat(),
                "expires_at": r.expires_at.isoformat() if r.expires_at else None,
                "can_review": r.requester_id != user.user_id,
            }
            for r in pending
        ],
    }


@router.get("/{request_id}")
async def get_approval_request(
    request_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_role(Role.VIEWER)),
) -> dict[str, Any]:
    req = await _approval_svc.get_by_id(request_id, db)
    if req is None:
        raise HTTPException(status_code=404, detail="Request not found")
    return {
        "id": str(req.id),
        "action": req.action,
        "action_target": req.action_target,
        "requester": req.requester_id,
        "reason": req.reason,
        "blast_radius": req.blast_radius,
        "status": req.status,
        "reviewer": req.reviewer_id,
        "review_comment": req.review_comment,
        "reviewed_at": req.reviewed_at.isoformat() if req.reviewed_at else None,
        "executed_at": req.executed_at.isoformat() if req.executed_at else None,
        "created_at": req.created_at.isoformat(),
    }


@router.post("/{request_id}/approve")
async def approve_request(
    request_id: uuid.UUID,
    comment: str = Body(default="Approved", embed=True),
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> dict[str, Any]:
    """Approve a pending request. You cannot approve your own request."""
    try:
        req = await _approval_svc.review(
            request_id=request_id,
            reviewer_id=user.user_id,
            reviewer_role=user.role.value,
            approve=True,
            comment=comment,
            db=db,
        )
        await db.commit()
    except (ValueError, PermissionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "request_id": str(req.id),
        "status": req.status,
        "approved_by": req.reviewer_id,
        "next_step": "The original requester may now execute the action within 15 minutes.",
    }


@router.post("/{request_id}/reject")
async def reject_request(
    request_id: uuid.UUID,
    comment: str = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> dict[str, Any]:
    """Reject a pending request."""
    try:
        req = await _approval_svc.review(
            request_id=request_id,
            reviewer_id=user.user_id,
            reviewer_role=user.role.value,
            approve=False,
            comment=comment,
            db=db,
        )
        await db.commit()
    except (ValueError, PermissionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"request_id": str(req.id), "status": req.status, "rejected_by": req.reviewer_id}


@router.post("/{request_id}/execute")
async def execute_approved_action(
    request_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> dict[str, Any]:
    """Execute an approved action (original requester only)."""
    try:
        req = await _approval_svc.execute(
            request_id=request_id,
            executor_id=user.user_id,
            db=db,
        )
        await db.commit()
    except (ValueError, PermissionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "request_id": str(req.id),
        "status": req.status,
        "action": req.action,
        "executed_at": req.executed_at.isoformat() if req.executed_at else None,
        "message": "Action marked as executed. The actual system effect depends on which action was taken.",
    }


@router.post("/{request_id}/cancel")
async def cancel_request(
    request_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> dict[str, Any]:
    """Cancel a pending request (original requester only)."""
    try:
        req = await _approval_svc.cancel(request_id, user.user_id, db)
        await db.commit()
    except (ValueError, PermissionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"request_id": str(req.id), "status": req.status}
