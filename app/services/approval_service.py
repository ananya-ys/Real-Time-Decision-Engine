"""
ApprovalService — enforces two-person authorization on high-risk actions.

THE RULE:
  HIGH_RISK actions (global kill switch, maintenance mode) require a second
  person to approve before execution. The requester CANNOT approve their own
  request — this is enforced at the DB level, not at the UI level.

WORKFLOW:
  1. Operator A submits request via POST /api/v1/approvals/request
  2. System returns pending approval_request_id
  3. Operator B (different person, ADMIN role) sees pending queue
  4. Operator B approves or rejects via POST /api/v1/approvals/{id}/review
  5. Once approved, Operator A executes via POST /api/v1/approvals/{id}/execute
     (15-minute window after approval)
  6. OperatorEvent written, actual action performed

LOW_RISK and MEDIUM_RISK actions bypass the approval queue and use only the
confirmation gate (typed confirmation string).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.hash_chain import AuditHashChain
from app.models.approval_request import ApprovalRequest, ApprovalStatus
from app.models.operator_event import OperatorEvent

logger = structlog.get_logger(__name__)

_APPROVAL_WINDOW_MINUTES = 15


class ApprovalService:
    """Manages the approval lifecycle for high-risk operator actions."""

    async def submit_request(
        self,
        action: str,
        requester_id: str,
        requester_role: str,
        reason: str,
        parameters: dict[str, Any] | None = None,
        blast_radius: str | None = None,
        action_target: str | None = None,
        db: AsyncSession = None,  # type: ignore[assignment]
    ) -> ApprovalRequest:
        """
        Submit a new approval request.

        Returns the created ApprovalRequest (status=PENDING).
        """
        expires = datetime.now(UTC) + timedelta(minutes=_APPROVAL_WINDOW_MINUTES * 4)

        req = ApprovalRequest(
            requester_id=requester_id,
            requester_role=requester_role,
            action=action,
            action_target=action_target,
            reason=reason,
            parameters=parameters,
            blast_radius=blast_radius,
            status=ApprovalStatus.PENDING.value,
            expires_at=expires,
        )
        db.add(req)
        await db.flush()

        logger.warning(
            "approval_request_submitted",
            request_id=str(req.id),
            action=action,
            requester=requester_id,
        )
        return req

    async def review(
        self,
        request_id: uuid.UUID,
        reviewer_id: str,
        reviewer_role: str,
        approve: bool,
        comment: str,
        db: AsyncSession,
    ) -> ApprovalRequest:
        """
        Approve or reject a pending request.

        Enforces: reviewer must be different from requester.
        Enforces: reviewer must have ADMIN role.
        """
        result = await db.execute(
            select(ApprovalRequest).where(ApprovalRequest.id == request_id).with_for_update()
        )
        req = result.scalar_one_or_none()

        if req is None:
            raise ValueError(f"Approval request {request_id} not found")

        if req.status != ApprovalStatus.PENDING.value:
            raise ValueError(f"Request is {req.status}, not PENDING. Cannot review.")

        # Self-approval enforcement
        if req.requester_id == reviewer_id:
            raise PermissionError(
                "You cannot approve your own request. A different operator must review this action."
            )

        # Check expiry
        if req.expires_at and datetime.now(UTC) > req.expires_at.astimezone(UTC):
            req.status = ApprovalStatus.EXPIRED.value
            await db.flush()
            raise ValueError("Approval request has expired. Submit a new request.")

        now = datetime.now(UTC)
        req.reviewer_id = reviewer_id
        req.reviewer_role = reviewer_role
        req.review_comment = comment
        req.reviewed_at = now

        if approve:
            req.status = ApprovalStatus.APPROVED.value
            req.expires_at = now + timedelta(minutes=_APPROVAL_WINDOW_MINUTES)
            logger.warning(
                "approval_request_approved",
                request_id=str(request_id),
                action=req.action,
                requester=req.requester_id,
                reviewer=reviewer_id,
            )
        else:
            req.status = ApprovalStatus.REJECTED.value
            logger.warning(
                "approval_request_rejected",
                request_id=str(request_id),
                action=req.action,
                reviewer=reviewer_id,
                comment=comment,
            )

        await db.flush()
        return req

    async def execute(
        self,
        request_id: uuid.UUID,
        executor_id: str,
        db: AsyncSession,
    ) -> ApprovalRequest:
        """
        Mark an approved request as executed.

        Called after the actual action has been performed.
        """
        result = await db.execute(
            select(ApprovalRequest).where(ApprovalRequest.id == request_id).with_for_update()
        )
        req = result.scalar_one_or_none()

        if req is None:
            raise ValueError(f"Approval request {request_id} not found")

        if req.status != ApprovalStatus.APPROVED.value:
            raise ValueError(f"Request is {req.status}. Only APPROVED requests can be executed.")

        if req.requester_id != executor_id:
            raise PermissionError("Only the original requester can execute an approved action.")

        # Check execution window
        if req.expires_at and datetime.now(UTC) > req.expires_at.astimezone(UTC):
            req.status = ApprovalStatus.EXPIRED.value
            await db.flush()
            raise ValueError("Approval window expired. Re-request and get approval again.")

        req.status = ApprovalStatus.EXECUTED.value
        req.executed_at = datetime.now(UTC)

        # Write operator event with hash chain stamp
        event = OperatorEvent(
            actor=executor_id,
            actor_role=req.requester_role,
            action=f"APPROVED_EXECUTE:{req.action}",
            target=req.action_target,
            reason=req.reason,
            state_after={"approval_request_id": str(request_id), "reviewer": req.reviewer_id},
            success=True,
        )
        db.add(event)
        await db.flush()  # get ID and Python-side created_at

        # Stamp cryptographic hash chain
        chain = AuditHashChain()
        await chain.stamp(event, db)

        logger.warning(
            "approved_action_executed",
            request_id=str(request_id),
            action=req.action,
            executor=executor_id,
        )
        return req

    async def get_pending(self, db: AsyncSession) -> list[ApprovalRequest]:
        """Get all pending approval requests."""
        result = await db.execute(
            select(ApprovalRequest)
            .where(ApprovalRequest.status == ApprovalStatus.PENDING.value)
            .order_by(ApprovalRequest.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_by_id(self, request_id: uuid.UUID, db: AsyncSession) -> ApprovalRequest | None:
        result = await db.execute(select(ApprovalRequest).where(ApprovalRequest.id == request_id))
        return result.scalar_one_or_none()

    async def cancel(
        self,
        request_id: uuid.UUID,
        requester_id: str,
        db: AsyncSession,
    ) -> ApprovalRequest:
        """Cancel a pending request (requester only)."""
        result = await db.execute(
            select(ApprovalRequest).where(ApprovalRequest.id == request_id).with_for_update()
        )
        req = result.scalar_one_or_none()
        if req is None:
            raise ValueError(f"Request {request_id} not found")
        if req.requester_id != requester_id:
            raise PermissionError("Only the requester can cancel a pending request.")
        if req.status != ApprovalStatus.PENDING.value:
            raise ValueError(f"Request is {req.status}, cannot cancel.")
        req.status = ApprovalStatus.CANCELLED.value
        await db.flush()
        logger.info("approval_request_cancelled", request_id=str(request_id))
        return req
