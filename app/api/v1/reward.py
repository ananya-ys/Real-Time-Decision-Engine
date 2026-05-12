"""
Reward submission endpoint — POST /api/v1/rewards.

WHY THIS EXISTS:
- Reward arrives ASYNCHRONOUSLY after the decision is committed.
- Environment responds N ticks later.
- Endpoint receives reward feedback for a historical decision_log_id.

FIXES INCLUDED:
- Proper DB rollback on failure
- Proper HTTPException handling
- FK validation handling
- No false-positive 201 responses
- Cleaner production-safe transaction flow
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies.db import get_db
from app.schemas.reward import RewardCreate

router = APIRouter(prefix="/api/v1", tags=["rewards"])
logger = structlog.get_logger(__name__)


@router.post("/rewards", status_code=201)
async def submit_reward(
    body: RewardCreate,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Submit reward feedback for a past decision.

    Called asynchronously after the environment observes
    the result of a scaling action.
    """

    logger.info(
        "reward_received",
        decision_log_id=str(body.decision_log_id),
        reward=body.reward,
    )

    from app.services.reward_service import RewardComponents, RewardService

    svc = RewardService()

    # ------------------------------------------------------------------
    # Compute reward components
    # ------------------------------------------------------------------

    components = svc.compute_reward(
        p99_latency_ms=(
            body.p99_latency_ms
            if hasattr(body, "p99_latency_ms")
            else 200.0
        ),
        instance_count=(
            body.instance_count
            if hasattr(body, "instance_count")
            else 5
        ),
        last_action_delta=(
            body.action_delta
            if hasattr(body, "action_delta")
            else 0
        ),
    )

    # If environment already computed reward,
    # override computed reward components.
    if body.reward is not None:
        components = RewardComponents(
            latency_penalty=0.0,
            cost_penalty=0.0,
            sla_violation_penalty=0.0,
            instability_penalty=0.0,
            total_reward=body.reward,
            sla_violated=body.reward < -1.0,
        )

    n_step = svc.compute_n_step_reward(
        components.total_reward
    )

    # ------------------------------------------------------------------
    # Persist reward safely
    # ------------------------------------------------------------------

    try:
        reward_log = await svc.log_reward(
            decision_log_id=body.decision_log_id,
            reward_components=components,
            n_step_reward=n_step,
            cumulative_reward=components.total_reward,
            cumulative_regret=None,
            baseline_reward=None,
            db=db,
        )

        await db.commit()
        await db.refresh(reward_log)

        logger.info(
            "reward_logged_successfully",
            reward_log_id=str(reward_log.id),
            decision_log_id=str(body.decision_log_id),
        )

        return {
            "status": "accepted",
            "decision_log_id": str(body.decision_log_id),
            "reward_log_id": str(reward_log.id),
            "reward": components.total_reward,
            "sla_violated": components.sla_violated,
        }

    # --------------------------------------------------------------
    # FK violation / integrity problems
    # --------------------------------------------------------------

    except IntegrityError as exc:
        await db.rollback()

        logger.error(
            "reward_integrity_error",
            error=str(exc),
            decision_log_id=str(body.decision_log_id),
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid decision_log_id. "
                "Referenced decision does not exist."
            ),
        )

    # --------------------------------------------------------------
    # Unexpected system failures
    # --------------------------------------------------------------

    except Exception as exc:
        await db.rollback()

        logger.exception(
            "reward_log_failed",
            error=str(exc),
            decision_log_id=str(body.decision_log_id),
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to persist reward log.",
        )