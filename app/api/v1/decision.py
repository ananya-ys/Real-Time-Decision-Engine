"""
Decision API endpoint — POST /api/v1/decision.

WHY THIS EXISTS:
- Thin HTTP layer only
- Delegates all business logic to DecisionService
- Extracts trace correlation metadata
- Returns structured decision response

PRODUCTION GUARANTEES:
- No business logic in router
- Structured trace logging
- Typed response model
- Safe UUID handling
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies.db import get_db
from app.schemas.decision import DecisionRequest, DecisionResponse
from app.services.decision_service import DecisionService

router = APIRouter(prefix="/api/v1", tags=["decisions"])
logger = structlog.get_logger(__name__)

# ------------------------------------------------------------------
# Singleton service instance
# ------------------------------------------------------------------

_decision_service = DecisionService()


def get_decision_service() -> DecisionService:
    """
    Dependency injection factory.

    Returns:
        Shared DecisionService singleton
    """
    return _decision_service


@router.post(
    "/decision",
    response_model=DecisionResponse,
    status_code=200,
)
async def make_decision(
    body: DecisionRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    service: DecisionService = Depends(get_decision_service),
) -> DecisionResponse:
    """
    Make a scaling decision based on current system state.

    FLOW:
    1. Validate request payload
    2. Extract/generate trace_id
    3. Delegate orchestration to DecisionService
    4. Return typed response

    FAILURE MODES:
    - 422 → invalid payload
    - 500 → internal decision engine failure
    """

    # --------------------------------------------------------------
    # Trace correlation
    # --------------------------------------------------------------

    raw_trace_id = getattr(
        request.state,
        "trace_id",
        str(uuid.uuid4()),
    )

    try:
        trace_id = uuid.UUID(str(raw_trace_id))
    except ValueError:
        logger.warning(
            "invalid_trace_id_received",
            raw_trace_id=raw_trace_id,
        )
        trace_id = uuid.uuid4()

    logger.info(
        "decision_request_received",
        trace_id=str(trace_id),
    )

    # --------------------------------------------------------------
    # Delegate to service layer
    # --------------------------------------------------------------

    try:
        response = await service.make_decision(
            state=body.state,
            trace_id=trace_id,
            db=db,
        )

        logger.info(
            "decision_request_completed",
            trace_id=str(trace_id),
            action=getattr(response, "action", None),
        )

        return response

    except Exception as exc:
        logger.exception(
            "decision_request_failed",
            trace_id=str(trace_id),
            error=str(exc),
        )

        raise HTTPException(
            status_code=500,
            detail="Decision engine failed.",
        )