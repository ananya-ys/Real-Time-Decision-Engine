"""
Audit API — decision replay, incident timelines, postmortems.

ENDPOINTS:
  GET  /api/v1/audit/decisions/{id}/replay    → full replay frame for one decision
  GET  /api/v1/audit/incidents/timeline       → timeline for time window
  GET  /api/v1/audit/postmortems/{drift_id}   → auto-generated postmortem
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.replay_engine import DecisionReplayEngine
from app.audit.timeline_builder import IncidentTimelineBuilder
from app.core.auth import CurrentUser, Role
from app.dependencies.auth import require_role
from app.dependencies.db import get_db
from app.workflow.postmortem import PostmortemGenerator

router = APIRouter(prefix="/api/v1/audit", tags=["audit"])
logger = structlog.get_logger(__name__)

_replay_engine = DecisionReplayEngine()
_timeline_builder = IncidentTimelineBuilder()
_postmortem_gen = PostmortemGenerator()


@router.get("/decisions/{decision_id}/replay")
async def replay_decision(
    decision_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_role(Role.VIEWER)),
) -> dict[str, Any]:
    """
    Replay a past decision — reconstruct exact state, policy, reward, drift context.
    The time-travel debugger. Resolves in < 500ms.
    """
    import time

    start = time.perf_counter()
    try:
        frame = await _replay_engine.replay(decision_id, db)
    except ValueError as exc:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=str(exc)) from exc

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    result = frame.to_dict()
    result["replay_latency_ms"] = latency_ms

    logger.info(
        "decision_replayed_via_api",
        decision_id=str(decision_id),
        latency_ms=latency_ms,
        actor=user.user_id,
    )
    return result


@router.get("/incidents/timeline")
async def get_incident_timeline(
    window_hours: int = Query(default=2, ge=1, le=48),
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_role(Role.VIEWER)),
) -> dict[str, Any]:
    """
    Build incident timeline for the given window.
    Covers: decisions, drift events, operator actions, policy changes.
    """
    timeline = await _timeline_builder.build(db=db, window_hours=window_hours)
    return timeline.to_dict()


@router.get("/postmortems/{drift_event_id}")
async def get_postmortem(
    drift_event_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_role(Role.VIEWER)),
) -> dict[str, Any]:
    """
    Get auto-generated postmortem for a drift/rollback event.
    Returns structured JSON + Markdown text.
    """
    try:
        report = await _postmortem_gen.generate(drift_event_id, db)
    except ValueError as exc:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=str(exc)) from exc

    result = report.to_dict()
    result["markdown"] = report.to_markdown()
    return result
