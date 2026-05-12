"""
Incident Management API — alert→acknowledge→mitigate→resolve→postmortem.

ENDPOINTS:
  POST /api/v1/incidents/open          → open a new incident
  GET  /api/v1/incidents/               → list open incidents
  GET  /api/v1/incidents/{id}          → get incident details
  POST /api/v1/incidents/{id}/acknowledge → acknowledge (assign to self)
  POST /api/v1/incidents/{id}/mitigate  → record mitigation action
  POST /api/v1/incidents/{id}/resolve   → mark resolved
  GET  /api/v1/incidents/{id}/timeline  → get incident event timeline
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import CurrentUser, Role
from app.dependencies.auth import require_role
from app.dependencies.db import get_db
from app.models.incident import Incident, IncidentStatus

router = APIRouter(prefix="/api/v1/incidents", tags=["incidents"])
logger = structlog.get_logger(__name__)


def _incident_to_dict(inc: Incident) -> dict[str, Any]:
    return {
        "id": str(inc.id),
        "title": inc.title,
        "severity": inc.severity,
        "status": inc.status,
        "trigger_type": inc.trigger_type,
        "trigger_detail": inc.trigger_detail,
        "acknowledged_by": inc.acknowledged_by,
        "acknowledged_at": inc.acknowledged_at.isoformat() if inc.acknowledged_at else None,
        "mitigation_action": inc.mitigation_action,
        "mitigated_by": inc.mitigated_by,
        "mitigated_at": inc.mitigated_at.isoformat() if inc.mitigated_at else None,
        "resolved_by": inc.resolved_by,
        "resolved_at": inc.resolved_at.isoformat() if inc.resolved_at else None,
        "resolution_note": inc.resolution_note,
        "postmortem_id": inc.postmortem_id,
        "postmortem_complete": inc.postmortem_complete,
        "timeline": inc.timeline or [],
        "created_at": inc.created_at.isoformat(),
    }


def _append_timeline(inc: Incident, actor: str, action: str, note: str) -> None:
    """Append an event to the incident timeline (immutable append)."""
    timeline = list(inc.timeline or [])
    timeline.append(
        {
            "ts": datetime.now(UTC).isoformat(),
            "actor": actor,
            "action": action,
            "note": note,
        }
    )
    inc.timeline = timeline


@router.post("/open")
async def open_incident(
    title: str = Body(..., embed=True),
    severity: str = Body(default="P2", embed=True),
    trigger_type: str = Body(default="MANUAL", embed=True),
    trigger_detail: dict | None = Body(default=None, embed=True),  # type: ignore[type-arg]
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> dict[str, Any]:
    """Open a new incident. Auto-triggers on drift rollback via Celery."""
    inc = Incident(
        title=title,
        severity=severity,
        status=IncidentStatus.OPEN.value,
        trigger_type=trigger_type,
        trigger_detail=trigger_detail,
        timeline=[
            {
                "ts": datetime.now(UTC).isoformat(),
                "actor": user.user_id,
                "action": "OPENED",
                "note": f"Incident opened by {user.user_id}",
            }
        ],
    )
    db.add(inc)
    await db.commit()
    await db.refresh(inc)
    logger.warning("incident_opened", incident_id=str(inc.id), severity=severity, title=title)
    return _incident_to_dict(inc)


@router.get("/")
async def list_incidents(
    status: str | None = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_role(Role.VIEWER)),
) -> dict[str, Any]:
    q = select(Incident).order_by(Incident.created_at.desc()).limit(limit)
    if status:
        q = q.where(Incident.status == status)
    result = await db.execute(q)
    incidents = result.scalars().all()
    return {"total": len(incidents), "incidents": [_incident_to_dict(i) for i in incidents]}


@router.get("/{incident_id}")
async def get_incident(
    incident_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_role(Role.VIEWER)),
) -> dict[str, Any]:
    result = await db.execute(select(Incident).where(Incident.id == incident_id))
    inc = result.scalar_one_or_none()
    if inc is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    return _incident_to_dict(inc)


@router.post("/{incident_id}/acknowledge")
async def acknowledge_incident(
    incident_id: uuid.UUID,
    note: str = Body(default="", embed=True),
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> dict[str, Any]:
    result = await db.execute(select(Incident).where(Incident.id == incident_id))
    inc = result.scalar_one_or_none()
    if inc is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    if inc.status != IncidentStatus.OPEN.value:
        raise HTTPException(status_code=400, detail=f"Incident is {inc.status}, not OPEN")
    inc.status = IncidentStatus.ACKNOWLEDGED.value
    inc.acknowledged_by = user.user_id
    inc.acknowledged_at = datetime.now(UTC)
    _append_timeline(inc, user.user_id, "ACKNOWLEDGED", note or "Incident acknowledged")
    await db.commit()
    return _incident_to_dict(inc)


@router.post("/{incident_id}/mitigate")
async def mitigate_incident(
    incident_id: uuid.UUID,
    mitigation_action: str = Body(..., embed=True),
    mitigation_detail: str = Body(default="", embed=True),
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> dict[str, Any]:
    result = await db.execute(select(Incident).where(Incident.id == incident_id))
    inc = result.scalar_one_or_none()
    if inc is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    inc.status = IncidentStatus.MITIGATING.value
    inc.mitigation_action = mitigation_action
    inc.mitigation_detail = mitigation_detail
    inc.mitigated_by = user.user_id
    inc.mitigated_at = datetime.now(UTC)
    _append_timeline(inc, user.user_id, "MITIGATING", f"Action: {mitigation_action}")
    await db.commit()
    return _incident_to_dict(inc)


@router.post("/{incident_id}/resolve")
async def resolve_incident(
    incident_id: uuid.UUID,
    resolution_note: str = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> dict[str, Any]:
    result = await db.execute(select(Incident).where(Incident.id == incident_id))
    inc = result.scalar_one_or_none()
    if inc is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    inc.status = IncidentStatus.RESOLVED.value
    inc.resolved_by = user.user_id
    inc.resolved_at = datetime.now(UTC)
    inc.resolution_note = resolution_note
    _append_timeline(inc, user.user_id, "RESOLVED", resolution_note)
    await db.commit()
    logger.info("incident_resolved", incident_id=str(incident_id), by=user.user_id)
    return _incident_to_dict(inc)
