"""
Explainability API — deep decision explanation endpoint.

ENDPOINTS:
  POST /api/v1/explain/decision  → explain a decision from current state + metadata
  GET  /api/v1/explain/history/{id}  → explain a past decision from audit log
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.auth import CurrentUser, Role
from app.dependencies.auth import require_role
from app.schemas.state import SystemState
from app.services.explainability_service import ExplainabilityService

router = APIRouter(prefix="/api/v1/explain", tags=["explainability"])

_svc = ExplainabilityService()


class ExplainRequest(BaseModel):
    state: SystemState
    chosen_action: str
    q_values: dict[str, float] | None = None
    explore_allowed: bool = True
    suppression_reason: str | None = None
    policy_type: str = "BASELINE"


@router.post("/decision")
async def explain_decision(
    body: ExplainRequest,
    user: CurrentUser = Depends(require_role(Role.VIEWER)),
) -> dict[str, Any]:
    """
    Generate a full explanation for a decision.
    Can be called with live state or with replayed data from audit logs.
    """
    explanation = await _svc.explain(
        state=body.state,
        chosen_action=body.chosen_action,
        q_values=body.q_values,
        explore_allowed=body.explore_allowed,
        suppression_reason=body.suppression_reason,
        policy_type=body.policy_type,
    )
    return explanation.to_dict()
