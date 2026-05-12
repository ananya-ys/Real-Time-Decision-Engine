"""
Pydantic schemas for scaling decisions and decision logs.

WHY THIS EXISTS:
- ScalingDecision is the typed output contract from policies. A in the MDP.
- DecisionRequest wraps SystemState for the /v1/decision endpoint.
- DecisionResponse returns action + metadata + trace_id to the caller.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import ActionType, PolicyMode, PolicyType
from app.schemas.state import SystemState


class ScalingDecision(BaseModel):
    """Output contract from any policy. This is A in the MDP."""

    action: ActionType
    instances_before: int = Field(..., ge=0)
    instances_after: int = Field(..., ge=0)
    policy_type: PolicyType
    policy_mode: PolicyMode = PolicyMode.ACTIVE
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    q_values: dict[str, float] | None = None


class DecisionRequest(BaseModel):
    """Request body for POST /v1/decision."""

    state: SystemState


class DecisionResponse(BaseModel):
    """Response from POST /v1/decision."""

    decision_log_id: UUID
    trace_id: UUID

    action: ActionType
    instances_before: int
    instances_after: int

    policy_type: PolicyType
    policy_mode: PolicyMode

    latency_ms: float

    fallback_used: bool = False

    shadow_decision: ScalingDecision | None = None

    model_config = ConfigDict(from_attributes=True)


class DecisionLogRead(BaseModel):
    """Schema for reading a decision log entry."""

    id: UUID
    trace_id: UUID
    policy_type: PolicyType
    action: ActionType
    state_snapshot: dict  # type: ignore[type-arg]
    q_values: dict | None = None  # type: ignore[type-arg]
    confidence_spread: float | None = None
    reward: float | None = None
    latency_ms: float | None = None
    fallback_flag: bool
    shadow_flag: bool
    drift_flag: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ScalingActionRead(BaseModel):
    """Schema for reading a scaling action record."""

    id: UUID
    action_type: ActionType
    instances_before: int
    instances_after: int
    policy_type: PolicyType
    policy_mode: PolicyMode
    state_id: UUID
    success_flag: bool
    rollback_trigger: bool
    committed_at: datetime

    model_config = ConfigDict(from_attributes=True)