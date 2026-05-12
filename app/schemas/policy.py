"""
Pydantic schemas for policy registry and checkpoints.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import PolicyStatus, PolicyType


class PolicyVersionCreate(BaseModel):
    """Schema for creating a new policy version entry."""

    policy_type: PolicyType
    algorithm: str | None = None
    weights_path: str | None = None
    normalizer_path: str | None = None


class PolicyVersionRead(BaseModel):
    """Schema for reading a policy version from the registry."""

    id: UUID
    policy_type: PolicyType
    version: int
    algorithm: str | None = None
    training_run_id: UUID | None = None
    weights_path: str | None = None
    normalizer_path: str | None = None
    eval_reward_mean: float | None = None
    eval_reward_std: float | None = None
    eval_seeds: int | None = None
    status: PolicyStatus
    promoted_at: datetime | None = None
    demoted_at: datetime | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PolicyCheckpointRead(BaseModel):
    """Schema for reading a policy checkpoint."""

    id: UUID
    policy_version_id: UUID
    weights: dict | None = None  # type: ignore[type-arg]
    step_count: int
    performance_metric: float | None = None
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ActivePolicyInfo(BaseModel):
    """Info about the currently active policy — returned by /health."""

    policy_type: PolicyType
    policy_version: int | None = None
    algorithm: str | None = None
    status: PolicyStatus = PolicyStatus.ACTIVE
    eval_reward_mean: float | None = None


class PolicyPromotionRequest(BaseModel):
    """Request to promote a shadow policy to active."""

    policy_version_id: UUID
    min_eval_seeds: int = Field(default=5, ge=1)
    promotion_threshold: float = Field(
        default=0.05, ge=0.0, description="Shadow must beat active by this fraction"
    )
