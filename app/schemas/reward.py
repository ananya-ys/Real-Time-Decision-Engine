"""
Pydantic schemas for reward tracking.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RewardCreate(BaseModel):
    """Schema for submitting a reward for a decision."""

    decision_log_id: UUID
    reward: float
    baseline_reward: float | None = None


class RewardLogRead(BaseModel):
    """Schema for reading a reward log entry."""

    id: UUID
    decision_log_id: UUID
    reward: float
    n_step_reward: float | None = None
    cumulative_reward: float | None = None
    cumulative_regret: float | None = None
    baseline_reward: float | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class RewardWeights(BaseModel):
    """Configurable reward function weights for A/B testing."""

    alpha_latency: float = Field(default=1.0, ge=0.0, description="Latency penalty weight")
    beta_cost: float = Field(default=0.5, ge=0.0, description="Cost penalty weight")
    gamma_sla: float = Field(default=2.0, ge=0.0, description="SLA violation penalty weight")
    delta_instability: float = Field(default=0.3, ge=0.0, description="Instability penalty weight")
