"""
Shared types, enums, and base schemas.

WHY THIS EXISTS:
- Enums defined once, imported everywhere. Prevents typo-driven bugs.
- Common schemas (pagination, timestamps) prevent duplication.
- One change here updates all endpoints that use these types.

WHAT BREAKS IF WRONG:
- Enum values scattered as strings = silent mismatches between DB and API.
- Duplicated types = one gets updated, the other doesn't = data corruption.
"""

from __future__ import annotations

import enum
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# ── Domain Enums ────────────────────────────────────────────────


class ActionType(str, enum.Enum):
    """Discrete scaling actions available to all policies."""

    SCALE_UP_1 = "SCALE_UP_1"
    SCALE_UP_3 = "SCALE_UP_3"
    SCALE_DOWN_1 = "SCALE_DOWN_1"
    SCALE_DOWN_3 = "SCALE_DOWN_3"
    HOLD = "HOLD"


class PolicyType(str, enum.Enum):
    """Policy algorithm identifiers."""

    BASELINE = "BASELINE"
    BANDIT = "BANDIT"
    RL = "RL"


class PolicyMode(str, enum.Enum):
    """Whether a policy's decisions are committed or only logged."""

    ACTIVE = "ACTIVE"
    SHADOW = "SHADOW"


class PolicyStatus(str, enum.Enum):
    """Lifecycle status for policy versions in the registry."""

    TRAINING = "TRAINING"
    SHADOW = "SHADOW"
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"


class TrafficRegime(str, enum.Enum):
    """Traffic pattern classification — used by two-signal drift detector."""

    STEADY = "STEADY"
    BURST = "BURST"
    PERIODIC = "PERIODIC"
    UNKNOWN = "UNKNOWN"


class StateSource(str, enum.Enum):
    """Origin of environment state data."""

    REAL = "REAL"
    SIMULATED = "SIMULATED"


class DriftSignal(str, enum.Enum):
    """Which drift detection signal triggered the rollback."""

    REWARD_DEGRADATION = "REWARD_DEGRADATION"
    INPUT_DRIFT = "INPUT_DRIFT"
    BOTH = "BOTH"


class SuppressionReason(str, enum.Enum):
    """Why ExplorationGuard suppressed exploration."""

    HIGH_LATENCY = "HIGH_LATENCY"
    HIGH_LOAD = "HIGH_LOAD"
    SLA_VIOLATION_STREAK = "SLA_VIOLATION_STREAK"
    MANUAL = "MANUAL"


# ── Shared Schemas ──────────────────────────────────────────────


class TimestampMixin(BaseModel):
    """Mixin for created_at timestamps."""

    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PaginationParams(BaseModel):
    """Standard pagination parameters."""

    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=200)

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


class PaginatedResponse(BaseModel):
    """Standard paginated response wrapper."""

    total: int
    page: int
    page_size: int
    items: list[dict]  # type: ignore[type-arg]


class HealthResponse(BaseModel):
    """Health check response shape."""

    status: str
    app: str
    db: str
    redis: str
    worker_broker: str = "unknown"


class ErrorResponse(BaseModel):
    """Standard error response shape."""

    error: str
    message: str


class UUIDResponse(BaseModel):
    """Response containing just an ID."""

    id: UUID
