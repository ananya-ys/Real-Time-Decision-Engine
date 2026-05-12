"""
Pydantic schemas for EnvironmentState.

WHY THIS EXISTS:
- Separate Create/Read schemas prevent internal fields (id, version) from being settable via API.
- SystemState is the typed contract that policies receive. Validated before any policy reads it.
- Invalid state rejected at API boundary — never passes corrupt state to a policy.

WHAT BREAKS IF WRONG:
- No validation = cpu_pct=2.0 reaches policy = garbage action = SLA breach.
- No Create/Read separation = client can set version field = bypasses concurrency control.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import StateSource, TrafficRegime


class SystemState(BaseModel):
    """The typed state contract that policies receive. This is S in the MDP."""

    cpu_utilization: float = Field(..., ge=0.0, le=1.0, description="CPU ratio [0,1]")
    request_rate: float = Field(..., ge=0.0, description="Requests per second")
    p99_latency_ms: float = Field(..., ge=0.0, description="P99 latency in ms")
    instance_count: int = Field(..., ge=1, description="Current instance count")
    hour_of_day: int = Field(default=0, ge=0, le=23)
    day_of_week: int = Field(default=0, ge=0, le=6)
    traffic_regime: TrafficRegime = TrafficRegime.UNKNOWN

    model_config = ConfigDict(from_attributes=True)

    def to_feature_vector(self) -> list[float]:
        """Convert to flat feature vector for neural network input."""
        return [
            self.cpu_utilization,
            self.request_rate,
            self.p99_latency_ms,
            float(self.instance_count),
            float(self.hour_of_day) / 23.0,  # normalize to [0,1]
            float(self.day_of_week) / 6.0,  # normalize to [0,1]
        ]

    def to_snapshot_dict(self) -> dict[str, float | int | str]:
        """Convert to JSONB-safe dict for DecisionLog.state_snapshot."""
        return {
            "cpu_utilization": self.cpu_utilization,
            "request_rate": self.request_rate,
            "p99_latency_ms": self.p99_latency_ms,
            "instance_count": self.instance_count,
            "hour_of_day": self.hour_of_day,
            "day_of_week": self.day_of_week,
            "traffic_regime": self.traffic_regime.value,
        }


class EnvironmentStateCreate(BaseModel):
    """Schema for creating a new environment state record."""

    cpu_utilization: float = Field(..., ge=0.0, le=1.0)
    request_rate: float = Field(..., ge=0.0)
    p99_latency_ms: float = Field(..., ge=0.0)
    instance_count: int = Field(..., ge=1)
    hour_of_day: int = Field(default=0, ge=0, le=23)
    day_of_week: int = Field(default=0, ge=0, le=6)
    traffic_regime: TrafficRegime = TrafficRegime.UNKNOWN
    source: StateSource = StateSource.SIMULATED


class EnvironmentStateRead(BaseModel):
    """Schema for reading an environment state from the database."""

    id: UUID
    cpu_utilization: float
    request_rate: float
    p99_latency_ms: float
    instance_count: int
    hour_of_day: int
    day_of_week: int
    traffic_regime: TrafficRegime
    source: StateSource
    version: int
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)
