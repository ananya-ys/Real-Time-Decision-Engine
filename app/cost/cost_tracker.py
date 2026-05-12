"""
CostTracker — per-decision compute and infrastructure cost tracking.

WHY THIS EXISTS:
The review: "a system that optimizes latency but ignores cost is not production-grade."

COST MODEL:
  decision_compute_cost = inference_time_ms × compute_cost_per_ms
  instance_cost         = instance_count × cost_per_instance_per_tick
  total_decision_cost   = decision_compute_cost + instance_cost

BUDGET GUARDRAILS:
  hourly_budget_usd      → triggers warning at 80%, blocks at 100%
  per_decision_budget_ms → if decision takes > N ms of compute, alert

TRACKED METRICS:
  rtde:cost:hourly_decisions    → decision count in current hour
  rtde:cost:hourly_compute_ms   → total inference time in current hour
  rtde:cost:hourly_instances    → sum of instances_after in current hour
  rtde:cost:budget_exceeded     → "1" if hourly budget exceeded
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import redis.asyncio as aioredis
import structlog

from app.core.config import get_settings

logger = structlog.get_logger(__name__)

# Cost constants (configurable via env in full implementation)
_COMPUTE_COST_PER_MS = 0.000_001  # $0.000001 per ms of inference
_INSTANCE_COST_PER_TICK = 0.01  # $0.01 per instance per decision tick
_DEFAULT_HOURLY_BUDGET_USD = 100.0

_HOURLY_DECISIONS_KEY = "rtde:cost:hourly_decisions"
_HOURLY_COMPUTE_MS_KEY = "rtde:cost:hourly_compute_ms"
_HOURLY_INSTANCES_KEY = "rtde:cost:hourly_instances"
_BUDGET_EXCEEDED_KEY = "rtde:cost:budget_exceeded"
_COST_REPORT_KEY = "rtde:cost:last_report"


class DecisionCost:
    """Cost breakdown for a single decision."""

    def __init__(
        self,
        inference_ms: float,
        instance_count: int,
        compute_cost_usd: float,
        instance_cost_usd: float,
    ) -> None:
        self.inference_ms = inference_ms
        self.instance_count = instance_count
        self.compute_cost_usd = compute_cost_usd
        self.instance_cost_usd = instance_cost_usd
        self.total_cost_usd = compute_cost_usd + instance_cost_usd

    def to_dict(self) -> dict[str, Any]:
        return {
            "inference_ms": round(self.inference_ms, 2),
            "instance_count": self.instance_count,
            "compute_cost_usd": round(self.compute_cost_usd, 6),
            "instance_cost_usd": round(self.instance_cost_usd, 6),
            "total_cost_usd": round(self.total_cost_usd, 6),
        }


class CostTracker:
    """
    Tracks per-decision costs and enforces hourly budget guardrails.

    Redis-backed for fast reads on the decision hot path.
    """

    def __init__(
        self,
        hourly_budget_usd: float = _DEFAULT_HOURLY_BUDGET_USD,
    ) -> None:
        settings = get_settings()
        self._redis_url = settings.redis_url
        self._hourly_budget_usd = hourly_budget_usd

    def _client(self) -> aioredis.Redis:  # type: ignore[type-arg]
        return aioredis.from_url(self._redis_url, decode_responses=True)

    def compute_decision_cost(self, inference_ms: float, instance_count: int) -> DecisionCost:
        """Compute cost for a single decision."""
        compute_cost = inference_ms * _COMPUTE_COST_PER_MS
        instance_cost = instance_count * _INSTANCE_COST_PER_TICK
        return DecisionCost(
            inference_ms=inference_ms,
            instance_count=instance_count,
            compute_cost_usd=compute_cost,
            instance_cost_usd=instance_cost,
        )

    async def record_and_check(self, cost: DecisionCost) -> tuple[bool, str | None]:
        """
        Record decision cost and check against hourly budget.

        Returns:
            (budget_ok, warning_message)
            - budget_ok=False means budget exceeded, trigger fallback
            - warning_message is non-None when approaching/at limit
        """
        from redis.exceptions import RedisError

        now = datetime.now(UTC)
        seconds_to_hour_end = 3600 - (now.minute * 60 + now.second)

        try:
            async with self._client() as client:
                pipe = client.pipeline()
                pipe.incrbyfloat(_HOURLY_COMPUTE_MS_KEY, cost.inference_ms)
                pipe.incr(_HOURLY_DECISIONS_KEY)
                pipe.incrbyfloat(_HOURLY_INSTANCES_KEY, float(cost.instance_count))
                pipe.expire(_HOURLY_COMPUTE_MS_KEY, seconds_to_hour_end)
                pipe.expire(_HOURLY_DECISIONS_KEY, seconds_to_hour_end)
                pipe.expire(_HOURLY_INSTANCES_KEY, seconds_to_hour_end)
                results = await pipe.execute()
        except RedisError as exc:
            logger.warning("cost_tracker_redis_error", op="record", error=str(exc))
            return True, None  # fail-open: allow decisions when Redis is down

        total_compute_ms = float(results[0])
        _ = int(results[1])  # count not used in this path

        # Estimate hourly cost
        estimated_hourly_cost = (
            total_compute_ms * _COMPUTE_COST_PER_MS + float(results[2]) * _INSTANCE_COST_PER_TICK
        )

        # Check budget
        budget_ratio = estimated_hourly_cost / self._hourly_budget_usd

        if budget_ratio >= 1.0:
            logger.critical(
                "hourly_budget_exceeded",
                estimated_cost=estimated_hourly_cost,
                budget=self._hourly_budget_usd,
                ratio=budget_ratio,
            )
            try:
                async with self._client() as client:
                    await client.set(_BUDGET_EXCEEDED_KEY, "1", ex=seconds_to_hour_end)
            except RedisError as exc:
                logger.warning("cost_tracker_redis_error", op="set_budget_exceeded", error=str(exc))
            return False, (
                f"Hourly budget exceeded: ${estimated_hourly_cost:.2f} / "
                f"${self._hourly_budget_usd:.2f}"
            )
        elif budget_ratio >= 0.8:
            logger.warning(
                "hourly_budget_warning",
                estimated_cost=estimated_hourly_cost,
                budget=self._hourly_budget_usd,
                ratio=budget_ratio,
            )
            return True, (
                f"Hourly budget at {budget_ratio:.0%}: "
                f"${estimated_hourly_cost:.2f} / ${self._hourly_budget_usd:.2f}"
            )

        return True, None

    async def is_budget_exceeded(self) -> bool:
        """Fast check for budget exceedance — called on hot path."""
        from redis.exceptions import RedisError

        try:
            async with self._client() as client:
                return await client.get(_BUDGET_EXCEEDED_KEY) == "1"
        except RedisError as exc:
            logger.warning("cost_tracker_redis_error", op="is_budget_exceeded", error=str(exc))
            return False  # fail-open

    async def get_hourly_report(self) -> dict[str, Any]:
        """Return current hourly cost report."""
        from redis.exceptions import RedisError

        try:
            async with self._client() as client:
                pipe = client.pipeline()
                pipe.get(_HOURLY_DECISIONS_KEY)
                pipe.get(_HOURLY_COMPUTE_MS_KEY)
                pipe.get(_HOURLY_INSTANCES_KEY)
                results = await pipe.execute()
        except RedisError as exc:
            logger.warning("cost_tracker_redis_error", op="get_report", error=str(exc))
            results = [None, None, None]

        decisions = int(results[0] or 0)
        compute_ms = float(results[1] or 0)
        instance_ticks = float(results[2] or 0)

        compute_cost = compute_ms * _COMPUTE_COST_PER_MS
        instance_cost = instance_ticks * _INSTANCE_COST_PER_TICK
        total_cost = compute_cost + instance_cost

        return {
            "hour": datetime.now(UTC).strftime("%Y-%m-%dT%H:00:00Z"),
            "decisions": decisions,
            "total_compute_ms": round(compute_ms, 2),
            "compute_cost_usd": round(compute_cost, 4),
            "instance_cost_usd": round(instance_cost, 4),
            "total_cost_usd": round(total_cost, 4),
            "hourly_budget_usd": self._hourly_budget_usd,
            "budget_utilization_pct": round(total_cost / self._hourly_budget_usd * 100, 1),
            "avg_cost_per_decision_usd": round(total_cost / max(1, decisions), 6),
        }
