"""
Cost + Circuit Breaker API — operational infrastructure status.

ENDPOINTS:
  GET /api/v1/cost/report         → hourly cost report
  GET /api/v1/cost/budget         → budget utilization status
  GET /api/v1/infra/circuit-breakers → circuit breaker states
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.circuit_breaker.db_breaker import celery_breaker, db_breaker, redis_breaker
from app.core.auth import CurrentUser, Role
from app.cost.cost_tracker import CostTracker
from app.dependencies.auth import require_role

cost_router = APIRouter(prefix="/api/v1/cost", tags=["cost"])
infra_router = APIRouter(prefix="/api/v1/infra", tags=["infrastructure"])

_cost_tracker = CostTracker()


@cost_router.get("/report")
async def get_cost_report(
    user: CurrentUser = Depends(require_role(Role.VIEWER)),
) -> dict[str, Any]:
    """Return current hourly cost breakdown."""
    return await _cost_tracker.get_hourly_report()


@cost_router.get("/budget")
async def get_budget_status(
    user: CurrentUser = Depends(require_role(Role.VIEWER)),
) -> dict[str, Any]:
    """Return budget utilization and whether budget is exceeded."""
    report = await _cost_tracker.get_hourly_report()
    exceeded = await _cost_tracker.is_budget_exceeded()
    return {
        "budget_exceeded": exceeded,
        "budget_utilization_pct": report["budget_utilization_pct"],
        "total_cost_usd": report["total_cost_usd"],
        "hourly_budget_usd": report["hourly_budget_usd"],
    }


@infra_router.get("/circuit-breakers")
async def get_circuit_breaker_status(
    user: CurrentUser = Depends(require_role(Role.VIEWER)),
) -> dict[str, Any]:
    """Return current state of all circuit breakers."""
    db_status = await db_breaker.get_status()
    redis_status = await redis_breaker.get_status()
    celery_status = await celery_breaker.get_status()

    all_closed = all(s["state"] == "CLOSED" for s in [db_status, redis_status, celery_status])

    return {
        "overall_healthy": all_closed,
        "circuit_breakers": {
            "postgres": db_status,
            "redis": redis_status,
            "celery": celery_status,
        },
    }
