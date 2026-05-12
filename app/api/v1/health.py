"""
Health and metrics endpoints.

WHY THIS EXISTS:
- Load balancer / Kubernetes needs health endpoint to detect failure.
- Separate DB and Redis checks — if either is down, the app is degraded.
- /metrics endpoint for Prometheus scraping.

WHAT BREAKS IF WRONG:
- No health check = load balancer sends traffic to dead instances.
- Combined health check = can't tell if DB or Redis failed.
"""

from __future__ import annotations

from typing import Any

import redis.asyncio as aioredis
import structlog
from fastapi import APIRouter, Depends
from prometheus_client import generate_latest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import Response

from app.core.config import Settings, get_settings
from app.dependencies.db import get_db

router = APIRouter(tags=["infrastructure"])
logger = structlog.get_logger(__name__)


@router.get("/health")
async def health_check(
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """
    Health check endpoint.

    Returns:
        JSON with status of each dependency: db, redis, worker.
        HTTP 200 if all healthy, 503 if any dependency is down.
    """
    health: dict[str, Any] = {"status": "ok", "app": settings.app_name}
    status_code = 200

    # ── Database Check ──────────────────────────────────────────
    try:
        await db.execute(text("SELECT 1"))
        health["db"] = "ok"
    except Exception as e:
        logger.error("health_check_db_failed", error=str(e))
        health["db"] = "error"
        health["status"] = "degraded"
        status_code = 503

    # ── Redis Check ─────────────────────────────────────────────
    try:
        redis_client = aioredis.from_url(settings.redis_url)
        await redis_client.ping()
        await redis_client.aclose()
        health["redis"] = "ok"
    except Exception as e:
        logger.error("health_check_redis_failed", error=str(e))
        health["redis"] = "error"
        health["status"] = "degraded"
        status_code = 503

    # ── Celery Worker Check (best-effort) ───────────────────────
    try:
        redis_client = aioredis.from_url(settings.celery_broker_url)
        await redis_client.ping()
        await redis_client.aclose()
        health["worker_broker"] = "ok"
    except Exception:
        health["worker_broker"] = "unknown"

    if status_code == 503:
        # Return 503 for orchestrator health probes
        from starlette.responses import JSONResponse

        return JSONResponse(content=health, status_code=503)

    return health


@router.get("/metrics")
async def metrics() -> Response:
    """Prometheus-compatible metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
