"""
RTDE Application Entry Point.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.core.config import get_settings
from app.core.database import engine
from app.core.error_handlers import register_exception_handlers
from app.core.idempotency import IdempotencyMiddleware
from app.core.logging import setup_logging
from app.core.middleware import RequestLoggingMiddleware
from app.core.rate_limiter import RateLimitMiddleware

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    settings = get_settings()
    logger.info("starting_rtde", app_name=settings.app_name, env=settings.app_env)

    # Verify database — fatal if down
    async with engine.begin() as conn:
        await conn.execute(text("SELECT 1"))
    logger.info("database_connected")

    # Verify Redis — non-fatal for demo resilience
    try:
        redis_client = aioredis.from_url(settings.redis_url)
        await redis_client.ping()
        await redis_client.aclose()
        logger.info("redis_connected")
    except Exception as exc:
        logger.warning("redis_unavailable_continuing", error=str(exc))

    logger.info("startup_complete", env=settings.app_env)

    yield

    logger.info("shutting_down")
    await engine.dispose()
    logger.info("shutdown_complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RTDE — Real-Time Decision Engine",
        description="Production-grade ML decision system for dynamic resource allocation",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — open for demo (frontend on Vercel)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting (gracefully skips if Redis unavailable)
    settings_obj = get_settings()
    app.add_middleware(RateLimitMiddleware, redis_url=settings_obj.redis_url)
    app.add_middleware(IdempotencyMiddleware, redis_url=settings_obj.redis_url)
    app.add_middleware(RequestLoggingMiddleware)

    register_exception_handlers(app)

    # ── Routers ─────────────────────────────────────────────────
    from app.api.v1.health import router as health_router

    app.include_router(health_router)

    from app.api.v1.decision import router as decision_router
    from app.api.v1.monitoring import router as monitoring_router
    from app.api.v1.operator import router as operator_router
    from app.api.v1.policies import router as policies_router
    from app.api.v1.reward import router as reward_router

    app.include_router(decision_router)
    app.include_router(reward_router)
    app.include_router(policies_router)
    app.include_router(monitoring_router)
    app.include_router(operator_router)

    from app.api.v1.audit import router as audit_router
    from app.api.v1.auth import router as auth_router
    from app.api.v1.backtesting import router as backtesting_router
    from app.api.v1.canary import router as canary_router
    from app.api.v1.cost_and_infra import cost_router, infra_router
    from app.api.v1.trust import router as trust_router
    from app.api.v1.websocket import router as ws_router

    app.include_router(auth_router)
    app.include_router(audit_router)
    app.include_router(canary_router)
    app.include_router(backtesting_router)
    app.include_router(trust_router)
    app.include_router(cost_router)
    app.include_router(infra_router)
    app.include_router(ws_router)

    from app.api.v1.approvals import router as approvals_router
    from app.api.v1.explainability import router as explain_router
    from app.api.v1.incidents import router as incidents_router

    app.include_router(approvals_router)
    app.include_router(explain_router)
    app.include_router(incidents_router)

    return app


app = create_app()
