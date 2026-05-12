"""
IdempotencyMiddleware — prevent double-submission on write endpoints.

WHY THIS EXISTS:
Network retries cause duplicate operator actions. Without idempotency:
- Operator clicks "force baseline" → request times out → retries → kills twice

HOW IT WORKS:
  Client sends: Idempotency-Key: <uuid> header on any POST request.
  Server caches the response for 5 minutes in Redis.
  Duplicate request with same key → returns cached response immediately.
  No key → request processed normally (no idempotency check).

WHICH ENDPOINTS NEED IT:
  POST /api/v1/operator/*   → all operator controls
  POST /api/v1/approvals/*  → approval workflow
  POST /api/v1/canary/*     → canary controls
  POST /api/v1/decision     → NOT idempotent (each decision is distinct)
  POST /api/v1/rewards      → NOT idempotent (rewards are distinct events)

CLIENT USAGE:
  POST /api/v1/operator/kill-switch/activate
  Idempotency-Key: 550e8400-e29b-41d4-a716-446655440000

  On retry with same key: returns original response, no double-action.

CACHE TTL: 5 minutes (enough for network retries, short enough to clear naturally)
"""

from __future__ import annotations

import json
from collections.abc import Callable

import redis.asyncio as aioredis
import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = structlog.get_logger(__name__)

_IDEMPOTENT_PREFIXES = (
    "/api/v1/operator/",
    "/api/v1/approvals/",
    "/api/v1/canary/",
    "/api/v1/incidents/",
)
_TTL = 300  # 5 minutes


class IdempotencyMiddleware(BaseHTTPMiddleware):
    """Cache responses for idempotent write operations."""

    def __init__(self, app, redis_url: str) -> None:
        super().__init__(app)
        self._redis_url = redis_url

    def _client(self) -> aioredis.Redis:  # type: ignore[type-arg]
        return aioredis.from_url(self._redis_url, decode_responses=True)

    def _needs_idempotency(self, request: Request) -> bool:
        if request.method != "POST":
            return False
        return any(request.url.path.startswith(p) for p in _IDEMPOTENT_PREFIXES)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self._needs_idempotency(request):
            return await call_next(request)

        idempotency_key = request.headers.get("Idempotency-Key", "").strip()
        if not idempotency_key:
            return await call_next(request)

        # Validate key format (must be UUID-like or similar)
        if len(idempotency_key) < 8 or len(idempotency_key) > 128:
            return JSONResponse(
                status_code=400,
                content={"error": "Idempotency-Key must be 8-128 characters"},
            )

        cache_key = f"rtde:idem:{request.url.path}:{idempotency_key}"

        try:
            async with self._client() as client:
                cached = await client.get(cache_key)

            if cached:
                cached_data = json.loads(cached)
                logger.info(
                    "idempotency_cache_hit",
                    key=idempotency_key[:12],
                    path=request.url.path,
                )
                return JSONResponse(
                    status_code=cached_data["status_code"],
                    content=cached_data["body"],
                    headers={"X-Idempotency-Replayed": "true"},
                )
        except Exception as exc:
            logger.error("idempotency_redis_error", error=str(exc))
            return await call_next(request)

        response = await call_next(request)

        # Cache successful responses only
        if 200 <= response.status_code < 300:
            try:
                body = b""
                async for chunk in response.body_iterator:
                    body += chunk

                # Only cache valid JSON — skip binary/HTML gracefully
                try:
                    parsed_body = json.loads(body.decode())
                    async with self._client() as client:
                        await client.set(
                            cache_key,
                            json.dumps({"status_code": response.status_code, "body": parsed_body}),
                            ex=_TTL,
                        )
                except (json.JSONDecodeError, UnicodeDecodeError):
                    logger.debug("idempotency_skip_non_json", path=str(request.url.path))

                # MUST rebuild response — body_iterator already consumed
                return Response(
                    content=body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                )
            except Exception as exc:
                logger.error("idempotency_cache_store_error", error=str(exc))
                return JSONResponse(status_code=500, content={"error": "Idempotency buffer error"})

        return response
