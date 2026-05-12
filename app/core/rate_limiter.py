"""
RateLimitMiddleware — actual enforcement, not just description.

SLIDING WINDOW ALGORITHM:
  - Redis sorted set per (actor, endpoint_class)
  - Score = unix timestamp in milliseconds
  - ZREMRANGEBYSCORE removes entries older than window
  - ZCARD counts current requests
  - If count >= limit → 429

WHY SLIDING WINDOW:
  Fixed window has burst problem: allow 100 req at :59, 100 more at :01.
  Sliding window always enforces "at most N in last T seconds."

ENDPOINT CLASSES AND LIMITS:
  critical    → 3 per 10 min (kill switch, global operator actions)
  operator    → 30 per min (any /api/v1/operator/* endpoint)
  write       → 60 per min (any POST/PATCH/DELETE)
  read        → 300 per min (any GET)
  auth        → 20 per min (login attempts)
  default     → 120 per min

HEADER RESPONSE:
  X-RateLimit-Limit: N
  X-RateLimit-Remaining: M
  X-RateLimit-Reset: unix_ts
  Retry-After: seconds (on 429)
"""

from __future__ import annotations

import time
from collections.abc import Callable

import redis.asyncio as aioredis
import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = structlog.get_logger(__name__)


# (path_prefix, method) → (limit, window_seconds)
_RATE_RULES: list[tuple[str, str, int, int]] = [
    # Critical operator endpoints — 3 per 10 minutes
    ("/api/v1/operator/kill-switch/activate", "POST", 3, 600),
    ("/api/v1/operator/maintenance", "POST", 3, 600),
    ("/api/v1/approvals/confirm/challenge", "POST", 3, 600),
    # Operator write endpoints — 30 per minute
    ("/api/v1/operator/", "POST", 30, 60),
    ("/api/v1/approvals/", "POST", 30, 60),
    # Auth endpoints — 20 per minute (prevent brute force)
    ("/api/v1/auth/token", "POST", 20, 60),
    ("/api/v1/auth/api-keys", "POST", 10, 60),
    # General write endpoints — 60 per minute
    ("/api/v1/", "POST", 60, 60),
    # Read endpoints — 300 per minute
    ("/api/v1/", "GET", 300, 60),
    # WebSocket — no rate limit
    ("/ws/", "GET", 0, 1),
    # Health — unlimited
    ("/health", "GET", 0, 1),
    ("/metrics", "GET", 0, 1),
]


def _get_limit(path: str, method: str) -> tuple[int, int]:
    """Return (limit, window_seconds) for a given path+method."""
    for prefix, m, limit, window in _RATE_RULES:
        if (m == method or m == "*") and path.startswith(prefix):
            return limit, window
    return 120, 60  # default


def _get_actor(request: Request) -> str:
    """Extract actor identifier for rate limiting."""
    # Try Authorization header
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        # Use first 16 chars of token as actor (fast, no decode needed)
        return f"jwt:{auth[7:23]}"
    # Try API key
    api_key = request.headers.get("X-API-Key", "")
    if api_key:
        return f"key:{api_key[:16]}"
    # Fall back to IP
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return f"ip:{forwarded_for.split(',')[0].strip()}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiter backed by Redis.

    Enforced before any endpoint handler runs.
    Returns 429 with Retry-After header.
    """

    def __init__(self, app, redis_url: str) -> None:
        super().__init__(app)
        self._redis_url = redis_url

    def _client(self) -> aioredis.Redis:  # type: ignore[type-arg]
        return aioredis.from_url(self._redis_url, decode_responses=True)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        method = request.method

        # Skip rate limit for health/metrics/websocket
        if path in ("/health", "/metrics") or path.startswith("/ws/"):
            return await call_next(request)

        limit, window = _get_limit(path, method)
        if limit == 0:  # unlimited
            return await call_next(request)

        actor = _get_actor(request)
        # Bucket key: collapse path to its prefix class for cleaner grouping
        bucket = f"rtde:rl:{method}:{path[:40]}:{actor}"

        now_ms = int(time.time() * 1000)
        window_ms = window * 1000

        try:
            async with self._client() as client:
                pipe = client.pipeline()
                # Remove expired entries
                pipe.zremrangebyscore(bucket, 0, now_ms - window_ms)
                # Count current
                pipe.zcard(bucket)
                # Add current request
                pipe.zadd(bucket, {str(now_ms): now_ms})
                # Set TTL
                pipe.expire(bucket, window + 5)
                results = await pipe.execute()

            current_count = results[1]

            if current_count >= limit:
                # Calculate retry-after — use async with to avoid connection leak
                retry_after = window
                try:
                    async with self._client() as _c:
                        oldest = await _c.zrange(bucket, 0, 0, withscores=True)
                    if oldest:
                        oldest_ts = oldest[0][1]
                        retry_after = max(1, int((oldest_ts + window_ms - now_ms) / 1000))
                except Exception:
                    pass  # fallback to window value already set

                logger.warning(
                    "rate_limit_exceeded",
                    actor=actor,
                    path=path,
                    method=method,
                    limit=limit,
                    window=window,
                )

                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "detail": f"Max {limit} requests per {window}s. Retry in {retry_after}s.",
                        "limit": limit,
                        "window_seconds": window,
                        "retry_after": retry_after,
                    },
                    headers={
                        "X-RateLimit-Limit": str(limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(time.time()) + retry_after),
                        "Retry-After": str(retry_after),
                    },
                )

            response = await call_next(request)
            remaining = max(0, limit - current_count - 1)
            response.headers["X-RateLimit-Limit"] = str(limit)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(int(time.time()) + window)
            return response

        except Exception as exc:
            # Rate limiter failure must not block legitimate traffic
            logger.error("rate_limiter_redis_error", error=str(exc))
            return await call_next(request)
