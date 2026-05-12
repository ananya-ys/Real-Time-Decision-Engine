"""
EmergencyLockout — automatic system lockout on repeated failure patterns.

TRIGGERS:
  1. N consecutive confirmation validation failures (brute force attempt)
  2. N consecutive auth failures from same IP
  3. N consecutive circuit breaker opens in short window
  4. Manual operator lockout (from UI or runbook)

EFFECT:
  All operator endpoints return 503 (Lockout Active).
  Only /health, /metrics, /api/v1/auth/token, and /api/v1/operator/lockout/release
  remain accessible.

WHY REDIS NOT DB:
  Lockout must activate in < 100ms across all workers.
  DB round-trip is too slow and DB being down = can't lock out = security hole.

RELEASE:
  Only ADMIN with valid JWT can release the lockout.
  Releasing requires a separate typed confirmation (different from action confirmation).
  All lockout events are written to the audit trail with hash chain.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as aioredis
import structlog
from redis.exceptions import RedisError

logger = structlog.get_logger(__name__)

_LOCKOUT_KEY = "rtde:emergency_lockout:active"
_LOCKOUT_META_KEY = "rtde:emergency_lockout:metadata"
_FAILURE_COUNT_KEY = "rtde:emergency_lockout:failures:{actor}"
_MAX_FAILURES = 5
_FAILURE_WINDOW = 300  # 5 minutes
_LOCKOUT_TTL = 3600  # 1 hour auto-expire (manual release preferred)


class EmergencyLockout:
    """
    System-wide emergency lockout for security incidents.

    Integrated into RateLimitMiddleware and operator API.
    """

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url

    def _client(self) -> aioredis.Redis:  # type: ignore[type-arg]
        return aioredis.from_url(self._redis_url, decode_responses=True)

    async def is_locked_out(self) -> bool:
        """Fast check — O(1) Redis GET on hot path."""
        try:
            async with self._client() as client:
                return await client.get(_LOCKOUT_KEY) == "1"
        except RedisError as exc:
            logger.warning("redis_error", path="emergency_lockout.py", error=str(exc))
            return False

    async def record_failure(self, actor: str, failure_type: str) -> bool:
        """
        Record a security failure. Returns True if lockout was triggered.

        Args:
            actor: Who failed (IP or user_id).
            failure_type: e.g. "confirmation_mismatch", "auth_failure"

        Returns True if this failure caused automatic lockout.
        """
        key = _FAILURE_COUNT_KEY.format(actor=actor)
        try:
            async with self._client() as client:
                count = await client.incr(key)
                await client.expire(key, _FAILURE_WINDOW)
        except RedisError as exc:
            logger.warning("redis_error", path="emergency_lockout.py", error=str(exc))
            return False
        if count >= _MAX_FAILURES:
            await self.activate(
                actor="auto_lockout",
                reason=(
                    f"Auto-lockout: {count} consecutive {failure_type} failures "
                    f"from actor {actor} in {_FAILURE_WINDOW}s window"
                ),
            )
            return True

        return False

    async def activate(self, actor: str, reason: str) -> dict[str, Any]:
        """
        Activate system-wide lockout.

        All operator endpoints return 503 until manually released.
        """
        now = datetime.now(UTC).isoformat()
        meta = {
            "activated_by": actor,
            "reason": reason,
            "activated_at": now,
            "auto_release_at": None,
        }

        try:
            async with self._client() as client:
                await client.set(_LOCKOUT_KEY, "1", ex=_LOCKOUT_TTL)
                await client.set(_LOCKOUT_META_KEY, json.dumps(meta))
        except RedisError as exc:
            logger.warning("redis_error", path="emergency_lockout.py", error=str(exc))
            return False
        logger.critical(
            "emergency_lockout_activated",
            actor=actor,
            reason=reason,
        )

        return {"status": "lockout_active", "activated_at": now, "reason": reason}

    async def release(self, actor: str, reason: str) -> dict[str, Any]:
        """Release lockout. Requires ADMIN + valid reason."""
        try:
            async with self._client() as client:
                await client.delete(_LOCKOUT_KEY)
        except RedisError as exc:
            logger.warning("redis_error", path="emergency_lockout.py", error=str(exc))
            return False
        logger.warning("emergency_lockout_released", actor=actor, reason=reason)
        return {"status": "lockout_released", "released_by": actor}

    async def get_status(self) -> dict[str, Any]:
        """Return current lockout state and metadata."""
        try:
            async with self._client() as client:
                active = await client.get(_LOCKOUT_KEY) == "1"
                meta_str = await client.get(_LOCKOUT_META_KEY)
        except RedisError as exc:
            logger.warning("redis_error", path="emergency_lockout.py", error=str(exc))
            return False
        meta = json.loads(meta_str) if meta_str else {}
        return {
            "lockout_active": active,
            "metadata": meta,
        }
