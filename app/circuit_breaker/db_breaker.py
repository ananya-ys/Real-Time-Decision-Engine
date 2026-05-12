"""
CircuitBreaker — prevents cascading failures when dependencies are down.

WHY THIS EXISTS:
The review: "circuit breakers for Redis, DB, and worker calls."
Without circuit breakers:
  DB goes down → every request waits 30s for timeout → queue builds up → OOM

With circuit breakers:
  DB goes down → circuit opens after N failures → baseline serves immediately
  DB recovers → circuit half-opens → one probe request → success → circuit closes

STATES:
  CLOSED   → normal operation (calls pass through)
  OPEN     → dependency failed (calls return immediately with error)
  HALF_OPEN → testing recovery (one probe request allowed through)

THRESHOLDS:
  failure_threshold: N consecutive failures to open circuit
  recovery_timeout:  seconds before half-open probe
  success_threshold: N consecutive successes to close from half-open

REDIS-BACKED:
- State stored in Redis so all workers share circuit state
- One worker detects DB failure → all workers get circuit-open
- DB recovers → one worker probes → all workers get circuit-close
"""

from __future__ import annotations

import json
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import Any, TypeVar

import redis.asyncio as aioredis
import structlog
from redis.exceptions import RedisError

from app.core.config import get_settings

logger = structlog.get_logger(__name__)

T = TypeVar("T")

_BREAKER_KEY_PREFIX = "rtde:circuit_breaker:"


class CircuitState:
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open — dependency unavailable."""

    def __init__(self, name: str, state: str) -> None:
        self.name = name
        self.state = state
        super().__init__(f"Circuit breaker '{name}' is {state}")


class RedisCircuitBreaker:
    """
    Redis-backed circuit breaker for shared state across workers.

    Usage:
        breaker = RedisCircuitBreaker("postgres", failure_threshold=5)
        try:
            result = await breaker.call(some_db_query())
        except CircuitBreakerOpenError:
            # Use fallback (baseline policy, cached response, etc.)
            result = fallback_value
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout_seconds: int = 30,
        success_threshold: int = 2,
    ) -> None:
        self._name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_seconds
        self._success_threshold = success_threshold
        settings = get_settings()
        self._redis_url = settings.redis_url

    def _client(self) -> aioredis.Redis:  # type: ignore[type-arg]
        return aioredis.from_url(self._redis_url, decode_responses=True)

    def _key(self) -> str:
        return f"{_BREAKER_KEY_PREFIX}{self._name}"

    async def _get_state(self) -> dict[str, Any]:
        """Read current circuit state from Redis."""
        try:
            async with self._client() as client:
                raw = await client.get(self._key())
        except RedisError as exc:
            logger.warning("redis_error", path="db_breaker.py", error=str(exc))
        if raw is None:
            return {
                "state": CircuitState.CLOSED,
                "failure_count": 0,
                "success_count": 0,
                "last_failure_at": None,
                "opened_at": None,
            }
        return json.loads(raw)

    async def _save_state(self, state: dict[str, Any]) -> None:
        try:
            async with self._client() as client:
                # TTL: circuit state expires after 1 hour of no activity
                await client.set(self._key(), json.dumps(state), ex=3600)
        except RedisError as exc:
            logger.warning("redis_error", path="db_breaker.py", error=str(exc))

    async def is_open(self) -> bool:
        """Check if circuit is open — fast path for non-call checks."""
        state_data = await self._get_state()
        return state_data["state"] != CircuitState.CLOSED

    async def call(self, coro: Awaitable[T]) -> T:
        """
        Execute a coroutine through the circuit breaker.

        Raises CircuitBreakerOpenError if circuit is OPEN.
        Records success/failure to update circuit state.
        """
        state_data = await self._get_state()
        current_state = state_data["state"]

        if current_state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed → try half-open
            opened_at_str = state_data.get("opened_at")
            if opened_at_str:
                opened_at = datetime.fromisoformat(opened_at_str)
                now = datetime.now(UTC)
                if (now - opened_at).total_seconds() >= self._recovery_timeout:
                    # Transition to half-open to probe recovery
                    state_data["state"] = CircuitState.HALF_OPEN
                    state_data["success_count"] = 0
                    await self._save_state(state_data)
                    current_state = CircuitState.HALF_OPEN
                    logger.info("circuit_breaker_half_open", name=self._name)
                else:
                    raise CircuitBreakerOpenError(self._name, CircuitState.OPEN)
            else:
                raise CircuitBreakerOpenError(self._name, CircuitState.OPEN)

        # Execute the coroutine
        try:
            result = await coro
            await self._record_success(state_data)
            return result
        except Exception as exc:
            await self._record_failure(state_data, exc)
            raise

    async def _record_success(self, state_data: dict[str, Any]) -> None:
        state_data["failure_count"] = 0
        state_data["success_count"] = state_data.get("success_count", 0) + 1

        if state_data["state"] == CircuitState.HALF_OPEN:
            if state_data["success_count"] >= self._success_threshold:
                state_data["state"] = CircuitState.CLOSED
                state_data["opened_at"] = None
                logger.info("circuit_breaker_closed", name=self._name)

        await self._save_state(state_data)

    async def _record_failure(self, state_data: dict[str, Any], exc: Exception) -> None:
        state_data["failure_count"] = state_data.get("failure_count", 0) + 1
        state_data["success_count"] = 0
        state_data["last_failure_at"] = datetime.now(UTC).isoformat()

        if state_data["failure_count"] >= self._failure_threshold:
            if state_data["state"] != CircuitState.OPEN:
                state_data["state"] = CircuitState.OPEN
                state_data["opened_at"] = datetime.now(UTC).isoformat()
                logger.critical(
                    "circuit_breaker_opened",
                    name=self._name,
                    failure_count=state_data["failure_count"],
                    error=str(exc),
                )

        await self._save_state(state_data)

    async def get_status(self) -> dict[str, Any]:
        state_data = await self._get_state()
        return {
            "name": self._name,
            "state": state_data["state"],
            "failure_count": state_data.get("failure_count", 0),
            "success_count": state_data.get("success_count", 0),
            "last_failure_at": state_data.get("last_failure_at"),
            "opened_at": state_data.get("opened_at"),
            "failure_threshold": self._failure_threshold,
            "recovery_timeout_seconds": self._recovery_timeout,
        }


# Singleton circuit breakers — shared across all service instances
db_breaker = RedisCircuitBreaker("postgres", failure_threshold=5, recovery_timeout_seconds=30)
redis_breaker = RedisCircuitBreaker("redis", failure_threshold=3, recovery_timeout_seconds=15)
celery_breaker = RedisCircuitBreaker("celery", failure_threshold=10, recovery_timeout_seconds=60)
