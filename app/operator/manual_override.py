"""
ManualOverride — operator control surfaces beyond kill switch.

All Redis calls are fail-open: if Redis is unavailable, operations return
safe defaults (False for checks, success dicts for writes with a warning).
"""

from __future__ import annotations

from datetime import UTC, datetime

import redis.asyncio as aioredis
import structlog
from redis.exceptions import RedisError

from app.core.config import get_settings
from app.operator.kill_switch import KillSwitch

logger = structlog.get_logger(__name__)

_BASELINE_OVERRIDE_KEY = "rtde:override:force_baseline"
_MAINTENANCE_KEY = "rtde:override:maintenance_mode"


class ManualOverride:
    def __init__(self) -> None:
        settings = get_settings()
        self._redis_url = settings.redis_url
        self._kill_switch = KillSwitch()

    def _client(self) -> aioredis.Redis:  # type: ignore[type-arg]
        return aioredis.from_url(self._redis_url, decode_responses=True)

    async def force_baseline(self, actor: str, reason: str) -> dict:
        try:
            async with self._client() as client:
                await client.set(_BASELINE_OVERRIDE_KEY, "1")
        except RedisError as exc:
            logger.warning("manual_override_redis_error", op="force_baseline", error=str(exc))

        logger.critical("baseline_override_activated", actor=actor, reason=reason)
        return {
            "status": "baseline_override_active",
            "actor": actor,
            "reason": reason,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def release_baseline(self, actor: str, reason: str) -> dict:
        try:
            async with self._client() as client:
                await client.set(_BASELINE_OVERRIDE_KEY, "0")
        except RedisError as exc:
            logger.warning("manual_override_redis_error", op="release_baseline", error=str(exc))

        logger.warning("baseline_override_released", actor=actor, reason=reason)
        return {
            "status": "baseline_override_inactive",
            "actor": actor,
            "reason": reason,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def is_baseline_forced(self) -> bool:
        """Check if baseline override is active — called on every decision path."""
        try:
            async with self._client() as client:
                return await client.get(_BASELINE_OVERRIDE_KEY) == "1"
        except RedisError as exc:
            logger.warning("kill_switch_redis_unavailable_using_safe_defaults", error=str(exc))
            return False  # fail-open: allow ML decisions when Redis is down

    async def enter_maintenance_mode(self, actor: str, reason: str) -> dict:
        try:
            async with self._client() as client:
                await client.set(_MAINTENANCE_KEY, "1")
                await client.set(_BASELINE_OVERRIDE_KEY, "1")
        except RedisError as exc:
            logger.warning("manual_override_redis_error", op="enter_maintenance", error=str(exc))

        try:
            await self._kill_switch.freeze_exploration(actor, f"maintenance_mode: {reason}")
            await self._kill_switch.freeze_promotion(actor, f"maintenance_mode: {reason}")
        except RedisError as exc:
            logger.warning("manual_override_redis_error", op="freeze_switches", error=str(exc))

        logger.critical("maintenance_mode_entered", actor=actor, reason=reason)
        return {
            "status": "maintenance_mode_active",
            "effects": ["baseline_override: active", "exploration: frozen", "promotion: frozen"],
            "actor": actor,
            "reason": reason,
        }

    async def exit_maintenance_mode(self, actor: str, reason: str) -> dict:
        try:
            async with self._client() as client:
                await client.set(_MAINTENANCE_KEY, "0")
                await client.set(_BASELINE_OVERRIDE_KEY, "0")
        except RedisError as exc:
            logger.warning("manual_override_redis_error", op="exit_maintenance", error=str(exc))

        try:
            await self._kill_switch.unfreeze_exploration(actor, f"maintenance_ended: {reason}")
            await self._kill_switch.unfreeze_promotion(actor, f"maintenance_ended: {reason}")
        except RedisError as exc:
            logger.warning("manual_override_redis_error", op="unfreeze_switches", error=str(exc))

        logger.warning("maintenance_mode_exited", actor=actor, reason=reason)
        return {"status": "maintenance_mode_inactive", "actor": actor, "reason": reason}

    async def is_maintenance_mode(self) -> bool:
        try:
            async with self._client() as client:
                return await client.get(_MAINTENANCE_KEY) == "1"
        except RedisError as exc:
            logger.warning("manual_override_redis_error", op="is_maintenance", error=str(exc))
            return False

    async def get_override_status(self) -> dict:
        try:
            async with self._client() as client:
                pipe = client.pipeline()
                pipe.get(_BASELINE_OVERRIDE_KEY)
                pipe.get(_MAINTENANCE_KEY)
                results = await pipe.execute()
            baseline = results[0] == "1"
            maintenance = results[1] == "1"
        except RedisError as exc:
            logger.warning("manual_override_redis_error", op="get_status", error=str(exc))
            baseline = False
            maintenance = False

        try:
            kill_status = await self._kill_switch.full_status()
        except RedisError as exc:
            logger.warning("manual_override_redis_error", op="kill_status", error=str(exc))
            kill_status = {}

        return {
            "baseline_override": baseline,
            "maintenance_mode": maintenance,
            "kill_switch": kill_status,
        }
