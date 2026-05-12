"""
KillSwitch — global and per-policy emergency stop.

WHY THIS EXISTS:
The review's #1 criticism: "the system is still too autonomous for production."
Without a kill switch, an operator watching a runaway policy causing SLA breaches
has no recourse except manual DB edits or restarting pods.

THE PATTERN:
- Kill switch state stored in Redis (not DB) — instant propagation, no DB round-trip.
- Every DecisionService call checks kill switch FIRST, before policy.decide().
- < 100ms from operator click to all workers respecting the switch.
- Full audit log: who killed it, when, why.

STATES:
  GLOBAL_OFF   → all policies disabled, all decisions use baseline
  POLICY_OFF   → specific policy_type disabled
  EXPLORATION_FREEZE → all exploration suppressed (ExplorationGuard forced off)
  PROMOTION_FREEZE   → no policy promotions allowed

REDIS KEYS:
  rtde:killswitch:global          → "1" if global kill active
  rtde:killswitch:policy:{type}   → "1" if that policy_type is killed
  rtde:killswitch:exploration      → "1" if exploration frozen
  rtde:killswitch:promotion        → "1" if promotion frozen
  rtde:killswitch:metadata         → JSON with who/when/why for each key

WHAT BREAKS IF WRONG:
- Kill switch in DB: 30-50ms per check = P99 latency breach.
- Kill switch in process memory: doesn't propagate to other workers.
- No kill switch: operator is helpless during incident.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as aioredis
import structlog

from app.core.config import get_settings
from app.schemas.common import PolicyType

logger = structlog.get_logger(__name__)

_GLOBAL_KEY = "rtde:killswitch:global"
_EXPLORATION_KEY = "rtde:killswitch:exploration"
_PROMOTION_KEY = "rtde:killswitch:promotion"
_METADATA_KEY = "rtde:killswitch:metadata"
_POLICY_KEY_PREFIX = "rtde:killswitch:policy:"


def _policy_key(policy_type: PolicyType) -> str:
    return f"{_POLICY_KEY_PREFIX}{policy_type.value}"


class KillSwitchState:
    """Current state of all kill switches."""

    def __init__(
        self,
        global_killed: bool,
        exploration_frozen: bool,
        promotion_frozen: bool,
        killed_policies: set[PolicyType],
    ) -> None:
        self.global_killed = global_killed
        self.exploration_frozen = exploration_frozen
        self.promotion_frozen = promotion_frozen
        self.killed_policies = killed_policies

    def is_policy_active(self, policy_type: PolicyType) -> bool:
        """Return True if this policy can serve decisions."""
        if self.global_killed:
            return False
        return policy_type not in self.killed_policies

    def allow_exploration(self) -> bool:
        """Return True if exploration is permitted."""
        return not self.global_killed and not self.exploration_frozen

    def allow_promotion(self) -> bool:
        """Return True if policy promotion is permitted."""
        return not self.global_killed and not self.promotion_frozen


class KillSwitch:
    """
    Redis-backed kill switch for all RTDE control surfaces.

    Checks are O(1) Redis GET operations — fast enough for every request path.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._redis_url = settings.redis_url

    def _client(self) -> aioredis.Redis:  # type: ignore[type-arg]
        return aioredis.from_url(self._redis_url, decode_responses=True)

    async def get_state(self) -> KillSwitchState:
        """Read full kill switch state in one pipeline call.

        Returns safe defaults (all switches OFF = allow everything) when Redis
        is unavailable. This is the correct fail-open behaviour: we must not
        freeze the system just because the state store is temporarily down.
        """
        try:
            async with self._client() as client:
                pipe = client.pipeline()
                pipe.get(_GLOBAL_KEY)
                pipe.get(_EXPLORATION_KEY)
                pipe.get(_PROMOTION_KEY)
                for pt in PolicyType:
                    pipe.get(_policy_key(pt))
                results = await pipe.execute()

            global_killed = results[0] == "1"
            exploration_frozen = results[1] == "1"
            promotion_frozen = results[2] == "1"

            killed_policies: set[PolicyType] = set()
            for i, pt in enumerate(PolicyType):
                if results[3 + i] == "1":
                    killed_policies.add(pt)

            return KillSwitchState(
                global_killed=global_killed,
                exploration_frozen=exploration_frozen,
                promotion_frozen=promotion_frozen,
                killed_policies=killed_policies,
            )
        except Exception as exc:
            logger.warning(
                "kill_switch_redis_unavailable_using_safe_defaults",
                error=str(exc),
            )
            # Fail-open: Redis down → treat all switches as OFF so inference
            # continues with baseline policy. Do NOT block traffic.
            return KillSwitchState(
                global_killed=False,
                exploration_frozen=False,
                promotion_frozen=False,
                killed_policies=set(),
            )

    async def is_globally_killed(self) -> bool:
        """Fast single-key check for hot path. Fail-open on Redis error."""
        try:
            async with self._client() as client:
                return await client.get(_GLOBAL_KEY) == "1"
        except Exception as exc:
            logger.warning("kill_switch_redis_unavailable", error=str(exc))
            return False  # fail-open

    async def activate_global(self, actor: str, reason: str) -> None:
        """
        EMERGENCY: Disable all ML policies. Only baseline serves.

        Sets global kill in Redis — all workers see it on next request.
        """
        await self._set_switch(_GLOBAL_KEY, actor, reason, value="1")
        logger.critical(
            "kill_switch_global_activated",
            actor=actor,
            reason=reason,
        )

    async def deactivate_global(self, actor: str, reason: str) -> None:
        """Re-enable ML policies after incident is resolved."""
        await self._set_switch(_GLOBAL_KEY, actor, reason, value="0")
        logger.warning("kill_switch_global_deactivated", actor=actor, reason=reason)

    async def kill_policy(self, policy_type: PolicyType, actor: str, reason: str) -> None:
        """Disable a specific policy type."""
        await self._set_switch(_policy_key(policy_type), actor, reason, value="1")
        logger.warning(
            "kill_switch_policy_activated",
            policy_type=policy_type.value,
            actor=actor,
            reason=reason,
        )

    async def restore_policy(self, policy_type: PolicyType, actor: str, reason: str) -> None:
        """Re-enable a specific policy type."""
        await self._set_switch(_policy_key(policy_type), actor, reason, value="0")
        logger.info(
            "kill_switch_policy_restored",
            policy_type=policy_type.value,
            actor=actor,
            reason=reason,
        )

    async def freeze_exploration(self, actor: str, reason: str) -> None:
        """Suppress all exploration immediately. Policies still serve (exploit only)."""
        await self._set_switch(_EXPLORATION_KEY, actor, reason, value="1")
        logger.warning("exploration_frozen", actor=actor, reason=reason)

    async def unfreeze_exploration(self, actor: str, reason: str) -> None:
        """Re-enable exploration."""
        await self._set_switch(_EXPLORATION_KEY, actor, reason, value="0")
        logger.info("exploration_unfrozen", actor=actor, reason=reason)

    async def freeze_promotion(self, actor: str, reason: str) -> None:
        """Prevent any policy promotion during incident or maintenance."""
        await self._set_switch(_PROMOTION_KEY, actor, reason, value="1")
        logger.warning("promotion_frozen", actor=actor, reason=reason)

    async def unfreeze_promotion(self, actor: str, reason: str) -> None:
        """Re-enable policy promotion."""
        await self._set_switch(_PROMOTION_KEY, actor, reason, value="0")
        logger.info("promotion_unfrozen", actor=actor, reason=reason)

    async def _set_switch(self, key: str, actor: str, reason: str, value: str) -> None:
        """Write switch state + metadata atomically. Fail-open if Redis is down."""
        from redis.exceptions import RedisError

        now = datetime.now(UTC).isoformat()
        meta: dict[str, Any] = {"actor": actor, "reason": reason, "timestamp": now}
        try:
            async with self._client() as client:
                pipe = client.pipeline()
                pipe.set(key, value)
                pipe.hset(_METADATA_KEY, key, json.dumps(meta))
                await pipe.execute()
        except RedisError as exc:
            logger.warning("kill_switch_redis_write_error", key=key, error=str(exc))

    async def get_metadata(self) -> dict[str, Any]:
        """Return who activated each switch and when."""
        from redis.exceptions import RedisError

        try:
            async with self._client() as client:
                raw = await client.hgetall(_METADATA_KEY)
            return {k: json.loads(v) for k, v in raw.items()}
        except RedisError as exc:
            logger.warning("kill_switch_redis_metadata_error", error=str(exc))
            return {}

    async def full_status(self) -> dict[str, Any]:
        """Return complete kill switch status for operator dashboard."""
        state = await self.get_state()
        metadata = await self.get_metadata()

        return {
            "global_killed": state.global_killed,
            "exploration_frozen": state.exploration_frozen,
            "promotion_frozen": state.promotion_frozen,
            "killed_policies": [p.value for p in state.killed_policies],
            "metadata": metadata,
        }
