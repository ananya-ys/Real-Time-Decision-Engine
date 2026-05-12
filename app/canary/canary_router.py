"""
CanaryRouter — progressive traffic splitting for policy rollout.

WHY THIS EXISTS:
The review: "shadow mode is not enough."
Shadow mode: logs decisions but never commits. Zero real-world exposure.
Canary: commits to REAL traffic at a controlled percentage.

THE DIFFERENCE:
  Shadow:  policy A makes decision, shadow policy B logs but doesn't commit.
           Result: B never faces real traffic, real latency, real consequences.

  Canary:  10% of requests go to policy B (commits), 90% go to policy A.
           Result: B faces real traffic. If SLA breach → auto-abort.
           If stable → bump to 25% → 50% → 100% → promote.

TRAFFIC SPLITTING:
  Redis key: rtde:canary:policy_{type}:traffic_pct → "10"
  DecisionService reads this on each request.
  Using crypto-random (not sequential) for unbiased sampling.

AUTO-ABORT CONDITIONS:
  1. SLA breach rate during canary > 5% (vs < 1% baseline)
  2. Fallback rate during canary > 2%
  3. Manual abort by operator

STAGES:
  INACTIVE → 10% → 25% → 50% → 100% → PROMOTED
  Any stage → ABORTED on SLA breach
"""

from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as aioredis
import structlog
from redis.exceptions import RedisError

from app.core.config import get_settings
from app.schemas.common import PolicyType

logger = structlog.get_logger(__name__)

# Redis keys
_CANARY_KEY = "rtde:canary:{policy_type}"
_CANARY_METRICS_KEY = "rtde:canary:{policy_type}:metrics"

# Standard canary stages
CANARY_STAGES = [10, 25, 50, 100]


class CanaryConfig:
    """Current canary state for a policy type."""

    def __init__(
        self,
        policy_type: PolicyType,
        traffic_pct: int,  # 0 = inactive, 100 = full rollout
        stage: str,  # INACTIVE | CANARY_10 | CANARY_25 | CANARY_50 | ACTIVE | ABORTED
        started_at: str | None,
        aborted_at: str | None,
        abort_reason: str | None,
        version_id: str | None,
    ) -> None:
        self.policy_type = policy_type
        self.traffic_pct = traffic_pct
        self.stage = stage
        self.started_at = started_at
        self.aborted_at = aborted_at
        self.abort_reason = abort_reason
        self.version_id = version_id

    @property
    def is_active(self) -> bool:
        return self.traffic_pct > 0 and self.stage != "ABORTED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_type": self.policy_type.value,
            "traffic_pct": self.traffic_pct,
            "stage": self.stage,
            "started_at": self.started_at,
            "aborted_at": self.aborted_at,
            "abort_reason": self.abort_reason,
            "version_id": self.version_id,
        }


class CanaryRouter:
    """
    Routes a fraction of live traffic to the canary policy.

    Integrated into DecisionService: before deciding which policy to use,
    check if canary is active and route this request to canary based on
    traffic percentage.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._redis_url = settings.redis_url

    def _client(self) -> aioredis.Redis:  # type: ignore[type-arg]
        return aioredis.from_url(self._redis_url, decode_responses=True)

    def _key(self, policy_type: PolicyType) -> str:
        return _CANARY_KEY.format(policy_type=policy_type.value)

    def _metrics_key(self, policy_type: PolicyType) -> str:
        return _CANARY_METRICS_KEY.format(policy_type=policy_type.value)

    def should_use_canary(self, traffic_pct: int) -> bool:
        """
        Probabilistic routing decision.

        Uses cryptographically secure random for unbiased sampling.
        10% canary = 10% of requests route to canary policy.

        Called on HOT PATH — must be fast.
        """
        if traffic_pct <= 0:
            return False
        if traffic_pct >= 100:
            return True
        # Generate random int in [0, 99], compare to threshold
        return secrets.randbelow(100) < traffic_pct

    async def get_canary_config(self, policy_type: PolicyType) -> CanaryConfig:
        """Read canary state from Redis."""
        try:
            async with self._client() as client:
                raw = await client.get(self._key(policy_type))
        except RedisError as exc:
            logger.warning("canary_redis_error", op="get_config", error=str(exc))
            raw = None

        if raw is None:
            return CanaryConfig(
                policy_type=policy_type,
                traffic_pct=0,
                stage="INACTIVE",
                started_at=None,
                aborted_at=None,
                abort_reason=None,
                version_id=None,
            )

        data = json.loads(raw)
        return CanaryConfig(
            policy_type=policy_type,
            traffic_pct=data.get("traffic_pct", 0),
            stage=data.get("stage", "INACTIVE"),
            started_at=data.get("started_at"),
            aborted_at=data.get("aborted_at"),
            abort_reason=data.get("abort_reason"),
            version_id=data.get("version_id"),
        )

    async def start_canary(
        self,
        policy_type: PolicyType,
        version_id: str,
        initial_pct: int = 10,
        actor: str = "system",
    ) -> CanaryConfig:
        """
        Start a canary rollout for a policy at the given traffic percentage.

        Does NOT automatically progress stages — operator or auto-abort
        controls stage progression.
        """
        if initial_pct not in CANARY_STAGES and initial_pct != 0:
            raise ValueError(f"Invalid canary percentage: {initial_pct}. Use {CANARY_STAGES}")

        config = {
            "traffic_pct": initial_pct,
            "stage": f"CANARY_{initial_pct}",
            "started_at": datetime.now(UTC).isoformat(),
            "aborted_at": None,
            "abort_reason": None,
            "version_id": version_id,
            "started_by": actor,
        }

        try:
            async with self._client() as client:
                await client.set(self._key(policy_type), json.dumps(config))
        except RedisError as exc:
            logger.warning("canary_redis_error", op="start_canary", error=str(exc))

        logger.warning(
            "canary_started",
            policy_type=policy_type.value,
            version_id=version_id,
            traffic_pct=initial_pct,
            actor=actor,
        )

        return CanaryConfig(
            policy_type=policy_type,
            traffic_pct=initial_pct,
            stage=f"CANARY_{initial_pct}",
            started_at=config["started_at"],
            aborted_at=None,
            abort_reason=None,
            version_id=version_id,
        )

    async def advance_stage(self, policy_type: PolicyType, actor: str = "system") -> CanaryConfig:
        """
        Advance canary to the next traffic stage.
        10% → 25% → 50% → 100%
        """
        current = await self.get_canary_config(policy_type)
        if not current.is_active:
            raise ValueError(f"No active canary for {policy_type.value}")

        current_idx = (
            CANARY_STAGES.index(current.traffic_pct) if current.traffic_pct in CANARY_STAGES else -1
        )
        if current_idx == -1 or current_idx >= len(CANARY_STAGES) - 1:
            raise ValueError("Cannot advance beyond 100%. Promote the policy instead.")

        next_pct = CANARY_STAGES[current_idx + 1]

        try:
            async with self._client() as client:
                raw = await client.get(self._key(policy_type))
                config = json.loads(raw) if raw else {}
                config["traffic_pct"] = next_pct
                config["stage"] = f"CANARY_{next_pct}" if next_pct < 100 else "FULL"
                config["advanced_by"] = actor
                config["advanced_at"] = datetime.now(UTC).isoformat()
                await client.set(self._key(policy_type), json.dumps(config))
        except RedisError as exc:
            logger.warning("canary_redis_error", op="redis_op", error=str(exc))
        logger.warning(
            "canary_advanced",
            policy_type=policy_type.value,
            from_pct=current.traffic_pct,
            to_pct=next_pct,
            actor=actor,
        )

        return await self.get_canary_config(policy_type)

    async def abort_canary(
        self,
        policy_type: PolicyType,
        reason: str,
        actor: str = "auto_abort",
    ) -> CanaryConfig:
        """
        Abort canary rollout. Sets traffic_pct=0 immediately.
        All traffic returns to the main active policy.
        """
        try:
            async with self._client() as client:
                raw = await client.get(self._key(policy_type))
                config = json.loads(raw) if raw else {}
                config["traffic_pct"] = 0
                config["stage"] = "ABORTED"
                config["aborted_at"] = datetime.now(UTC).isoformat()
                config["abort_reason"] = reason
                config["aborted_by"] = actor
                await client.set(self._key(policy_type), json.dumps(config))
        except RedisError as exc:
            logger.warning("canary_redis_error", op="redis_op", error=str(exc))
        logger.critical(
            "canary_aborted",
            policy_type=policy_type.value,
            reason=reason,
            actor=actor,
        )

        return await self.get_canary_config(policy_type)

    async def record_canary_decision(
        self,
        policy_type: PolicyType,
        was_canary: bool,
        sla_violated: bool,
        fallback_used: bool,
    ) -> None:
        """Track canary metrics for auto-abort evaluation."""
        key = self._metrics_key(policy_type)
        try:
            async with self._client() as client:
                pipe = client.pipeline()
                pipe.hincrby(key, "total_decisions", 1)
                if was_canary:
                    pipe.hincrby(key, "canary_decisions", 1)
                    if sla_violated:
                        pipe.hincrby(key, "canary_sla_violations", 1)
                    if fallback_used:
                        pipe.hincrby(key, "canary_fallbacks", 1)
                pipe.expire(key, 3600)  # metrics expire after 1 hour
                await pipe.execute()
        except RedisError as exc:
            logger.warning("canary_redis_error", op="redis_op", error=str(exc))

    async def should_auto_abort(self, policy_type: PolicyType) -> tuple[bool, str | None]:
        """
        Check if canary metrics warrant auto-abort.

        Auto-aborts if:
        - SLA violation rate during canary > 5%
        - Fallback rate during canary > 2%
        - Minimum 20 canary decisions required before evaluation
        """
        key = self._metrics_key(policy_type)
        try:
            async with self._client() as client:
                metrics = await client.hgetall(key)
        except RedisError as exc:
            logger.warning("canary_redis_error", op="should_auto_abort", error=str(exc))
            return False, None

        if not metrics:
            return False, None

        canary_decisions = int(metrics.get("canary_decisions", 0))
        if canary_decisions < 20:
            return False, None  # insufficient data

        sla_violations = int(metrics.get("canary_sla_violations", 0))
        fallbacks = int(metrics.get("canary_fallbacks", 0))

        sla_rate = sla_violations / canary_decisions
        fallback_rate = fallbacks / canary_decisions

        if sla_rate > 0.05:
            return True, f"SLA violation rate {sla_rate:.1%} > 5% threshold"
        if fallback_rate > 0.02:
            return True, f"Fallback rate {fallback_rate:.1%} > 2% threshold"

        return False, None

    async def get_metrics(self, policy_type: PolicyType) -> dict[str, Any]:
        """Return canary metrics for monitoring."""
        key = self._metrics_key(policy_type)
        config = await self.get_canary_config(policy_type)

        try:
            async with self._client() as client:
                metrics = await client.hgetall(key)
        except RedisError as exc:
            logger.warning("canary_redis_error", op="get_metrics", error=str(exc))
            metrics = {}

        canary_decisions = int(metrics.get("canary_decisions", 0))

        return {
            "config": config.to_dict(),
            "canary_decisions": canary_decisions,
            "sla_violations": int(metrics.get("canary_sla_violations", 0)),
            "fallbacks": int(metrics.get("canary_fallbacks", 0)),
            "sla_violation_rate": (
                int(metrics.get("canary_sla_violations", 0)) / max(1, canary_decisions)
            ),
            "fallback_rate": (int(metrics.get("canary_fallbacks", 0)) / max(1, canary_decisions)),
        }
