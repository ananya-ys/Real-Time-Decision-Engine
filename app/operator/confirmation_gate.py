"""
ConfirmationGate — typed confirmation tokens for destructive operations.

WHY THIS EXISTS:
The review: "kill switch, baseline override are all one-click actions behind
a text reason box. That is not enough."

PATTERN:
  1. Operator requests a destructive action.
  2. System returns a confirmation_token + challenge string.
     Challenge: "Type 'KILL_SWITCH_GLOBAL_<timestamp_short>' to confirm"
  3. Operator types the exact challenge string + submits again with token.
  4. Gate validates token + challenge match → permits the action.
  5. Token expires in 5 minutes. One-time use.

This forces the operator to:
  - Read what they're about to do (the challenge names the action)
  - Actively type it (not just click)
  - Do it within a time window (prevents stale approvals)

HIGH-RISK ACTIONS require two separate confirmations from different users
(second-person approval), enforced at the approval service level.

RATE LIMITING:
  Max 3 confirmation attempts per operator per action type per 10 minutes.
  Prevents brute-force or accidental spam of destructive actions.
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

logger = structlog.get_logger(__name__)

_TOKEN_TTL = 300  # 5 minutes
_RATE_LIMIT_WINDOW = 600  # 10 minutes
_RATE_LIMIT_MAX = 3

# Actions classified by risk level
HIGH_RISK_ACTIONS = {
    "KILL_SWITCH_GLOBAL",
    "KILL_SWITCH_POLICY",
    "MAINTENANCE_ENTER",
}
MEDIUM_RISK_ACTIONS = {
    "FORCE_BASELINE",
    "FREEZE_EXPLORATION",
    "FREEZE_PROMOTION",
    "CANARY_ABORT",
    "POLICY_RETIRE",
}


class ConfirmationChallenge:
    """A challenge issued to the operator before a destructive action."""

    def __init__(
        self,
        action: str,
        token: str,
        challenge_string: str,
        expires_at: str,
        risk_level: str,
        blast_radius: str,
        requires_second_approval: bool,
    ) -> None:
        self.action = action
        self.token = token
        self.challenge_string = challenge_string
        self.expires_at = expires_at
        self.risk_level = risk_level
        self.blast_radius = blast_radius
        self.requires_second_approval = requires_second_approval

    def to_dict(self) -> dict[str, Any]:
        return {
            "confirmation_token": self.token,
            "challenge": f"Type exactly: {self.challenge_string}",
            "challenge_string": self.challenge_string,
            "expires_at": self.expires_at,
            "risk_level": self.risk_level,
            "blast_radius": self.blast_radius,
            "requires_second_approval": self.requires_second_approval,
            "instructions": (
                "Submit this confirmation_token + the exact challenge_string "
                "in your next request to execute the action."
            ),
        }


class ConfirmationGate:
    """
    Issues and validates typed confirmation challenges for destructive actions.

    Flow:
      1. issue_challenge(action, actor, metadata) → ConfirmationChallenge
      2. validate_and_consume(token, actor, typed_string) → bool
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._redis_url = settings.redis_url

    def _client(self) -> aioredis.Redis:  # type: ignore[type-arg]
        return aioredis.from_url(self._redis_url, decode_responses=True)

    def _token_key(self, token: str) -> str:
        return f"rtde:confirm:token:{token}"

    def _rate_key(self, actor: str, action: str) -> str:
        return f"rtde:confirm:rate:{actor}:{action}"

    def _risk_level(self, action: str) -> str:
        if action in HIGH_RISK_ACTIONS:
            return "CRITICAL"
        if action in MEDIUM_RISK_ACTIONS:
            return "HIGH"
        return "MEDIUM"

    def _blast_radius(self, action: str) -> str:
        descriptions = {
            "KILL_SWITCH_GLOBAL": "ALL ML policies disabled. 100% traffic falls back to baseline.",
            "KILL_SWITCH_POLICY": "Specific policy disabled. Traffic redistributes to baseline.",
            "MAINTENANCE_ENTER": "Baseline forced + exploration frozen + promotion frozen.",
            "FORCE_BASELINE": "ML policies bypassed. Training state preserved.",
            "FREEZE_EXPLORATION": "All policies exploit-only. No learning during freeze.",
            "CANARY_ABORT": "Canary traffic returns to 0%. Active policy unchanged.",
            "POLICY_RETIRE": "Policy version permanently retired. Cannot be undone.",
        }
        return descriptions.get(action, "Impact unknown — proceed with caution.")

    def _challenge_string(self, action: str, actor: str) -> str:
        """
        Generate a challenge string the operator must type exactly.
        Includes the action name so the operator must read what they're doing.
        """
        ts = datetime.now(UTC).strftime("%H%M")
        return f"{action}_{ts}"

    async def check_rate_limit(self, actor: str, action: str) -> bool:
        """Return True if request is within rate limit, False if exceeded."""
        key = self._rate_key(actor, action)
        try:
            async with self._client() as client:
                count_str = await client.get(key)
                count = int(count_str) if count_str else 0
                if count >= _RATE_LIMIT_MAX:
                    logger.warning(
                        "confirmation_rate_limit_exceeded",
                        actor=actor,
                        action=action,
                        count=count,
                    )
                    return False
                pipe = client.pipeline()
                pipe.incr(key)
                pipe.expire(key, _RATE_LIMIT_WINDOW)
                await pipe.execute()
                return True
        except RedisError as exc:
            logger.warning("confirmation_gate_redis_error", error=str(exc))

    async def issue_challenge(
        self,
        action: str,
        actor: str,
        metadata: dict[str, Any] | None = None,
    ) -> ConfirmationChallenge:
        """
        Issue a typed confirmation challenge for a destructive action.

        Args:
            action: The action identifier (e.g., KILL_SWITCH_GLOBAL).
            actor: Who is requesting the confirmation.
            metadata: Optional context (policy_type, reason, etc.).

        Returns:
            ConfirmationChallenge with token and challenge string.

        Raises:
            RateLimitError: If operator has exceeded the rate limit.
        """
        within_limit = await self.check_rate_limit(actor, action)
        if not within_limit:
            raise ValueError(
                f"Rate limit exceeded for {action}. "
                f"Max {_RATE_LIMIT_MAX} confirmations per {_RATE_LIMIT_WINDOW // 60} minutes."
            )

        token = secrets.token_urlsafe(32)
        challenge = self._challenge_string(action, actor)
        risk = self._risk_level(action)
        requires_second = action in HIGH_RISK_ACTIONS

        # Store token in Redis
        token_data = {
            "action": action,
            "actor": actor,
            "challenge": challenge,
            "risk_level": risk,
            "metadata": json.dumps(metadata or {}),
            "issued_at": datetime.now(UTC).isoformat(),
            "used": "false",
        }
        try:
            async with self._client() as client:
                key = self._token_key(token)
                await client.hset(key, mapping=token_data)
                await client.expire(key, _TOKEN_TTL)
        except RedisError as exc:
            logger.warning("confirmation_gate_redis_error", error=str(exc))
        expires_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        ch = ConfirmationChallenge(
            action=action,
            token=token,
            challenge_string=challenge,
            expires_at=expires_at,
            risk_level=risk,
            blast_radius=self._blast_radius(action),
            requires_second_approval=requires_second,
        )

        logger.warning(
            "confirmation_challenge_issued",
            action=action,
            actor=actor,
            risk_level=risk,
            token_prefix=token[:8],
        )
        return ch

    async def validate_and_consume(
        self,
        token: str,
        actor: str,
        typed_string: str,
    ) -> tuple[bool, str, dict[str, Any]]:
        """
        Validate a confirmation token and typed string.

        Returns:
            (valid, action, metadata) if valid.
            (False, "", {}) if invalid.

        The token is consumed on first valid use (one-time).
        """
        key = self._token_key(token)
        try:
            async with self._client() as client:
                data = await client.hgetall(key)
        except RedisError as exc:
            logger.warning("confirmation_gate_redis_error", error=str(exc))
        if not data:
            logger.warning("confirmation_token_not_found", token_prefix=token[:8])
            return False, "", {}

        if data.get("used") == "true":
            logger.warning("confirmation_token_already_used", token_prefix=token[:8])
            return False, "", {}

        if data.get("actor") != actor:
            logger.warning(
                "confirmation_actor_mismatch",
                expected=data.get("actor"),
                got=actor,
            )
            return False, "", {}

        expected_challenge = data.get("challenge", "")
        if typed_string.strip() != expected_challenge:
            logger.warning(
                "confirmation_challenge_mismatch",
                expected=expected_challenge,
                got=typed_string.strip(),
            )
            return False, "", {}

        # Mark as used (idempotency)
        try:
            async with self._client() as client:
                await client.hset(key, "used", "true")
        except RedisError as exc:
            logger.warning("confirmation_gate_redis_error", error=str(exc))
        action = data.get("action", "")
        metadata = json.loads(data.get("metadata", "{}"))

        logger.warning(
            "confirmation_validated",
            action=action,
            actor=actor,
            token_prefix=token[:8],
        )
        return True, action, metadata
