"""
AuditHashChain — immutable audit trail with cryptographic hash chaining.

WHY THIS EXISTS:
The review: "immutable audit trail with request IDs and before/after state."

WITHOUT HASH CHAIN:
  An attacker or compromised admin can DELETE rows from operator_events.
  Postgres has no built-in immutability. "Immutable" without crypto = theater.

WITH HASH CHAIN:
  Each OperatorEvent contains:
    - Its own content hash (SHA-256 of all fields)
    - The previous event's hash (chain_prev_hash)

  To forge or delete event N, you must also recompute events N+1, N+2, ...
  Any break in the chain is detectable by verify_chain().

  Like a blockchain — but without the cryptocurrency nonsense.

CHAIN STRUCTURE:
  Event 1: prev_hash="genesis", self_hash=H(E1)
  Event 2: prev_hash=H(E1), self_hash=H(E2+H(E1))
  Event 3: prev_hash=H(E2+H(E1)), self_hash=H(E3+prev_hash)
  ...
  If Event 2 is deleted or modified → Event 3's prev_hash won't match.

VERIFICATION:
  GET /api/v1/audit/chain/verify → returns "intact" or "broken at event N"

PERFORMANCE:
  Hash computation: ~0.1ms per event.
  Chain verification: O(N) where N is event count.
  Not called on hot path — only on audit export or manual verification.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.operator_event import OperatorEvent

logger = structlog.get_logger(__name__)

_GENESIS_HASH = "0" * 64  # sentinel for the first event in chain


def compute_event_hash(
    event_id: str,
    actor: str,
    action: str,
    target: str | None,
    reason: str,
    created_at: str,
    prev_hash: str,
) -> str:
    """Compute deterministic SHA-256 hash of an event's content."""
    content = json.dumps(
        {
            "id": event_id,
            "actor": actor,
            "action": action,
            "target": target or "",
            "reason": reason,
            "created_at": created_at,
            "prev_hash": prev_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(content.encode()).hexdigest()


class AuditHashChain:
    """
    Maintains and verifies the cryptographic audit event chain.

    Usage:
        chain = AuditHashChain()
        # After creating an OperatorEvent:
        await chain.stamp(event, db)

        # To verify chain integrity:
        result = await chain.verify(db)
        if not result["intact"]:
            alert("AUDIT CHAIN BROKEN at " + result["broken_at"])
    """

    async def get_latest_hash(self, db: AsyncSession) -> str:
        """Get the hash of the most recent event (for chaining new events)."""
        result = await db.execute(
            select(OperatorEvent)
            .where(OperatorEvent.chain_hash.is_not(None))
            .order_by(OperatorEvent.created_at.desc())
            .limit(1)
        )
        latest = result.scalar_one_or_none()
        if latest is None:
            return _GENESIS_HASH
        return latest.chain_hash or _GENESIS_HASH

    async def stamp(self, event: OperatorEvent, db: AsyncSession) -> None:
        """
        Compute and store the event's hash and chain link.

        Must be called AFTER the event is flushed (has an ID).
        Must be called BEFORE db.commit().
        """
        prev_hash = await self.get_latest_hash(db)

        content_hash = compute_event_hash(
            event_id=str(event.id),
            actor=event.actor,
            action=event.action,
            target=event.target,
            reason=event.reason,
            created_at=event.created_at.isoformat()
            if event.created_at
            else datetime.now(UTC).isoformat(),
            prev_hash=prev_hash,
        )

        event.chain_hash = content_hash
        event.chain_prev_hash = prev_hash

        logger.debug(
            "audit_event_stamped",
            event_id=str(event.id),
            action=event.action,
            hash_prefix=content_hash[:12],
            prev_prefix=prev_hash[:12],
        )

    async def verify(self, db: AsyncSession, limit: int = 1000) -> dict[str, Any]:
        """
        Verify the integrity of the audit hash chain.

        Returns:
            {
                "intact": bool,
                "events_checked": int,
                "broken_at": event_id or None,
                "break_reason": description or None,
            }

        A broken chain means either:
        1. An event was deleted.
        2. An event was modified.
        3. Events were inserted out of order.
        """
        result = await db.execute(
            select(OperatorEvent)
            .where(OperatorEvent.chain_hash.is_not(None))
            .order_by(OperatorEvent.created_at.asc())
            .limit(limit)
        )
        events = result.scalars().all()

        if not events:
            return {
                "intact": True,
                "events_checked": 0,
                "broken_at": None,
                "break_reason": None,
                "message": "No chained events to verify.",
            }

        prev_hash = _GENESIS_HASH
        for i, event in enumerate(events):
            # Verify prev_hash matches
            if event.chain_prev_hash != prev_hash:
                return {
                    "intact": False,
                    "events_checked": i,
                    "broken_at": str(event.id),
                    "break_reason": (
                        f"Event {event.id} has prev_hash={event.chain_prev_hash[:16]}... "
                        f"but expected {prev_hash[:16]}... "
                        "This indicates an event was deleted or modified."
                    ),
                }

            # Verify own hash
            expected_hash = compute_event_hash(
                event_id=str(event.id),
                actor=event.actor,
                action=event.action,
                target=event.target,
                reason=event.reason,
                created_at=event.created_at.isoformat(),
                prev_hash=event.chain_prev_hash or _GENESIS_HASH,
            )

            if event.chain_hash != expected_hash:
                return {
                    "intact": False,
                    "events_checked": i,
                    "broken_at": str(event.id),
                    "break_reason": (
                        f"Event {event.id} hash mismatch. "
                        "Content of this event was modified after creation."
                    ),
                }

            prev_hash = event.chain_hash

        return {
            "intact": True,
            "events_checked": len(events),
            "broken_at": None,
            "break_reason": None,
            "latest_hash": prev_hash[:16] + "...",
            "message": f"Chain verified: {len(events)} events intact.",
        }
