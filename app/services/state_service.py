"""
StateService — state reads with row-level locking and optimistic concurrency.

WHY THIS EXISTS:
- RACE CONDITION: Two agents read state simultaneously, both see version=0, both proceed.
  Result: conflicting scaling decisions, corrupted state, silent data loss.
- Python asyncio.Lock does NOT fix this — it only works within a single process.
  Under load, multiple workers run concurrently. Only the DATABASE can serialize access.

THE FIX:
1. SELECT ... FOR UPDATE NOWAIT — acquires a PostgreSQL row-level lock.
   If another transaction holds the lock → raises LockUnavailableError immediately.
   Caller retries. No deadlock. No wait. First writer wins.

2. Optimistic concurrency — UPDATE WHERE version = expected.
   If 0 rows updated → another writer changed the row → StateVersionConflictError.
   Caller retries. Works across processes without persistent locks.

INDUSTRY PARALLEL:
- How banks prevent double charges.
- How ticketing systems prevent double bookings.
- How RTDE prevents conflicting concurrent scaling decisions.

WHAT BREAKS IF WRONG:
- asyncio.Lock alone: works in one process, fails with multiple workers.
- No NOWAIT: lock contention causes cascading waits → all requests pile up → timeout storm.
- No optimistic check: two writes commit on same version → silent state corruption.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import structlog
from sqlalchemy import select, text, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.exceptions import LockUnavailableError, StateVersionConflictError
from app.models.environment_state import EnvironmentState
from app.schemas.state import SystemState

logger = structlog.get_logger(__name__)


class StateService:
    """
    Manages environment state reads and writes with concurrency safety.

    Two-layer protection:
    1. FOR UPDATE NOWAIT — prevents concurrent reads from both proceeding to write.
    2. Optimistic version check — detects conflicting writes that slipped through.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._min_instances = settings.min_instances
        self._max_instances = settings.max_instances

    async def read_state_with_lock(
        self,
        state_id: uuid.UUID,
        db: AsyncSession,
    ) -> EnvironmentState:
        """
        Read an environment state row, acquiring a row-level lock.

        Uses SELECT ... FOR UPDATE NOWAIT:
        - FOR UPDATE: acquires exclusive row lock within the transaction.
        - NOWAIT: if another transaction holds the lock, raises immediately.
          This prevents cascading waits. Caller must retry.

        Args:
            state_id: UUID of the state row to lock.
            db: Active database session (must be in a transaction).

        Returns:
            EnvironmentState with exclusive lock held.

        Raises:
            LockUnavailableError: If row is already locked by another transaction.
        """
        try:
            result = await db.execute(
                select(EnvironmentState)
                .where(EnvironmentState.id == state_id)
                .with_for_update(nowait=True)
            )
            state = result.scalar_one_or_none()
            if state is None:
                raise LockUnavailableError(f"State {state_id} not found")
            logger.debug(
                "state_lock_acquired",
                state_id=str(state_id),
                version=state.version,
            )
            return state
        except OperationalError as exc:
            # PostgreSQL error code 55P03 = lock_not_available
            logger.warning(
                "state_lock_unavailable",
                state_id=str(state_id),
                error=str(exc),
            )
            raise LockUnavailableError(
                f"State {state_id} is locked by another transaction"
            ) from exc

    async def update_state_optimistic(
        self,
        state_id: uuid.UUID,
        new_instance_count: int,
        expected_version: int,
        db: AsyncSession,
    ) -> EnvironmentState:
        """
        Update state with optimistic concurrency check.

        The WHERE version = expected_version clause ensures we only update
        if nobody else has written to this row since we read it.

        If 0 rows updated → version mismatch → another writer won → retry.
        If 1 row updated → we won → commit.

        Args:
            state_id: UUID of the state row to update.
            new_instance_count: The target instance count after scaling.
            expected_version: The version we read. Must still match.
            db: Active database session.

        Raises:
            StateVersionConflictError: If version changed (another writer won).
        """
        # Clip before writing — safety boundary enforced at DB write time
        clipped_count = max(self._min_instances, min(new_instance_count, self._max_instances))

        result = await db.execute(
            update(EnvironmentState)
            .where(
                EnvironmentState.id == state_id,
                EnvironmentState.version == expected_version,  # optimistic check
            )
            .values(
                instance_count=clipped_count,
                version=expected_version + 1,  # increment version
                timestamp=datetime.now(UTC),
            )
            .returning(EnvironmentState)
        )
        updated = result.scalar_one_or_none()

        if updated is None:
            logger.warning(
                "state_version_conflict",
                state_id=str(state_id),
                expected_version=expected_version,
            )
            raise StateVersionConflictError(
                f"State {state_id} version conflict: expected={expected_version}"
            )

        logger.info(
            "state_updated",
            state_id=str(state_id),
            new_count=clipped_count,
            new_version=expected_version + 1,
        )
        return updated

    async def create_state(
        self,
        system_state: SystemState,
        db: AsyncSession,
    ) -> EnvironmentState:
        """
        Persist a new SystemState observation to the database.

        Called before every decision so the ScalingAction has a FK reference.
        """
        env_state = EnvironmentState(
            cpu_utilization=system_state.cpu_utilization,
            request_rate=system_state.request_rate,
            p99_latency_ms=system_state.p99_latency_ms,
            instance_count=system_state.instance_count,
            hour_of_day=system_state.hour_of_day,
            day_of_week=system_state.day_of_week,
            traffic_regime=system_state.traffic_regime.value,
            version=0,
        )
        db.add(env_state)
        await db.flush()  # get UUID without committing
        return env_state

    async def get_latest_state(self, db: AsyncSession) -> EnvironmentState | None:
        """Get the most recent environment state without locking."""
        result = await db.execute(
            select(EnvironmentState).order_by(EnvironmentState.timestamp.desc()).limit(1)
        )
        return result.scalar_one_or_none()

    async def verify_db_connectivity(self, db: AsyncSession) -> bool:
        """Health check — verify DB is reachable."""
        try:
            await db.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
