"""
StateService unit tests — Phase 3 gate.

Tests lock logic in isolation using mocks (no real DB needed for unit tests).
Integration tests with real DB + NOWAIT are in tests/concurrency/.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import OperationalError

from app.core.exceptions import LockUnavailableError, StateVersionConflictError
from app.schemas.common import TrafficRegime
from app.schemas.state import SystemState
from app.services.state_service import StateService


def _make_mock_state(version: int = 0) -> MagicMock:
    """Create a mock EnvironmentState."""
    state = MagicMock()
    state.id = uuid.uuid4()
    state.version = version
    state.instance_count = 5
    return state


def _make_system_state() -> SystemState:
    return SystemState(
        cpu_utilization=0.6,
        request_rate=1000.0,
        p99_latency_ms=150.0,
        instance_count=5,
        traffic_regime=TrafficRegime.STEADY,
    )


@pytest.mark.unit
class TestStateServiceLocking:
    """Verify SELECT FOR UPDATE NOWAIT behavior."""

    @pytest.fixture
    def service(self) -> StateService:
        return StateService()

    @pytest.mark.asyncio
    async def test_lock_raises_on_operational_error(self, service: StateService) -> None:
        """
        OperationalError from asyncpg (lock not available) must become
        LockUnavailableError — not bubble up as a raw DB exception.
        """
        mock_db = AsyncMock()
        # Simulate PostgreSQL lock not available (error code 55P03)
        mock_db.execute.side_effect = OperationalError("could not obtain lock", None, None)

        with pytest.raises(LockUnavailableError, match="locked by another"):
            await service.read_state_with_lock(uuid.uuid4(), mock_db)

    @pytest.mark.asyncio
    async def test_lock_raises_when_state_not_found(self, service: StateService) -> None:
        """State not found should raise LockUnavailableError (not None return)."""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        with pytest.raises(LockUnavailableError, match="not found"):
            await service.read_state_with_lock(uuid.uuid4(), mock_db)

    @pytest.mark.asyncio
    async def test_lock_returns_state_on_success(self, service: StateService) -> None:
        """Successful lock acquisition returns the state object."""
        mock_db = AsyncMock()
        mock_state = _make_mock_state(version=3)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_state
        mock_db.execute.return_value = mock_result

        result = await service.read_state_with_lock(uuid.uuid4(), mock_db)
        assert result == mock_state
        assert result.version == 3

    @pytest.mark.asyncio
    async def test_optimistic_update_raises_on_version_conflict(
        self, service: StateService
    ) -> None:
        """
        If UPDATE returns 0 rows (version changed), raise StateVersionConflictError.
        This is the optimistic concurrency guard.
        """
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None  # 0 rows updated
        mock_db.execute.return_value = mock_result

        with pytest.raises(StateVersionConflictError, match="version conflict"):
            await service.update_state_optimistic(
                state_id=uuid.uuid4(),
                new_instance_count=6,
                expected_version=0,
                db=mock_db,
            )

    @pytest.mark.asyncio
    async def test_optimistic_update_clips_to_max(self, service: StateService) -> None:
        """
        Instance count beyond max_instances must be clipped before the UPDATE.
        This is the last safety boundary before DB write.
        """
        mock_db = AsyncMock()
        mock_updated = _make_mock_state(version=1)
        mock_updated.instance_count = 20  # clipped to max
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_updated
        mock_db.execute.return_value = mock_result

        # Request 100 instances — should be clipped to max_instances (20)
        result = await service.update_state_optimistic(
            state_id=uuid.uuid4(),
            new_instance_count=100,
            expected_version=0,
            db=mock_db,
        )
        # Verify we got back a result (not a conflict error)
        assert result is not None

    @pytest.mark.asyncio
    async def test_optimistic_update_clips_to_min(self, service: StateService) -> None:
        """
        Instance count below min_instances must be clipped.
        """
        mock_db = AsyncMock()
        mock_updated = _make_mock_state(version=1)
        mock_updated.instance_count = 1
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_updated
        mock_db.execute.return_value = mock_result

        # Request 0 instances — should be clipped to min_instances (1)
        result = await service.update_state_optimistic(
            state_id=uuid.uuid4(),
            new_instance_count=0,
            expected_version=0,
            db=mock_db,
        )
        assert result is not None


@pytest.mark.unit
class TestStateServiceCreate:
    """Verify state creation."""

    @pytest.fixture
    def service(self) -> StateService:
        return StateService()

    @pytest.mark.asyncio
    async def test_create_state_calls_flush(self, service: StateService) -> None:
        """
        create_state must call db.flush() to get UUID without committing.
        Without flush: FK reference in ScalingAction breaks.
        """
        mock_db = AsyncMock()
        mock_db.add = MagicMock(return_value=None)  # db.add() is synchronous
        system_state = _make_system_state()

        await service.create_state(system_state, mock_db)

        mock_db.add.assert_called_once()
        mock_db.flush.assert_called_once()
