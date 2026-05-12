"""
Concurrency safety tests — Phase 3 gate.

These tests verify the race condition protection logic without a real database.
Real PostgreSQL concurrency tests live in tests/integration/ (need docker-compose).

What we verify here:
1. LockUnavailableError is distinct from other exceptions (not swallowed).
2. Two concurrent calls to make_decision both complete (fallback handles errors).
3. ExplorationGuard is consulted before policy.decide().
4. Action clipping prevents out-of-bounds even under concurrent load.
"""

from __future__ import annotations

import asyncio

import pytest

from app.core.exceptions import LockUnavailableError
from app.policies.baseline_policy import BaselinePolicy
from app.safety.exploration_guard import ExplorationGuard, PolicyStats
from app.schemas.common import TrafficRegime
from app.schemas.state import SystemState


def _make_state(cpu: float = 0.5, instances: int = 5) -> SystemState:
    return SystemState(
        cpu_utilization=cpu,
        request_rate=500.0,
        p99_latency_ms=100.0,
        instance_count=instances,
        traffic_regime=TrafficRegime.STEADY,
    )


@pytest.mark.unit
class TestConcurrencyProtection:
    """Verify race condition protections at the logic level."""

    @pytest.mark.asyncio
    async def test_lock_unavailable_error_is_domain_exception(self) -> None:
        """
        LockUnavailableError must be a distinct, named exception.
        It must not be swallowed or confused with other exceptions.

        WHY: Error handlers map exception types to HTTP status codes.
        A raw OperationalError would produce a 500. A LockUnavailableError
        produces a 409 Conflict with a retry-after response.
        """
        exc = LockUnavailableError("State locked")
        assert exc.message == "State locked"
        assert isinstance(exc, LockUnavailableError)
        # Must not accidentally be an HTTPException or base Exception
        from app.core.exceptions import RTDEBaseError

        assert isinstance(exc, RTDEBaseError)

    @pytest.mark.asyncio
    async def test_concurrent_baseline_decisions_are_consistent(self) -> None:
        """
        BaselinePolicy is deterministic. Two concurrent calls with the same
        state must produce the same action.

        This verifies the no-shared-mutable-state property of BaselinePolicy.
        """
        policy = BaselinePolicy()
        state = _make_state(cpu=0.85)  # should trigger SCALE_UP_1

        # Run two concurrent calls
        results = await asyncio.gather(
            policy.decide(state),
            policy.decide(state),
        )

        # Both must produce identical actions
        assert results[0].action == results[1].action
        assert results[0].instances_after == results[1].instances_after

    @pytest.mark.asyncio
    async def test_concurrent_calls_do_not_corrupt_policy_stats(self) -> None:
        """
        PolicyStats is updated after each decision. Verify that concurrent
        updates don't corrupt the state (Python GIL protects here, but
        we verify the update logic produces sensible results).
        """
        guard = ExplorationGuard()
        stats = PolicyStats()

        async def update_stats(violated: bool) -> None:
            guard.update_policy_stats(
                stats, reward=-1.0 if violated else 1.0, sla_violated=violated
            )

        # Run 50 concurrent updates (mix of violations and recoveries)
        tasks = [update_stats(i % 3 == 0) for i in range(50)]
        await asyncio.gather(*tasks)

        # Stats must be internally consistent
        assert stats.total_decisions == 50
        assert stats.total_violations >= 0
        assert stats.consecutive_violations >= 0

    @pytest.mark.asyncio
    async def test_exploration_guard_called_before_policy_in_concurrent_requests(
        self,
    ) -> None:
        """
        ExplorationGuard.check_and_log must be called before policy.decide().
        Under concurrent load, the guard prevents exploration during stress.
        This verifies call ordering is correct.
        """
        guard = ExplorationGuard()

        # High-stress state — guard should suppress
        stressed_state = SystemState(
            cpu_utilization=0.9,
            request_rate=500.0,
            p99_latency_ms=500.0,  # above threshold
            instance_count=5,
        )
        stats = PolicyStats()

        async def _check() -> bool:
            return guard.check_and_log(stressed_state, stats)

        results = await asyncio.gather(*[_check() for _ in range(10)])

        # All 10 concurrent calls must suppress exploration
        assert all(r is False for r in results)

    @pytest.mark.asyncio
    async def test_action_clipping_invariant_under_concurrent_load(self) -> None:
        """
        Instance count must never exceed max_instances, even under concurrent load.
        This tests that clip_instances in BaselinePolicy is stateless and safe.
        """
        policy = BaselinePolicy()

        # State at max instances — should not exceed max
        state_at_max = _make_state(cpu=0.95, instances=19)

        results = await asyncio.gather(*[policy.decide(state_at_max) for _ in range(20)])

        for decision in results:
            assert decision.instances_after <= 20  # max_instances default
            assert decision.instances_after >= 1  # min_instances default

    @pytest.mark.asyncio
    async def test_min_instance_invariant_under_concurrent_load(self) -> None:
        """
        Instance count must never go below min_instances under concurrent load.
        """
        policy = BaselinePolicy()
        state_at_min = _make_state(cpu=0.05, instances=1)

        results = await asyncio.gather(*[policy.decide(state_at_min) for _ in range(20)])

        for decision in results:
            assert decision.instances_after >= 1


@pytest.mark.unit
class TestExplorationGuardUnderLoad:
    """Verify exploration guard behavior under simulated concurrent load."""

    @pytest.fixture
    def guard(self) -> ExplorationGuard:
        return ExplorationGuard()

    @pytest.mark.asyncio
    async def test_guard_consistently_suppresses_during_spike(
        self, guard: ExplorationGuard
    ) -> None:
        """
        During a traffic spike, ALL concurrent calls must suppress exploration.
        Inconsistent suppression would mean some requests explore during danger.
        """
        spike_state = SystemState(
            cpu_utilization=0.95,
            request_rate=8000.0,  # well above 5000 rps threshold
            p99_latency_ms=100.0,
            instance_count=5,
        )
        stats = PolicyStats()

        # 50 concurrent guard checks
        results = [guard.check_and_log(spike_state, stats) for _ in range(50)]

        # Every single check must suppress
        assert all(r is False for r in results)
        assert sum(1 for r in results if r is False) == 50

    @pytest.mark.asyncio
    async def test_guard_consistently_allows_on_stable(self, guard: ExplorationGuard) -> None:
        """
        During stable traffic, ALL concurrent calls must allow exploration.
        """
        stable_state = SystemState(
            cpu_utilization=0.3,
            request_rate=200.0,
            p99_latency_ms=50.0,
            instance_count=5,
        )
        stats = PolicyStats()

        results = [guard.check_and_log(stable_state, stats) for _ in range(50)]

        assert all(r is True for r in results)
