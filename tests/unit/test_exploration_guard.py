"""
ExplorationGuard tests — Phase 3 gate.

Verifies:
- Guard suppresses exploration when p99 latency is high
- Guard suppresses exploration during traffic spikes
- Guard suppresses exploration when SLA violation rate > threshold
- Guard suppresses exploration during consecutive violation streaks
- Guard allows exploration when system is stable
- Suppression reason correctly identifies the trigger
- PolicyStats update correctly tracks violation rate
"""

from __future__ import annotations

import pytest

from app.safety.exploration_guard import ExplorationGuard, PolicyStats
from app.schemas.common import SuppressionReason, TrafficRegime
from app.schemas.state import SystemState


def _stable_state() -> SystemState:
    """Create a state that should always allow exploration."""
    return SystemState(
        cpu_utilization=0.4,
        request_rate=500.0,
        p99_latency_ms=100.0,
        instance_count=5,
        traffic_regime=TrafficRegime.STEADY,
    )


def _stable_stats() -> PolicyStats:
    """Create stats showing a healthy system."""
    return PolicyStats(
        sla_violation_rate_5min=0.0,
        consecutive_violations=0,
        total_decisions=100,
        total_violations=0,
    )


@pytest.mark.unit
class TestExplorationGuardSuppression:
    """Verify guard blocks exploration under all stress conditions."""

    @pytest.fixture
    def guard(self) -> ExplorationGuard:
        return ExplorationGuard()

    def test_stable_system_allows_exploration(self, guard: ExplorationGuard) -> None:
        """Exploration must be allowed when system is healthy."""
        allowed, reason = guard.should_explore(_stable_state(), _stable_stats())
        assert allowed is True
        assert reason is None

    def test_high_latency_suppresses(self, guard: ExplorationGuard) -> None:
        """p99 > latency_warning_ms → suppress exploration."""
        stressed_state = SystemState(
            cpu_utilization=0.5,
            request_rate=500.0,
            p99_latency_ms=500.0,  # above 400ms default threshold
            instance_count=5,
        )
        allowed, reason = guard.should_explore(stressed_state, _stable_stats())
        assert allowed is False
        assert reason == SuppressionReason.HIGH_LATENCY

    def test_high_rps_suppresses(self, guard: ExplorationGuard) -> None:
        """request_rate > high_load_rps → suppress exploration."""
        high_load = SystemState(
            cpu_utilization=0.5,
            request_rate=6000.0,  # above 5000 rps default threshold
            p99_latency_ms=100.0,
            instance_count=5,
        )
        allowed, reason = guard.should_explore(high_load, _stable_stats())
        assert allowed is False
        assert reason == SuppressionReason.HIGH_LOAD

    def test_high_sla_violation_rate_suppresses(self, guard: ExplorationGuard) -> None:
        """sla_violation_rate > threshold → suppress exploration."""
        bad_stats = PolicyStats(
            sla_violation_rate_5min=0.05,  # above 3% default threshold
            consecutive_violations=0,
        )
        allowed, reason = guard.should_explore(_stable_state(), bad_stats)
        assert allowed is False
        assert reason == SuppressionReason.SLA_VIOLATION_STREAK

    def test_consecutive_violations_suppresses(self, guard: ExplorationGuard) -> None:
        """consecutive_violations >= threshold → suppress exploration."""
        streak_stats = PolicyStats(
            sla_violation_rate_5min=0.01,
            consecutive_violations=5,  # above 3 default threshold
        )
        allowed, reason = guard.should_explore(_stable_state(), streak_stats)
        assert allowed is False
        assert reason == SuppressionReason.SLA_VIOLATION_STREAK

    def test_latency_takes_priority_over_load(self, guard: ExplorationGuard) -> None:
        """When multiple conditions trigger, latency is checked first."""
        both_stressed = SystemState(
            cpu_utilization=0.9,
            request_rate=7000.0,  # high load
            p99_latency_ms=600.0,  # high latency
            instance_count=5,
        )
        allowed, reason = guard.should_explore(both_stressed, _stable_stats())
        assert allowed is False
        # Latency check comes first in the implementation
        assert reason == SuppressionReason.HIGH_LATENCY

    def test_exactly_at_threshold_allows(self, guard: ExplorationGuard) -> None:
        """At exactly the threshold (not above) → exploration allowed."""
        at_threshold = SystemState(
            cpu_utilization=0.5,
            request_rate=500.0,
            p99_latency_ms=400.0,  # exactly at 400ms, not above
            instance_count=5,
        )
        allowed, reason = guard.should_explore(at_threshold, _stable_stats())
        assert allowed is True
        assert reason is None


@pytest.mark.unit
class TestExplorationGuardLogging:
    """Verify check_and_log returns correct boolean."""

    @pytest.fixture
    def guard(self) -> ExplorationGuard:
        return ExplorationGuard()

    def test_logs_and_returns_true_when_stable(self, guard: ExplorationGuard) -> None:
        result = guard.check_and_log(_stable_state(), _stable_stats())
        assert result is True

    def test_logs_and_returns_false_when_stressed(self, guard: ExplorationGuard) -> None:
        stressed = SystemState(
            cpu_utilization=0.5,
            request_rate=500.0,
            p99_latency_ms=999.0,
            instance_count=5,
        )
        result = guard.check_and_log(stressed, _stable_stats())
        assert result is False


@pytest.mark.unit
class TestPolicyStatsUpdate:
    """Verify PolicyStats tracking logic."""

    @pytest.fixture
    def guard(self) -> ExplorationGuard:
        return ExplorationGuard()

    def test_violation_increments_consecutive_count(self, guard: ExplorationGuard) -> None:
        """SLA violation must increment consecutive_violations counter."""
        stats = _stable_stats()
        updated = guard.update_policy_stats(stats, reward=-5.0, sla_violated=True)
        assert updated.consecutive_violations == 1
        assert updated.total_violations == 1
        assert updated.total_decisions == 101

    def test_recovery_resets_consecutive_count(self, guard: ExplorationGuard) -> None:
        """Recovery from SLA breach must reset consecutive_violations to 0."""
        stats = PolicyStats(consecutive_violations=5, sla_violation_rate_5min=0.1)
        updated = guard.update_policy_stats(stats, reward=1.0, sla_violated=False)
        assert updated.consecutive_violations == 0

    def test_violation_rate_decays_on_recovery(self, guard: ExplorationGuard) -> None:
        """SLA violation rate should decay exponentially on recovery."""
        stats = PolicyStats(sla_violation_rate_5min=0.1)
        updated = guard.update_policy_stats(stats, reward=1.0, sla_violated=False)
        assert updated.sla_violation_rate_5min < 0.1

    def test_recent_rewards_buffer_capped(self, guard: ExplorationGuard) -> None:
        """recent_rewards buffer must not grow unboundedly."""
        stats = PolicyStats(recent_rewards=list(range(300)))
        updated = guard.update_policy_stats(stats, reward=1.0, sla_violated=False)
        assert len(updated.recent_rewards) == 300  # stays at cap after adding 1, removing 1

    def test_multiple_violations_accumulate(self, guard: ExplorationGuard) -> None:
        """Three consecutive violations should build up the streak."""
        stats = _stable_stats()
        for _ in range(3):
            stats = guard.update_policy_stats(stats, reward=-5.0, sla_violated=True)
        assert stats.consecutive_violations == 3
        assert stats.total_violations == 3
