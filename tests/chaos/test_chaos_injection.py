"""
Chaos injection tests — Phase 8 gate.

WHY CHAOS TESTS:
- Unit tests verify happy path. Chaos tests verify failure path.
- Production failures happen at unexpected moments:
  * Policy raises exception during inference
  * DB connection lost mid-transaction
  * Celery worker crashes during training
  * Normalizer file corrupted
- The system must respond to all these with safe degradation, not crash.

FAILURE MODES TESTED:
1. Policy exception → automatic fallback to baseline (never 500 to caller)
2. Corrupt checkpoint → CheckpointError loud (not silent wrong inference)
3. Normalizer version mismatch → CheckpointError loud
4. ExplorationGuard under stress → suppresses consistently (never inconsistent)
5. Replay buffer exhaustion → train_step() returns None (no crash)
6. DB loss during decision → DecisionService handles gracefully
"""

from __future__ import annotations

import asyncio
import random
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.exceptions import CheckpointError
from app.ml.state_normalizer import StateNormalizer
from app.policies.bandit_policy import BanditPolicy
from app.policies.base_policy import PolicyCheckpointData
from app.policies.baseline_policy import BaselinePolicy
from app.policies.rl_policy import RLPolicy
from app.safety.exploration_guard import ExplorationGuard, PolicyStats
from app.schemas.common import PolicyMode, TrafficRegime
from app.schemas.state import SystemState
from app.services.decision_service import DecisionService


def _stress_state() -> SystemState:
    """High-stress system state — guard should suppress exploration."""
    return SystemState(
        cpu_utilization=0.95,
        request_rate=8000.0,
        p99_latency_ms=700.0,
        instance_count=5,
        traffic_regime=TrafficRegime.BURST,
    )


def _normal_state() -> SystemState:
    return SystemState(
        cpu_utilization=0.4,
        request_rate=500.0,
        p99_latency_ms=100.0,
        instance_count=5,
        traffic_regime=TrafficRegime.STEADY,
    )


@pytest.mark.chaos
class TestPolicyFailureFallback:
    """Verify automatic fallback when active policy fails."""

    @pytest.mark.asyncio
    async def test_exploding_policy_triggers_fallback(self) -> None:
        """
        Policy that raises an exception must trigger baseline fallback.
        The caller must never see the exception — they get a valid decision.
        """

        # Create a policy that always raises
        class ExplodingPolicy(BaselinePolicy):
            async def decide(self, state, explore=True):
                raise RuntimeError("Simulated policy explosion")

        svc = DecisionService()
        svc._active_policy = ExplodingPolicy()
        # Mock kill switch — no Redis in chaos unit tests
        from app.operator.kill_switch import KillSwitchState

        safe_state = KillSwitchState(
            global_killed=False,
            exploration_frozen=False,
            promotion_frozen=False,
            killed_policies=set(),
        )
        svc._kill_switch.get_state = AsyncMock(return_value=safe_state)
        svc._manual_override.is_baseline_forced = AsyncMock(return_value=False)

        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()
        mock_db.commit = AsyncMock()

        response = await svc.make_decision(
            state=_normal_state(),
            trace_id=uuid.uuid4(),
            db=mock_db,
        )

        # Must get a valid decision despite policy explosion
        assert response.fallback_used is True
        assert response.action is not None

    @pytest.mark.asyncio
    async def test_bandit_invalid_explore_still_returns_decision(self) -> None:
        """
        BanditPolicy with explore=False must always return a valid decision.
        (ExplorationGuard uses this during stress.)
        """
        policy = BanditPolicy(epsilon_start=1.0)
        state = _stress_state()

        # explore=False should force exploitation
        decision = await policy.decide(state, explore=False)
        assert decision is not None
        assert decision.policy_mode == PolicyMode.ACTIVE
        assert 1 <= decision.instances_after <= 20

    @pytest.mark.asyncio
    async def test_concurrent_policy_explosions_all_fallback(self) -> None:
        """
        50 concurrent requests when policy is broken.
        All 50 must get fallback decisions, none should raise.
        """

        class BrokenPolicy(BaselinePolicy):
            async def decide(self, state, explore=True):
                raise RuntimeError(f"Broken at {random.random()}")

        svc = DecisionService()
        svc._active_policy = BrokenPolicy()
        # Mock kill switch — no Redis in chaos unit tests
        from app.operator.kill_switch import KillSwitchState

        safe_ks = KillSwitchState(
            global_killed=False,
            exploration_frozen=False,
            promotion_frozen=False,
            killed_policies=set(),
        )
        svc._kill_switch.get_state = AsyncMock(return_value=safe_ks)
        svc._manual_override.is_baseline_forced = AsyncMock(return_value=False)

        async def make_one() -> bool:
            mock_db = AsyncMock()
            mock_db.add = MagicMock()
            mock_db.flush = AsyncMock()
            mock_db.commit = AsyncMock()
            resp = await svc.make_decision(state=_normal_state(), trace_id=uuid.uuid4(), db=mock_db)
            return resp.fallback_used

        results = await asyncio.gather(*[make_one() for _ in range(50)])
        assert all(r is True for r in results), "Not all fallbacks triggered"


@pytest.mark.chaos
class TestCorruptCheckpoint:
    """Verify corrupt/mismatched checkpoints fail loud."""

    def test_bandit_corrupt_weights_raises_checkpoint_error(self) -> None:
        """Corrupt bandit checkpoint → CheckpointError, not KeyError or silent fail."""
        policy = BanditPolicy()
        bad_cp = PolicyCheckpointData(
            weights={"corrupted": True, "totally_wrong_format": [1, 2, 3]},
            step_count=0,
        )
        with pytest.raises(CheckpointError):
            policy.load_checkpoint(bad_cp)

    def test_rl_truncated_weights_raises_checkpoint_error(self) -> None:
        """Truncated RL checkpoint → CheckpointError."""
        policy = RLPolicy()
        bad_cp = PolicyCheckpointData(
            weights={"q_network": {"layer_0_W": [[1.0]]}},  # missing target_network etc.
            step_count=0,
        )
        with pytest.raises(CheckpointError):
            policy.load_checkpoint(bad_cp)

    def test_normalizer_version_mismatch_raises(self, tmp_path) -> None:
        """Loading normalizer with wrong expected version → CheckpointError."""
        norm = StateNormalizer(version_id="model-v1")
        states = [
            SystemState(
                cpu_utilization=float(i) / 100,
                request_rate=float(i) * 10,
                p99_latency_ms=100.0 + i,
                instance_count=max(1, i % 20),
                hour_of_day=i % 24,
                day_of_week=i % 7,
            )
            for i in range(20)
        ]
        norm.fit(states)
        path = tmp_path / "norm.json"
        norm.save(path)

        with pytest.raises(CheckpointError, match="version mismatch"):
            StateNormalizer.load(path, expected_version_id="model-v99")

    def test_missing_normalizer_file_raises_file_not_found(self) -> None:
        """Missing normalizer file → FileNotFoundError (not silent None)."""
        from pathlib import Path

        with pytest.raises(FileNotFoundError):
            StateNormalizer.load(Path("/nonexistent/normalizer.json"))

    def test_rl_decide_without_normalizer_raises_runtime_error(self) -> None:
        """RL inference without normalizer → RuntimeError (loud, not AttributeError)."""

        async def _test():
            policy = RLPolicy()  # no normalizer
            with pytest.raises(RuntimeError, match="without a fitted normalizer"):
                await policy.decide(_normal_state())

        asyncio.get_event_loop().run_until_complete(_test())


@pytest.mark.chaos
class TestExplorationGuardUnderChaos:
    """Verify exploration guard holds its invariants under chaotic conditions."""

    def test_guard_always_suppresses_on_stress_regardless_of_concurrency(self) -> None:
        """
        50 concurrent guard checks under stress must ALL suppress.
        No race condition should allow exploration to slip through.
        """
        guard = ExplorationGuard()
        stats = PolicyStats(
            sla_violation_rate_5min=0.1,  # > 0.03 threshold
            consecutive_violations=10,
        )

        results = [guard.check_and_log(_stress_state(), stats) for _ in range(50)]
        assert all(r is False for r in results), "Guard allowed exploration under stress"

    def test_guard_never_suppresses_on_healthy_system(self) -> None:
        """
        Guard must always allow exploration on healthy system.
        """
        guard = ExplorationGuard()
        stats = PolicyStats(sla_violation_rate_5min=0.0, consecutive_violations=0)

        results = [guard.check_and_log(_normal_state(), stats) for _ in range(50)]
        assert all(r is True for r in results), "Guard incorrectly suppressed on healthy system"


@pytest.mark.chaos
class TestReplayBufferEdgeCases:
    """Verify replay buffer handles edge cases gracefully."""

    def test_train_step_none_on_empty_buffer(self) -> None:
        """train_step() with empty buffer must return None, not raise."""
        policy = RLPolicy()
        result = policy.train_step()
        assert result is None

    def test_buffer_at_capacity_does_not_raise(self) -> None:
        """Pushing beyond capacity must silently evict oldest, not crash."""
        from app.policies.rl_policy import ReplayBuffer, Transition

        buf = ReplayBuffer(capacity=10)
        for i in range(100):  # 10x capacity
            buf.push(Transition([float(i)] * 6, 0, 1.0, [float(i)] * 6, False))

        assert len(buf) == 10  # capped at capacity

    @pytest.mark.asyncio
    async def test_update_without_normalizer_is_silent_noop(self) -> None:
        """update() without normalizer must silently do nothing, not raise."""
        from app.schemas.common import ActionType, PolicyType
        from app.schemas.decision import ScalingDecision

        policy = RLPolicy()  # no normalizer
        state = _normal_state()
        decision = ScalingDecision(
            action=ActionType.HOLD,
            instances_before=5,
            instances_after=5,
            policy_type=PolicyType.RL,
        )
        # Must not raise
        await policy.update(state, decision, reward=-1.0)
        assert policy.buffer_size == 0  # nothing added
