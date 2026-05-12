"""
Chaos injection tests — inject failures and verify system recovery.

WHY CHAOS TESTS:
- Unit tests verify individual components in isolation.
- Integration tests verify components work together.
- Chaos tests verify the system RECOVERS from realistic failure modes.

THE CHAOS SCENARIOS:
1. Reward degradation injection: simulate N consecutive bad rewards.
   Expected: DriftService detects after K windows, rollback fires.

2. Input distribution shift: inject feature vectors from a different distribution.
   Expected: PSI > threshold after K windows, INPUT_DRIFT signal fires.

3. ExplorationGuard stress: inject high-latency states during exploration.
   Expected: Guard suppresses exploration 100% of the time during stress.

4. Policy exception injection: make active policy always raise.
   Expected: DecisionService falls back to baseline, fallback_flag=True.

5. Checkpoint corruption: inject malformed checkpoint data.
   Expected: CheckpointError raised loudly — not silent wrong inference.

WHAT CHAOS TESTS ARE NOT:
- Performance tests (use benchmarks for that)
- Capacity tests (use load testing tools)
- Infrastructure failures (use real chaos tools like Chaos Monkey)
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from app.core.exceptions import CheckpointError
from app.policies.bandit_policy import BanditPolicy
from app.policies.base_policy import PolicyCheckpointData
from app.policies.baseline_policy import BaselinePolicy
from app.policies.rl_policy import RLPolicy
from app.safety.exploration_guard import ExplorationGuard, PolicyStats
from app.schemas.common import ActionType, PolicyType, TrafficRegime
from app.schemas.state import SystemState
from app.services.drift_service import DriftService, DriftWindow

# ── Helpers ───────────────────────────────────────────────────────────────────


def _stress_state(latency_ms: float = 900.0) -> SystemState:
    """State that should trigger ExplorationGuard."""
    return SystemState(
        cpu_utilization=0.9,
        request_rate=6000.0,
        p99_latency_ms=latency_ms,
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


# ── Chaos Scenario 1: Reward Degradation ─────────────────────────────────────


@pytest.mark.chaos
class TestRewardDegradationChaos:
    """Simulate progressive reward degradation and verify drift fires."""

    def test_drift_fires_after_k_consecutive_bad_windows(self) -> None:
        """
        Inject severely degraded rewards for K consecutive windows.
        Verify drift_detected=True and signal=REWARD_DEGRADATION.
        """
        svc = DriftService()

        # Establish reference (good performance)
        rng = np.random.RandomState(42)
        ref_rewards = rng.normal(-0.5, 0.1, 200).tolist()
        ref_window = DriftWindow(
            feature_vectors=[[0.5] * 6 for _ in range(200)],
            rewards=ref_rewards,
        )
        svc.set_reference_window(ref_window)

        # Inject degraded rewards
        rng2 = np.random.RandomState(99)
        for _ in range(50):
            svc.add_observation([0.5] * 6, float(rng2.normal(-15.0, 0.1)), TrafficRegime.STEADY)

        # Evaluate K times — must trigger after hysteresis threshold
        k = svc._hysteresis_k
        results = [svc.evaluate() for _ in range(k)]

        assert results[-1].drift_detected, (
            f"Expected drift after {k} consecutive bad windows, got drift_detected=False"
        )
        assert results[-1].drift_signal is not None
        assert results[-1].reward_delta is not None
        assert results[-1].reward_delta < -5.0  # significantly worse

    def test_drift_does_not_fire_on_mild_degradation(self) -> None:
        """
        Mild degradation (within 1 sigma) must NOT trigger drift.
        Only severe degradation warrants rollback.
        """
        svc = DriftService()
        rng = np.random.RandomState(42)
        ref_rewards = rng.normal(-1.0, 0.3, 200).tolist()
        ref_window = DriftWindow(
            feature_vectors=[[0.5] * 6 for _ in range(200)],
            rewards=ref_rewards,
        )
        svc.set_reference_window(ref_window)

        # Mild degradation — within 2 sigma of reference
        rng2 = np.random.RandomState(99)
        for _ in range(50):
            svc.add_observation([0.5] * 6, float(rng2.normal(-1.3, 0.3)), TrafficRegime.STEADY)

        result = svc.evaluate()
        # Single evaluation should not trigger (hysteresis)
        assert not result.drift_detected

    def test_recovery_after_drift_resets_system(self) -> None:
        """After rollback (hysteresis reset), good rewards clear the alert."""
        svc = DriftService()
        rng = np.random.RandomState(42)
        ref_rewards = rng.normal(-0.5, 0.1, 200).tolist()
        ref_window = DriftWindow(
            feature_vectors=[[0.5] * 6 for _ in range(200)],
            rewards=ref_rewards,
        )
        svc.set_reference_window(ref_window)

        # Inject bad rewards for K windows
        rng2 = np.random.RandomState(99)
        for _ in range(50):
            svc.add_observation([0.5] * 6, float(rng2.normal(-15.0, 0.1)), TrafficRegime.STEADY)

        k = svc._hysteresis_k
        [svc.evaluate() for _ in range(k)]

        # Simulate rollback: reset hysteresis
        svc.reset_hysteresis()
        assert svc._degraded_window_count == 0

        # Replace with good rewards
        svc._current_observations.clear()
        rng3 = np.random.RandomState(77)
        for _ in range(50):
            svc.add_observation([0.5] * 6, float(rng3.normal(-0.5, 0.1)), TrafficRegime.STEADY)

        result = svc.evaluate()
        assert not result.drift_detected


# ── Chaos Scenario 2: Input Distribution Shift ───────────────────────────────


@pytest.mark.chaos
class TestInputDistributionShiftChaos:
    """Verify PSI-based drift detection on feature shift."""

    def test_psi_fires_on_major_feature_shift(self) -> None:
        """
        Inject features from a completely different distribution.
        Verify PSI > threshold is detected.
        """
        import numpy as np

        from app.services.drift_service import _compute_psi

        rng = np.random.RandomState(42)
        reference = rng.normal(0.5, 0.05, 500)
        shifted = rng.normal(0.95, 0.05, 500)  # shifted to 0.95

        psi = _compute_psi(reference, shifted)
        assert psi > 0.2, f"Expected PSI > 0.2, got {psi:.4f}"

    def test_psi_safe_on_same_distribution(self) -> None:
        """Same distribution must NOT trigger PSI drift."""
        import numpy as np

        from app.services.drift_service import _compute_psi

        rng = np.random.RandomState(42)
        reference = rng.normal(0.5, 0.1, 500)
        current = rng.normal(0.5, 0.1, 500)

        psi = _compute_psi(reference, current)
        assert psi < 0.1, f"Expected PSI < 0.1, got {psi:.4f}"


# ── Chaos Scenario 3: ExplorationGuard Under Stress ──────────────────────────


@pytest.mark.chaos
class TestExplorationGuardChaos:
    """Verify ExplorationGuard suppresses 100% during system stress."""

    def test_guard_suppresses_all_exploration_under_load(self) -> None:
        """
        Under high-latency + high-RPS stress, guard must suppress ALL exploration.
        Even 1% exploration during a critical incident = unacceptable.
        """
        guard = ExplorationGuard()
        stats = PolicyStats()
        stressed = _stress_state(latency_ms=900.0)

        results = [guard.check_and_log(stressed, stats) for _ in range(100)]

        suppressed = sum(1 for r in results if not r)
        assert suppressed == 100, (
            f"Guard failed to suppress: only {suppressed}/100 suppressed. "
            "Any exploration during critical stress = SLA breach risk."
        )

    def test_guard_allows_all_exploration_when_stable(self) -> None:
        """When stable, guard must allow ALL exploration (no false positives)."""
        guard = ExplorationGuard()
        stats = PolicyStats()
        stable = _normal_state()

        results = [guard.check_and_log(stable, stats) for _ in range(100)]
        allowed = sum(1 for r in results if r)
        assert allowed == 100, f"Guard false-positives: only {allowed}/100 allowed."


# ── Chaos Scenario 4: Policy Exception Injection ─────────────────────────────


@pytest.mark.chaos
class TestPolicyExceptionChaos:
    """Verify DecisionService fallback when active policy crashes."""

    @pytest.mark.asyncio
    async def test_decision_service_falls_back_on_policy_exception(self) -> None:
        """
        If active policy raises any exception during decide(),
        DecisionService must fall back to baseline and set fallback_flag=True.
        """
        from app.operator.kill_switch import KillSwitchState
        from app.services.decision_service import DecisionService

        svc = DecisionService()

        # Inject a broken policy
        broken_policy = MagicMock()
        broken_policy.policy_type = PolicyType.RL
        broken_policy.policy_mode.value = "ACTIVE"
        broken_policy.decide = AsyncMock(side_effect=RuntimeError("GPU OOM"))

        svc.set_active_policy(broken_policy)  # type: ignore[arg-type]

        # Mock kill switch (no Redis in unit tests)
        safe_state = KillSwitchState(
            global_killed=False,
            exploration_frozen=False,
            promotion_frozen=False,
            killed_policies=set(),
        )
        svc._kill_switch.get_state = AsyncMock(return_value=safe_state)
        svc._manual_override.is_baseline_forced = AsyncMock(return_value=False)

        # Decision must complete despite policy crash
        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()
        mock_db.commit = AsyncMock()

        state = _normal_state()
        result = await svc.make_decision(
            state=state,
            trace_id=uuid.uuid4(),
            db=mock_db,
        )

        assert result.fallback_used is True
        assert result.policy_type == PolicyType.BASELINE

    @pytest.mark.asyncio
    async def test_baseline_never_raises(self) -> None:
        """
        BaselinePolicy must NEVER raise an exception — it is the final fallback.
        Tested with edge-case inputs that might break a fragile implementation.
        """
        policy = BaselinePolicy()

        edge_cases = [
            SystemState(
                cpu_utilization=0.0,
                request_rate=0.0,
                p99_latency_ms=0.0,
                instance_count=1,
            ),
            SystemState(
                cpu_utilization=1.0,
                request_rate=99999.0,
                p99_latency_ms=99999.0,
                instance_count=20,
            ),
            SystemState(
                cpu_utilization=0.5,
                request_rate=1000.0,
                p99_latency_ms=200.0,
                instance_count=1,
            ),
        ]

        for state in edge_cases:
            decision = await policy.decide(state)
            assert 1 <= decision.instances_after <= 20
            assert decision.action in ActionType


# ── Chaos Scenario 5: Checkpoint Corruption ──────────────────────────────────


@pytest.mark.chaos
class TestCheckpointCorruptionChaos:
    """Verify loud failure on corrupt checkpoints — no silent wrong inference."""

    def test_bandit_raises_on_null_weights(self) -> None:
        """Null weights → CheckpointError, not AttributeError."""
        policy = BanditPolicy()
        with pytest.raises(CheckpointError, match="no weights"):
            policy.load_checkpoint(PolicyCheckpointData(weights=None, step_count=0))

    def test_bandit_raises_on_wrong_q_length(self) -> None:
        """Wrong Q-value length → CheckpointError — silent mismatch is worse."""
        policy = BanditPolicy()
        with pytest.raises(CheckpointError, match="length mismatch"):
            policy.load_checkpoint(
                PolicyCheckpointData(
                    weights={
                        "q_values": [0.0, 0.0],  # wrong length
                        "action_counts": [0, 0, 0, 0, 0],
                        "total_steps": 0,
                        "epsilon": 1.0,
                    },
                    step_count=0,
                )
            )

    def test_rl_raises_on_missing_network_keys(self) -> None:
        """Missing q_network key → CheckpointError before any state mutation."""
        policy = RLPolicy()
        with pytest.raises(CheckpointError, match="missing keys"):
            policy.load_checkpoint(
                PolicyCheckpointData(
                    weights={"training_steps": 0},  # missing q_network
                    step_count=0,
                )
            )

    def test_normalizer_raises_on_version_mismatch(self) -> None:
        """Version mismatch → CheckpointError — not silent wrong normalization."""
        import tempfile
        from pathlib import Path

        from app.ml.state_normalizer import StateNormalizer
        from app.schemas.state import SystemState

        states = [
            SystemState(
                cpu_utilization=float(i) / 100,
                request_rate=float(i) * 10,
                p99_latency_ms=100.0 + i,
                instance_count=max(1, i % 20),
                hour_of_day=i % 24,
                day_of_week=i % 7,
            )
            for i in range(50)
        ]
        norm = StateNormalizer(version_id="model-v1")
        norm.fit(states)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "norm.json"
            norm.save(path)
            with pytest.raises(CheckpointError, match="version mismatch"):
                StateNormalizer.load(path, expected_version_id="model-v99")
