"""
BaselinePolicy tests — Phase 2 gate.

Verifies:
- SCALE_UP_3 on critical CPU or latency
- SCALE_UP_1 on high CPU or latency
- SCALE_DOWN_1 on low CPU + latency + room to scale down
- HOLD on normal conditions
- Action clipping at min/max boundaries
- Checkpoint round-trip (even though baseline has no trainable state)
- Update is a no-op (does not crash)
- Deterministic: same input → same output
"""

from __future__ import annotations

import pytest

from app.policies.baseline_policy import BaselinePolicy
from app.schemas.common import ActionType, PolicyType
from app.schemas.state import SystemState


def _state(
    cpu: float = 0.5,
    rps: float = 1000.0,
    latency: float = 200.0,
    instances: int = 5,
) -> SystemState:
    """Helper to create test states concisely."""
    return SystemState(
        cpu_utilization=cpu,
        request_rate=rps,
        p99_latency_ms=latency,
        instance_count=instances,
    )


@pytest.mark.unit
class TestBaselinePolicy:
    """Verify all 4 decision conditions + boundary behavior."""

    @pytest.fixture
    def policy(self) -> BaselinePolicy:
        return BaselinePolicy()

    @pytest.mark.asyncio
    async def test_critical_cpu_scales_up_3(self, policy: BaselinePolicy) -> None:
        """cpu > 0.90 → SCALE_UP_3."""
        decision = await policy.decide(_state(cpu=0.95))
        assert decision.action == ActionType.SCALE_UP_3
        assert decision.instances_after == decision.instances_before + 3

    @pytest.mark.asyncio
    async def test_critical_latency_scales_up_3(self, policy: BaselinePolicy) -> None:
        """p99 > 800ms → SCALE_UP_3."""
        decision = await policy.decide(_state(latency=900.0))
        assert decision.action == ActionType.SCALE_UP_3

    @pytest.mark.asyncio
    async def test_high_cpu_scales_up_1(self, policy: BaselinePolicy) -> None:
        """cpu > 0.80 → SCALE_UP_1."""
        decision = await policy.decide(_state(cpu=0.85))
        assert decision.action == ActionType.SCALE_UP_1
        assert decision.instances_after == decision.instances_before + 1

    @pytest.mark.asyncio
    async def test_high_latency_scales_up_1(self, policy: BaselinePolicy) -> None:
        """p99 > 500ms → SCALE_UP_1."""
        decision = await policy.decide(_state(latency=600.0))
        assert decision.action == ActionType.SCALE_UP_1

    @pytest.mark.asyncio
    async def test_low_cpu_low_latency_scales_down(self, policy: BaselinePolicy) -> None:
        """cpu < 0.30 AND latency < 100ms AND instances > min → SCALE_DOWN_1."""
        decision = await policy.decide(_state(cpu=0.1, latency=50.0, instances=5))
        assert decision.action == ActionType.SCALE_DOWN_1
        assert decision.instances_after == decision.instances_before - 1

    @pytest.mark.asyncio
    async def test_low_cpu_but_at_min_holds(self, policy: BaselinePolicy) -> None:
        """At min_instances, cannot scale down even if metrics are low."""
        decision = await policy.decide(_state(cpu=0.1, latency=50.0, instances=1))
        assert decision.action == ActionType.HOLD

    @pytest.mark.asyncio
    async def test_normal_conditions_hold(self, policy: BaselinePolicy) -> None:
        """Normal CPU (0.50) and latency (200ms) → HOLD."""
        decision = await policy.decide(_state(cpu=0.50, latency=200.0))
        assert decision.action == ActionType.HOLD
        assert decision.instances_after == decision.instances_before

    @pytest.mark.asyncio
    async def test_clipping_at_max(self, policy: BaselinePolicy) -> None:
        """instances_after never exceeds max_instances."""
        decision = await policy.decide(_state(cpu=0.95, instances=19))
        assert decision.instances_after <= 20  # max_instances default

    @pytest.mark.asyncio
    async def test_clipping_at_min(self, policy: BaselinePolicy) -> None:
        """instances_after never goes below min_instances."""
        decision = await policy.decide(_state(cpu=0.1, latency=50.0, instances=1))
        assert decision.instances_after >= 1

    @pytest.mark.asyncio
    async def test_policy_type_is_baseline(self, policy: BaselinePolicy) -> None:
        """Policy must identify itself as BASELINE."""
        assert policy.policy_type == PolicyType.BASELINE

    @pytest.mark.asyncio
    async def test_confidence_is_one(self, policy: BaselinePolicy) -> None:
        """Baseline is deterministic, confidence must be 1.0."""
        decision = await policy.decide(_state())
        assert decision.confidence == 1.0

    @pytest.mark.asyncio
    async def test_deterministic(self, policy: BaselinePolicy) -> None:
        """Same state must produce same action every time."""
        state = _state(cpu=0.85, latency=300.0, instances=5)
        d1 = await policy.decide(state)
        d2 = await policy.decide(state)
        assert d1.action == d2.action
        assert d1.instances_after == d2.instances_after

    @pytest.mark.asyncio
    async def test_checkpoint_roundtrip(self, policy: BaselinePolicy) -> None:
        """Checkpoint save/load should not crash (even if no-op)."""
        cp = policy.get_checkpoint()
        assert cp.weights is not None
        policy.load_checkpoint(cp)  # should not raise

    @pytest.mark.asyncio
    async def test_update_is_noop(self, policy: BaselinePolicy) -> None:
        """update() should not crash and should not change behavior."""
        state = _state(cpu=0.85)
        d1 = await policy.decide(state)
        await policy.update(state, d1, reward=-1.0)
        d2 = await policy.decide(state)
        assert d1.action == d2.action

    @pytest.mark.asyncio
    async def test_edge_cpu_zero(self, policy: BaselinePolicy) -> None:
        """cpu=0.0 should not crash."""
        decision = await policy.decide(_state(cpu=0.0, latency=50.0, instances=3))
        assert decision.action in (ActionType.SCALE_DOWN_1, ActionType.HOLD)

    @pytest.mark.asyncio
    async def test_edge_cpu_one(self, policy: BaselinePolicy) -> None:
        """cpu=1.0 should trigger scale up."""
        decision = await policy.decide(_state(cpu=1.0))
        assert decision.action in (ActionType.SCALE_UP_1, ActionType.SCALE_UP_3)
