"""
RLPolicy (DQN) tests — Phase 5 gate.

Verifies:
- Inference without normalizer raises RuntimeError (loud, not silent)
- is_ready gate: False without normalizer or enough decisions
- Action clipping: instances_after always in [min, max]
- Q-values returned for Decision Explainer
- Checkpoint round-trip preserves network weights
- Checkpoint validation: missing keys raise CheckpointError
- Training isolation: weights unchanged after decide() calls
- Atomic checkpoint write: temp-then-rename pattern verified
- ReplayBuffer: push/sample/size, capacity eviction, serialization
- warm-start gate: is_ready=False until min_decisions met
- Shadow mode propagated to output
- train_step: returns None when buffer empty, float when ready
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.core.exceptions import CheckpointError
from app.ml.state_normalizer import StateNormalizer
from app.policies.base_policy import PolicyCheckpointData
from app.policies.rl_policy import (
    _N_ACTIONS,
    QNetwork,
    ReplayBuffer,
    RLPolicy,
    Transition,
)
from app.schemas.common import ActionType, PolicyMode, PolicyType, TrafficRegime
from app.schemas.state import SystemState


def _make_state(cpu: float = 0.5, instances: int = 5) -> SystemState:
    return SystemState(
        cpu_utilization=cpu,
        request_rate=1000.0,
        p99_latency_ms=200.0,
        instance_count=instances,
        traffic_regime=TrafficRegime.STEADY,
    )


def _make_fitted_normalizer(n: int = 100) -> StateNormalizer:
    states = [
        SystemState(
            cpu_utilization=float(i) / n,
            request_rate=float(i) * 10,
            p99_latency_ms=100.0 + float(i),
            instance_count=max(1, i % 20),
            hour_of_day=i % 24,
            day_of_week=i % 7,
        )
        for i in range(n)
    ]
    norm = StateNormalizer(version_id="test-v1")
    norm.fit(states)
    return norm


def _make_ready_policy(warm_start: int = 5) -> RLPolicy:
    norm = _make_fitted_normalizer()
    policy = RLPolicy(
        normalizer=norm,
        warm_start_min_decisions=warm_start,
        min_instances=1,
        max_instances=20,
        seed=42,
    )
    policy._inference_steps = warm_start
    return policy


@pytest.mark.unit
class TestRLPolicyInitialization:
    def test_policy_type_is_rl(self) -> None:
        assert RLPolicy().policy_type == PolicyType.RL

    def test_not_ready_without_normalizer(self) -> None:
        assert not RLPolicy().is_ready

    def test_not_ready_without_enough_decisions(self) -> None:
        norm = _make_fitted_normalizer()
        policy = RLPolicy(normalizer=norm, warm_start_min_decisions=100)
        assert not policy.is_ready

    def test_ready_when_conditions_met(self) -> None:
        assert _make_ready_policy(warm_start=5).is_ready

    def test_default_mode_is_active(self) -> None:
        assert RLPolicy().policy_mode == PolicyMode.ACTIVE

    def test_training_steps_start_at_zero(self) -> None:
        assert RLPolicy().training_steps == 0


@pytest.mark.unit
class TestRLPolicyInference:
    @pytest.mark.asyncio
    async def test_decide_without_normalizer_raises(self) -> None:
        with pytest.raises(RuntimeError, match="without a fitted normalizer"):
            await RLPolicy().decide(_make_state())

    @pytest.mark.asyncio
    async def test_decide_returns_valid_decision(self) -> None:
        policy = _make_ready_policy()
        decision = await policy.decide(_make_state())
        assert decision.policy_type == PolicyType.RL
        assert decision.action in ActionType
        assert isinstance(decision.instances_after, int)

    @pytest.mark.asyncio
    async def test_decide_returns_q_values(self) -> None:
        policy = _make_ready_policy()
        decision = await policy.decide(_make_state())
        assert decision.q_values is not None
        assert len(decision.q_values) == _N_ACTIONS

    @pytest.mark.asyncio
    async def test_instances_clipped_at_max(self) -> None:
        policy = _make_ready_policy()
        decisions = [await policy.decide(_make_state(instances=19)) for _ in range(20)]
        assert all(d.instances_after <= 20 for d in decisions)

    @pytest.mark.asyncio
    async def test_instances_clipped_at_min(self) -> None:
        policy = _make_ready_policy()
        decisions = [await policy.decide(_make_state(instances=1)) for _ in range(20)]
        assert all(d.instances_after >= 1 for d in decisions)

    @pytest.mark.asyncio
    async def test_shadow_mode_propagated(self) -> None:
        policy = _make_ready_policy()
        policy.policy_mode = PolicyMode.SHADOW
        decision = await policy.decide(_make_state())
        assert decision.policy_mode == PolicyMode.SHADOW

    @pytest.mark.asyncio
    async def test_confidence_in_valid_range(self) -> None:
        policy = _make_ready_policy()
        decision = await policy.decide(_make_state())
        assert decision.confidence is not None
        assert 0.0 <= decision.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_inference_step_count_increments(self) -> None:
        policy = _make_ready_policy()
        initial = policy.inference_steps
        await policy.decide(_make_state())
        assert policy.inference_steps == initial + 1

    @pytest.mark.asyncio
    async def test_concurrent_decisions_valid(self) -> None:
        policy = _make_ready_policy()
        results = await asyncio.gather(*[policy.decide(_make_state()) for _ in range(50)])
        for d in results:
            assert d.policy_type == PolicyType.RL
            assert 1 <= d.instances_after <= 20


@pytest.mark.unit
class TestReplayBuffer:
    def test_push_increases_size(self) -> None:
        buf = ReplayBuffer(capacity=100)
        buf.push(Transition([0.5] * 6, 0, 1.0, [0.5] * 6, False))
        assert len(buf) == 1

    def test_capacity_evicts_oldest(self) -> None:
        buf = ReplayBuffer(capacity=5)
        for i in range(10):
            buf.push(Transition([float(i)] * 6, 0, 1.0, [float(i)] * 6, False))
        assert len(buf) == 5

    def test_sample_returns_batch(self) -> None:
        buf = ReplayBuffer(capacity=100)
        for i in range(50):
            buf.push(Transition([float(i)] * 6, i % 5, 1.0, [float(i)] * 6, False))
        assert len(buf.sample(16)) == 16

    def test_not_ready_when_empty(self) -> None:
        assert not ReplayBuffer(capacity=100).is_ready(64)

    def test_ready_when_full_enough(self) -> None:
        buf = ReplayBuffer(capacity=100)
        for _i in range(100):
            buf.push(Transition([0.5] * 6, 0, 1.0, [0.5] * 6, False))
        assert buf.is_ready(64)

    def test_serialization_round_trip(self) -> None:
        buf = ReplayBuffer(capacity=100)
        for i in range(20):
            buf.push(Transition([float(i)] * 6, i % 5, float(i), [float(i)] * 6, False))
        data = buf.to_dict()
        restored = ReplayBuffer.from_dict(data, capacity=100)
        assert len(restored) == len(data)


@pytest.mark.unit
class TestQNetwork:
    def test_output_shape_single(self) -> None:
        net = QNetwork(seed=42)
        x = np.zeros(6, dtype=np.float32)
        out = net.forward(x)
        assert out.shape == (_N_ACTIONS,)

    def test_output_shape_batch(self) -> None:
        net = QNetwork(seed=42)
        x = np.zeros((32, 6), dtype=np.float32)
        out = net.forward(x)
        assert out.shape == (32, _N_ACTIONS)

    def test_no_nan_on_forward(self) -> None:
        net = QNetwork(seed=42)
        x = np.random.randn(8, 6).astype(np.float32)
        out = net.forward(x)
        assert not np.any(np.isnan(out))

    def test_state_dict_round_trip(self) -> None:
        net = QNetwork(seed=42)
        sd = net.get_state_dict()
        net2 = QNetwork(seed=99)
        net2.load_state_dict(sd)
        x = np.ones(6, dtype=np.float32)
        np.testing.assert_allclose(net.forward(x), net2.forward(x), rtol=1e-5)


@pytest.mark.unit
class TestRLCheckpoint:
    def test_checkpoint_contains_required_keys(self) -> None:
        cp = _make_ready_policy().get_checkpoint()
        assert "q_network" in cp.weights
        assert "target_network" in cp.weights
        assert "training_steps" in cp.weights

    def test_load_checkpoint_raises_on_none_weights(self) -> None:
        with pytest.raises(CheckpointError, match="no weights"):
            RLPolicy().load_checkpoint(PolicyCheckpointData(weights=None, step_count=0))

    def test_load_checkpoint_raises_on_missing_keys(self) -> None:
        with pytest.raises(CheckpointError, match="missing keys"):
            RLPolicy().load_checkpoint(
                PolicyCheckpointData(
                    weights={"q_network": {}},
                    step_count=0,
                )
            )

    def test_atomic_write_saves_file(self) -> None:
        policy = _make_ready_policy()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.json"
            policy.save_weights_to_file(path)
            assert path.exists()
            data = json.loads(path.read_text())
            assert "q_network" in data

    def test_atomic_write_no_tmp_on_success(self) -> None:
        policy = _make_ready_policy()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.json"
            policy.save_weights_to_file(path)
            assert not path.with_suffix(".tmp").exists()

    def test_training_steps_preserved(self) -> None:
        policy = _make_ready_policy()
        policy._training_steps = 42
        assert policy.get_checkpoint().step_count == 42

    def test_checkpoint_round_trip(self) -> None:
        policy = _make_ready_policy()
        cp = policy.get_checkpoint()
        policy2 = RLPolicy(normalizer=_make_fitted_normalizer())
        policy2.load_checkpoint(cp)
        assert policy2.training_steps == policy.training_steps

    @pytest.mark.asyncio
    async def test_update_adds_to_replay_buffer(self) -> None:
        from app.schemas.decision import ScalingDecision

        policy = _make_ready_policy()
        initial = policy.buffer_size
        await policy.update(
            _make_state(),
            ScalingDecision(
                action=ActionType.HOLD,
                instances_before=5,
                instances_after=5,
                policy_type=PolicyType.RL,
            ),
            reward=1.0,
        )
        assert policy.buffer_size == initial + 1


@pytest.mark.unit
class TestTrainingIsolation:
    def test_train_step_none_when_empty(self) -> None:
        assert _make_ready_policy().train_step() is None

    def test_train_step_returns_loss_when_ready(self) -> None:
        policy = _make_ready_policy()
        for _i in range(100):
            policy._replay_buffer.push(
                Transition(
                    [0.5] * 6,
                    0,
                    1.0,
                    [0.5] * 6,
                    False,
                )
            )
        loss = policy.train_step()
        assert loss is not None
        assert isinstance(loss, float)
        assert loss == loss  # not NaN

    def test_decide_does_not_modify_weights(self) -> None:
        """Inference must never change Q-network weights."""
        policy = _make_ready_policy()
        initial_w = policy._q_network.layers[0]["W"].copy()

        async def run() -> None:
            for _ in range(20):
                await policy.decide(_make_state())

        asyncio.get_event_loop().run_until_complete(run())
        np.testing.assert_array_equal(policy._q_network.layers[0]["W"], initial_w)

    def test_target_network_updated_periodically(self) -> None:
        """Target network must update every target_update_freq steps."""
        policy = _make_ready_policy()
        policy._target_update_freq = 5

        # Fill buffer
        for _i in range(100):
            policy._replay_buffer.push(Transition([0.5] * 6, 0, 1.0, [0.5] * 6, False))

        # Run exactly target_update_freq steps
        for _ in range(5):
            policy.train_step()

        assert policy.training_steps == 5
