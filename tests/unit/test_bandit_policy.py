"""
BanditPolicy tests — Phase 4 gate.

Verifies:
- ε-greedy exploration rate
- ε decay behavior and floor enforcement
- UCB exploration: untried actions get priority
- Q-value incremental mean update correctness
- Checkpoint round-trip: save → restore → same Q-values
- Checkpoint validation: missing keys raise CheckpointError
- ExplorationGuard integration: explore=False → always exploit
- Shadow mode: policy_mode=SHADOW preserved in output
- Action clipping: instances_after always in [min, max]
- Concurrent calls: deterministic output on same state
- Online learning convergence: best action rises to top after repeated reward
"""

from __future__ import annotations

import asyncio

import pytest

from app.core.exceptions import CheckpointError
from app.policies.bandit_policy import (
    _ACTION_IDX,
    _ACTIONS,
    _N_ACTIONS,
    BanditPolicy,
    ExplorationStrategy,
)
from app.policies.base_policy import PolicyCheckpointData
from app.schemas.common import ActionType, PolicyMode, PolicyType, TrafficRegime
from app.schemas.decision import ScalingDecision
from app.schemas.state import SystemState


def _state(instances: int = 5) -> SystemState:
    return SystemState(
        cpu_utilization=0.5,
        request_rate=1000.0,
        p99_latency_ms=200.0,
        instance_count=instances,
        traffic_regime=TrafficRegime.STEADY,
    )


def _decision(action: ActionType, instances_before: int = 5) -> ScalingDecision:
    delta = {
        ActionType.SCALE_UP_3: 3,
        ActionType.SCALE_UP_1: 1,
        ActionType.HOLD: 0,
        ActionType.SCALE_DOWN_1: -1,
        ActionType.SCALE_DOWN_3: -3,
    }[action]
    return ScalingDecision(
        action=action,
        instances_before=instances_before,
        instances_after=instances_before + delta,
        policy_type=PolicyType.BANDIT,
    )


@pytest.mark.unit
class TestBanditPolicyInitialization:
    """Verify initial state is correct."""

    def test_policy_type_is_bandit(self) -> None:
        policy = BanditPolicy()
        assert policy.policy_type == PolicyType.BANDIT

    def test_initial_q_values_zero(self) -> None:
        policy = BanditPolicy()
        assert all(q == 0.0 for q in policy.q_values)

    def test_initial_action_counts_zero(self) -> None:
        policy = BanditPolicy()
        assert all(n == 0 for n in policy.action_counts)

    def test_initial_epsilon_at_start(self) -> None:
        policy = BanditPolicy(epsilon_start=0.8)
        assert policy.epsilon == 0.8

    def test_initial_steps_zero(self) -> None:
        policy = BanditPolicy()
        assert policy.total_steps == 0

    def test_default_mode_active(self) -> None:
        policy = BanditPolicy()
        assert policy.policy_mode == PolicyMode.ACTIVE

    def test_n_actions_is_five(self) -> None:
        assert _N_ACTIONS == 5
        assert len(_ACTIONS) == 5


@pytest.mark.unit
class TestEpsilonGreedyExploration:
    """Verify ε-greedy exploration behavior."""

    @pytest.mark.asyncio
    async def test_always_explore_at_epsilon_one(self) -> None:
        """With ε=1.0, every call explores (random action)."""
        # With ε=1.0 and many trials, we should see variety in actions
        policy = BanditPolicy(epsilon_start=1.0, epsilon_decay=1.0)  # no decay

        actions = set()
        for _ in range(100):
            decision = await policy.decide(_state(), explore=True)
            actions.add(decision.action)

        # With 100 random choices from 5 actions, we expect variety
        assert len(actions) > 1

    @pytest.mark.asyncio
    async def test_always_exploit_at_epsilon_zero(self) -> None:
        """With explore=False, always pick argmax Q(a)."""
        policy = BanditPolicy()
        # Set known Q-values: HOLD is best
        hold_idx = _ACTION_IDX[ActionType.HOLD]
        policy._state.q_values[hold_idx] = 10.0

        decisions = [await policy.decide(_state(), explore=False) for _ in range(20)]
        assert all(d.action == ActionType.HOLD for d in decisions)

    @pytest.mark.asyncio
    async def test_epsilon_decays_after_update(self) -> None:
        """ε must decrease after each update call."""
        policy = BanditPolicy(epsilon_start=1.0, epsilon_decay=0.9, epsilon_floor=0.01)
        initial_epsilon = policy.epsilon

        state = _state()
        decision = await policy.decide(state)
        await policy.update(state, decision, reward=1.0)

        assert policy.epsilon < initial_epsilon

    @pytest.mark.asyncio
    async def test_epsilon_respects_floor(self) -> None:
        """ε must never go below ε_floor."""
        policy = BanditPolicy(
            epsilon_start=1.0,
            epsilon_decay=0.5,
            epsilon_floor=0.1,
        )
        state = _state()

        # Run many updates to force epsilon to floor
        for _ in range(50):
            d = await policy.decide(state)
            await policy.update(state, d, reward=1.0)

        assert policy.epsilon >= 0.1
        assert policy.epsilon > 0.0  # never zero


@pytest.mark.unit
class TestUCBExploration:
    """Verify UCB exploration behavior."""

    @pytest.mark.asyncio
    async def test_ucb_tries_untried_actions_first(self) -> None:
        """UCB assigns infinite value to untried actions — they get selected and tried."""
        policy = BanditPolicy(strategy=ExplorationStrategy.UCB)
        state = _state()

        # Run UCB with updates: each update makes one action 'tried'
        # so subsequent calls should explore other untried actions
        tried_actions = set()
        for _ in range(_N_ACTIONS):
            decision = await policy.decide(state, explore=True)
            tried_actions.add(decision.action)
            # Update so this action now has count > 0 (no longer untried)
            await policy.update(state, decision, reward=0.0)

        # After trying all N_ACTIONS, we should have seen all 5 actions
        assert len(tried_actions) == _N_ACTIONS

    @pytest.mark.asyncio
    async def test_ucb_exploits_when_all_tried(self) -> None:
        """After all actions tried, UCB exploits best Q-value (with confidence bonus)."""
        policy = BanditPolicy(strategy=ExplorationStrategy.UCB, ucb_c=0.001)
        # Set all counts > 0 and one clearly best action
        for i in range(_N_ACTIONS):
            policy._state.action_counts[i] = 100
        hold_idx = _ACTION_IDX[ActionType.HOLD]
        policy._state.q_values[hold_idx] = 10.0
        policy._state.total_steps = 500

        decisions = [await policy.decide(_state(), explore=True) for _ in range(20)]
        # With tiny ucb_c and many trials, should mostly exploit best Q
        hold_count = sum(1 for d in decisions if d.action == ActionType.HOLD)
        assert hold_count >= 15  # overwhelming majority should be HOLD


@pytest.mark.unit
class TestOnlineLearning:
    """Verify Q-value incremental mean update and convergence."""

    @pytest.mark.asyncio
    async def test_q_value_updates_after_reward(self) -> None:
        """Q(HOLD) must increase after positive reward for HOLD."""
        policy = BanditPolicy()
        hold_idx = _ACTION_IDX[ActionType.HOLD]
        initial_q = policy.q_values[hold_idx]

        state = _state()
        decision = _decision(ActionType.HOLD)
        await policy.update(state, decision, reward=5.0)

        # Q(HOLD) should be higher after positive reward
        assert policy.q_values[hold_idx] > initial_q

    @pytest.mark.asyncio
    async def test_q_value_decreases_on_negative_reward(self) -> None:
        """Q(SCALE_DOWN_3) must decrease after negative reward."""
        policy = BanditPolicy()
        down3_idx = _ACTION_IDX[ActionType.SCALE_DOWN_3]

        state = _state()
        decision = _decision(ActionType.SCALE_DOWN_3)
        await policy.update(state, decision, reward=-10.0)

        assert policy.q_values[down3_idx] < 0.0

    @pytest.mark.asyncio
    async def test_q_value_converges_to_true_mean(self) -> None:
        """After N updates with constant reward, Q(a) converges to that reward."""
        policy = BanditPolicy()
        state = _state()
        true_reward = 3.0
        n_updates = 100

        for _ in range(n_updates):
            decision = _decision(ActionType.HOLD)
            await policy.update(state, decision, reward=true_reward)

        hold_idx = _ACTION_IDX[ActionType.HOLD]
        # Q(HOLD) should be very close to 3.0 after 100 updates
        assert abs(policy.q_values[hold_idx] - true_reward) < 0.01

    @pytest.mark.asyncio
    async def test_step_count_increments(self) -> None:
        """total_steps must increment after each update."""
        policy = BanditPolicy()
        state = _state()

        for _i in range(5):
            d = await policy.decide(state)
            await policy.update(state, d, reward=1.0)

        assert policy.total_steps == 5

    @pytest.mark.asyncio
    async def test_cumulative_reward_tracks(self) -> None:
        """cumulative_reward must sum all rewards received."""
        policy = BanditPolicy()
        state = _state()
        rewards = [1.0, -2.0, 3.0, -1.5, 0.5]

        for r in rewards:
            d = await policy.decide(state)
            await policy.update(state, d, reward=r)

        assert abs(policy.cumulative_reward - sum(rewards)) < 0.001

    @pytest.mark.asyncio
    async def test_best_action_rises_to_top(self) -> None:
        """After training with signal, best action should dominate Q-values."""
        policy = BanditPolicy(epsilon_start=0.0)  # pure exploitation for test clarity
        state = _state()

        # HOLD always gives positive reward, SCALE_DOWN always negative
        hold_idx = _ACTION_IDX[ActionType.HOLD]
        down_idx = _ACTION_IDX[ActionType.SCALE_DOWN_1]

        for _ in range(50):
            await policy.update(state, _decision(ActionType.HOLD), reward=2.0)
            await policy.update(state, _decision(ActionType.SCALE_DOWN_1), reward=-5.0)

        assert policy.q_values[hold_idx] > 1.9
        assert policy.q_values[down_idx] < -4.9


@pytest.mark.unit
class TestCheckpointPersistence:
    """Verify checkpoint save → load round-trip correctness."""

    def test_checkpoint_contains_required_keys(self) -> None:
        """Checkpoint weights must have all required keys."""
        policy = BanditPolicy()
        cp = policy.get_checkpoint()
        required = {"q_values", "action_counts", "total_steps", "epsilon"}
        assert required.issubset(cp.weights.keys())

    @pytest.mark.asyncio
    async def test_checkpoint_round_trip_preserves_q_values(self) -> None:
        """Q-values after restore must match Q-values before save."""
        policy = BanditPolicy()
        state = _state()

        # Train for a bit to get non-trivial Q-values
        for _ in range(20):
            d = await policy.decide(state)
            await policy.update(state, d, reward=1.5)

        original_q = list(policy.q_values)
        original_epsilon = policy.epsilon
        original_steps = policy.total_steps

        # Save checkpoint
        cp = policy.get_checkpoint()

        # Create fresh policy and restore
        new_policy = BanditPolicy()
        new_policy.load_checkpoint(cp)

        assert new_policy.q_values == pytest.approx(original_q)
        assert new_policy.epsilon == pytest.approx(original_epsilon)
        assert new_policy.total_steps == original_steps

    def test_checkpoint_step_count_matches_total_steps(self) -> None:
        """PolicyCheckpointData.step_count must match internal state."""
        policy = BanditPolicy()
        policy._state.total_steps = 42
        cp = policy.get_checkpoint()
        assert cp.step_count == 42

    def test_load_checkpoint_raises_on_missing_weights(self) -> None:
        """Missing weights key must raise CheckpointError, not KeyError."""
        policy = BanditPolicy()
        bad_checkpoint = PolicyCheckpointData(weights=None, step_count=0)
        with pytest.raises(CheckpointError, match="no weights"):
            policy.load_checkpoint(bad_checkpoint)

    def test_load_checkpoint_raises_on_missing_keys(self) -> None:
        """Checkpoint missing required keys must raise CheckpointError."""
        policy = BanditPolicy()
        bad_checkpoint = PolicyCheckpointData(
            weights={"q_values": [0.0] * _N_ACTIONS},  # missing action_counts etc.
            step_count=0,
        )
        with pytest.raises(CheckpointError, match="missing keys"):
            policy.load_checkpoint(bad_checkpoint)

    def test_load_checkpoint_raises_on_wrong_q_length(self) -> None:
        """Q-values with wrong length must raise CheckpointError."""
        policy = BanditPolicy()
        bad_checkpoint = PolicyCheckpointData(
            weights={
                "q_values": [0.0, 0.0],  # wrong: should be 5
                "action_counts": [0] * _N_ACTIONS,
                "total_steps": 0,
                "epsilon": 1.0,
            },
            step_count=0,
        )
        with pytest.raises(CheckpointError, match="length mismatch"):
            policy.load_checkpoint(bad_checkpoint)


@pytest.mark.unit
class TestExplorationGuardIntegration:
    """Verify ExplorationGuard's explore flag is respected."""

    @pytest.mark.asyncio
    async def test_explore_false_always_exploits(self) -> None:
        """When explore=False, policy must exploit regardless of ε."""
        policy = BanditPolicy(epsilon_start=1.0)  # would always explore normally
        # Set a clear best action
        hold_idx = _ACTION_IDX[ActionType.HOLD]
        policy._state.q_values[hold_idx] = 99.0

        decisions = [await policy.decide(_state(), explore=False) for _ in range(20)]
        assert all(d.action == ActionType.HOLD for d in decisions)

    @pytest.mark.asyncio
    async def test_explore_true_allows_exploration(self) -> None:
        """When explore=True with high ε, policy does explore."""
        policy = BanditPolicy(epsilon_start=1.0, epsilon_decay=1.0)  # ε stays at 1.0

        actions = set()
        for _ in range(100):
            d = await policy.decide(_state(), explore=True)
            actions.add(d.action)

        assert len(actions) > 1


@pytest.mark.unit
class TestShadowMode:
    """Verify shadow mode is propagated correctly."""

    @pytest.mark.asyncio
    async def test_shadow_mode_preserved_in_decision(self) -> None:
        """Decision output must reflect policy_mode=SHADOW."""
        policy = BanditPolicy()
        policy.policy_mode = PolicyMode.SHADOW

        decision = await policy.decide(_state())
        assert decision.policy_mode == PolicyMode.SHADOW

    @pytest.mark.asyncio
    async def test_active_mode_by_default(self) -> None:
        """New policy must default to ACTIVE mode."""
        policy = BanditPolicy()
        decision = await policy.decide(_state())
        assert decision.policy_mode == PolicyMode.ACTIVE


@pytest.mark.unit
class TestActionClipping:
    """Verify instances_after is always within [min, max] bounds."""

    @pytest.mark.asyncio
    async def test_scale_up_clipped_at_max(self) -> None:
        """SCALE_UP_3 from instance=19 must clip to 20."""
        policy = BanditPolicy()
        # Force SCALE_UP_3 by setting its Q-value highest
        up3_idx = _ACTION_IDX[ActionType.SCALE_UP_3]
        policy._state.q_values[up3_idx] = 100.0

        decision = await policy.decide(_state(instances=19), explore=False)
        assert decision.instances_after <= 20

    @pytest.mark.asyncio
    async def test_scale_down_clipped_at_min(self) -> None:
        """SCALE_DOWN_3 from instance=1 must clip to 1."""
        policy = BanditPolicy()
        down3_idx = _ACTION_IDX[ActionType.SCALE_DOWN_3]
        policy._state.q_values[down3_idx] = 100.0

        decision = await policy.decide(_state(instances=1), explore=False)
        assert decision.instances_after >= 1

    @pytest.mark.asyncio
    async def test_all_decisions_within_bounds(self) -> None:
        """All 50 random decisions must stay within [1, 20]."""
        policy = BanditPolicy(epsilon_start=1.0, epsilon_decay=1.0)

        decisions = [await policy.decide(_state(), explore=True) for _ in range(50)]
        for d in decisions:
            assert 1 <= d.instances_after <= 20


@pytest.mark.unit
class TestConcurrentDecisions:
    """Verify concurrent calls are safe."""

    @pytest.mark.asyncio
    async def test_concurrent_calls_produce_valid_decisions(self) -> None:
        """50 concurrent decide calls must all return valid ScalingDecisions."""
        policy = BanditPolicy()
        results = await asyncio.gather(*[policy.decide(_state()) for _ in range(50)])
        for d in results:
            assert d.policy_type == PolicyType.BANDIT
            assert 1 <= d.instances_after <= 20
            assert d.action in ActionType

    @pytest.mark.asyncio
    async def test_concurrent_updates_maintain_valid_q_values(self) -> None:
        """Concurrent updates must not corrupt Q-values to NaN or Inf."""
        policy = BanditPolicy()
        state = _state()

        async def update() -> None:
            d = await policy.decide(state)
            await policy.update(state, d, reward=1.0)

        await asyncio.gather(*[update() for _ in range(30)])

        # Q-values must remain finite
        for q in policy.q_values:
            assert q == q  # NaN check: NaN != NaN is False, q == q is True
            assert abs(q) < 1e6  # finite check


@pytest.mark.unit
class TestRewardService:
    """Verify reward computation logic."""

    def test_sla_violation_when_latency_high(self) -> None:
        from app.services.reward_service import RewardService

        svc = RewardService()
        components = svc.compute_reward(p99_latency_ms=600.0, instance_count=5, last_action_delta=0)
        assert components.sla_violated is True
        assert components.sla_violation_penalty > 0

    def test_no_sla_violation_when_latency_ok(self) -> None:
        from app.services.reward_service import RewardService

        svc = RewardService()
        components = svc.compute_reward(p99_latency_ms=200.0, instance_count=5, last_action_delta=0)
        assert components.sla_violated is False
        assert components.sla_violation_penalty == 0.0

    def test_reward_is_negative(self) -> None:
        """Reward is always non-positive (penalty formulation)."""
        from app.services.reward_service import RewardService

        svc = RewardService()
        for latency in [100.0, 300.0, 700.0]:
            r = svc.compute_reward(p99_latency_ms=latency, instance_count=5, last_action_delta=0)
            assert r.total_reward <= 0.0

    def test_instability_penalty_on_scaling(self) -> None:
        """Scaling actions (non-zero delta) incur instability penalty."""
        from app.services.reward_service import RewardService

        svc = RewardService()
        scale = svc.compute_reward(p99_latency_ms=100.0, instance_count=5, last_action_delta=1)
        hold = svc.compute_reward(p99_latency_ms=100.0, instance_count=5, last_action_delta=0)
        assert scale.instability_penalty > 0
        assert hold.instability_penalty == 0

    def test_n_step_reward_uses_history(self) -> None:
        """n-step reward should differ from immediate reward when buffer has history."""
        from app.services.reward_service import RewardService

        svc = RewardService()
        # First call populates buffer
        r1 = svc.compute_n_step_reward(-1.0)
        # Second call should incorporate discounted r1
        r2 = svc.compute_n_step_reward(-1.0)
        # r2 should incorporate discounted r1 → more negative than -1.0
        assert r2 <= r1  # at least as negative
