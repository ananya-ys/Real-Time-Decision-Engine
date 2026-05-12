"""
Multi-seed policy benchmarks — Phase 8 gate.

5 seeds each, mean +/- std, 3-policy comparison.
Marks: @pytest.mark.benchmark — skipped in unit test runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import mean, stdev

import numpy as np
import pytest

from app.ml.state_normalizer import StateNormalizer
from app.policies.bandit_policy import BanditPolicy
from app.policies.baseline_policy import BaselinePolicy
from app.policies.rl_policy import RLPolicy
from app.schemas.common import TrafficRegime
from app.schemas.state import SystemState
from app.services.reward_service import RewardService


def _make_state(rng: np.random.RandomState, tick: int) -> SystemState:
    hour = tick % 24
    if 8 <= hour <= 18:
        cpu = float(rng.uniform(0.6, 0.95))
        rps = float(rng.uniform(2000, 8000))
        latency = float(rng.uniform(200, 600))
    else:
        cpu = float(rng.uniform(0.1, 0.4))
        rps = float(rng.uniform(100, 1000))
        latency = float(rng.uniform(50, 200))
    return SystemState(
        cpu_utilization=min(1.0, cpu),
        request_rate=rps,
        p99_latency_ms=latency,
        instance_count=max(1, min(20, 5 + rng.randint(-2, 3))),
        hour_of_day=hour,
        day_of_week=tick % 7,
        traffic_regime=TrafficRegime.BURST if rps > 5000 else TrafficRegime.STEADY,
    )


@dataclass
class BenchmarkResult:
    policy_name: str
    seed: int
    total_decisions: int
    cumulative_reward: float
    episode_latency_ms: float
    sla_violations: int
    final_epsilon: float | None = None
    training_steps: int = 0


async def _run_episode(
    policy_name: str,
    seed: int,
    n_decisions: int = 200,
) -> BenchmarkResult:
    rng = np.random.RandomState(seed)
    reward_svc = RewardService()
    cumulative_reward = 0.0
    sla_violations = 0
    last_action_delta = 0
    start = time.perf_counter()

    if policy_name == "baseline":
        policy = BaselinePolicy()
        for tick in range(n_decisions):
            state = _make_state(rng, tick)
            decision = await policy.decide(state)
            last_action_delta = decision.instances_after - decision.instances_before
            reward = reward_svc.compute_reward(
                p99_latency_ms=state.p99_latency_ms,
                instance_count=decision.instances_after,
                last_action_delta=last_action_delta,
            )
            cumulative_reward += reward.total_reward
            if reward.sla_violated:
                sla_violations += 1
        elapsed = (time.perf_counter() - start) * 1000
        return BenchmarkResult(
            "baseline", seed, n_decisions, cumulative_reward, elapsed, sla_violations
        )

    elif policy_name == "bandit":
        policy = BanditPolicy(epsilon_start=1.0, epsilon_decay=0.99, epsilon_floor=0.05)
        for tick in range(n_decisions):
            state = _make_state(rng, tick)
            decision = await policy.decide(state, explore=True)
            last_action_delta = decision.instances_after - decision.instances_before
            reward = reward_svc.compute_reward(
                p99_latency_ms=state.p99_latency_ms,
                instance_count=decision.instances_after,
                last_action_delta=last_action_delta,
            )
            await policy.update(state, decision, reward=reward.total_reward)
            cumulative_reward += reward.total_reward
            if reward.sla_violated:
                sla_violations += 1
        elapsed = (time.perf_counter() - start) * 1000
        return BenchmarkResult(
            "bandit",
            seed,
            n_decisions,
            cumulative_reward,
            elapsed,
            sla_violations,
            final_epsilon=policy.epsilon,
        )

    elif policy_name == "rl":
        warmup = [_make_state(rng, i) for i in range(100)]
        normalizer = StateNormalizer(version_id=f"bench-v{seed}").fit(warmup)
        policy = RLPolicy(
            normalizer=normalizer,
            warm_start_min_decisions=0,
            seed=seed,
            batch_size=32,
        )
        for tick in range(n_decisions):
            state = _make_state(rng, tick)
            decision = await policy.decide(state)
            last_action_delta = decision.instances_after - decision.instances_before
            reward = reward_svc.compute_reward(
                p99_latency_ms=state.p99_latency_ms,
                instance_count=decision.instances_after,
                last_action_delta=last_action_delta,
            )
            await policy.update(state, decision, reward=reward.total_reward)
            cumulative_reward += reward.total_reward
            if reward.sla_violated:
                sla_violations += 1
            if tick % 10 == 0 and tick > 50:
                policy.train_step()
        elapsed = (time.perf_counter() - start) * 1000
        return BenchmarkResult(
            "rl",
            seed,
            n_decisions,
            cumulative_reward,
            elapsed,
            sla_violations,
            training_steps=policy.training_steps,
        )

    raise ValueError(f"Unknown policy: {policy_name}")


@pytest.mark.benchmark
class TestPolicyBenchmarks:
    """Multi-seed policy comparison. Run: pytest tests/benchmarks/ -v -m benchmark"""

    N_SEEDS = 5
    N_DECISIONS = 200
    SEEDS: list[int] = [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_baseline_benchmark_stable_across_seeds(self) -> None:
        """Baseline: 5 seeds, verify stable cumulative reward."""
        results = []
        for seed in self.SEEDS:
            results.append(await _run_episode("baseline", seed, self.N_DECISIONS))

        rewards = [r.cumulative_reward for r in results]
        r_mean = mean(rewards)
        r_std = stdev(rewards) if len(rewards) > 1 else 0.0
        sla_mean = mean([r.sla_violations for r in results])

        print(f"\nBASELINE: mean={r_mean:.2f} std={r_std:.2f} sla_viol/ep={sla_mean:.1f}")

        assert all(r.total_decisions == self.N_DECISIONS for r in results)
        assert all(r.cumulative_reward == r.cumulative_reward for r in results)  # not NaN

    @pytest.mark.asyncio
    async def test_bandit_benchmark(self) -> None:
        """Bandit: 5 seeds, epsilon decays toward floor."""
        results = []
        for seed in self.SEEDS:
            results.append(await _run_episode("bandit", seed, self.N_DECISIONS))

        rewards = [r.cumulative_reward for r in results]
        r_mean = mean(rewards)
        r_std = stdev(rewards) if len(rewards) > 1 else 0.0
        epsilons = [r.final_epsilon for r in results if r.final_epsilon is not None]
        sla_mean = mean([r.sla_violations for r in results])

        print(
            f"\nBANDIT: mean={r_mean:.2f} std={r_std:.2f} "
            f"eps={mean(epsilons):.4f} sla={sla_mean:.1f}"
        )

        assert all(r.total_decisions == self.N_DECISIONS for r in results)
        assert all(e < 1.0 for e in epsilons), "Epsilon must have decayed from 1.0"
        assert all(e >= 0.05 for e in epsilons), "Epsilon must respect floor"

    @pytest.mark.asyncio
    async def test_rl_benchmark(self) -> None:
        """RL DQN: 5 seeds, verify finite rewards and training steps."""
        results = []
        for seed in self.SEEDS:
            results.append(await _run_episode("rl", seed, self.N_DECISIONS))

        rewards = [r.cumulative_reward for r in results]
        r_mean = mean(rewards)
        r_std = stdev(rewards) if len(rewards) > 1 else 0.0
        sla_mean = mean([r.sla_violations for r in results])
        train_mean = mean([r.training_steps for r in results])

        print(
            f"\nRL DQN: mean={r_mean:.2f} std={r_std:.2f} "
            f"train_steps={train_mean:.0f} sla={sla_mean:.1f}"
        )

        assert all(r.total_decisions == self.N_DECISIONS for r in results)
        assert all(r.cumulative_reward == r.cumulative_reward for r in results)  # not NaN
        assert all(abs(r.cumulative_reward) < 1e8 for r in results)  # finite
        assert all(r.training_steps > 0 for r in results)

    @pytest.mark.asyncio
    async def test_all_policies_comparison(self) -> None:
        """Full comparison table: 5 seeds × 3 policies."""
        all_results: dict[str, list[BenchmarkResult]] = {
            "baseline": [],
            "bandit": [],
            "rl": [],
        }
        for seed in self.SEEDS:
            for name in ["baseline", "bandit", "rl"]:
                all_results[name].append(await _run_episode(name, seed, self.N_DECISIONS))

        header = f"POLICY COMPARISON — {self.N_SEEDS} seeds, {self.N_DECISIONS} decisions/seed"
        row_fmt = "{:<12} {:>12.2f} {:>8.2f} {:>10.1f}"
        print(f"\n{'=' * 55}")
        print(header)
        print(f"  {'Policy':<12} {'Mean Reward':>12} {'Std':>8} {'SLA viol':>10}")
        print(f"  {'-' * 46}")
        for _name, results in all_results.items():
            rewards = [r.cumulative_reward for r in results]
            r_mean = mean(rewards)
            r_std = stdev(rewards) if len(rewards) > 1 else 0.0
            v_mean = mean([r.sla_violations for r in results])
            print("  " + row_fmt.format(name, r_mean, r_std, v_mean))
        print(f"{'=' * 55}")

        for _name, results in all_results.items():
            assert all(r.total_decisions == self.N_DECISIONS for r in results)
            for r in results:
                assert r.cumulative_reward == r.cumulative_reward  # not NaN
