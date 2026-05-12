"""
RewardService — computes and persists reward signals.

WHY THIS EXISTS:
- Reward computation is a domain concept, not a DB concern. Service layer owns it.
- n-step discounted return handles the delayed reward problem:
  The action taken at t=0 causes latency improvement at t=N.
  Single-step reward attributes improvement to the WRONG action.
- SLA violation detection feeds ExplorationGuard.policy_stats for real-time safety.

THE N-STEP RETURN:
  R_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γⁿ·r_{t+n}

  In simulation: we compute this using the pending_rewards buffer.
  When reward at t arrives, we also add discounted future rewards from the buffer.
  This teaches the RL policy that "scale up now → latency drops in N ticks".

WHAT BREAKS IF WRONG:
- Single-step reward only: policy learns that action A caused reward at the same tick.
  In reality, provisioning takes N ticks. The policy trains on wrong causality.
- No SLA violation tracking: ExplorationGuard has no signal to suppress exploration.
- Reward not logged: drift detector has no history to test against.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.reward_log import RewardLog

logger = structlog.get_logger(__name__)

# SLA threshold: P99 latency above this = SLA violation
_SLA_LATENCY_THRESHOLD_MS = 500.0


@dataclass
class RewardComponents:
    """Breakdown of reward computation for logging and debugging."""

    latency_penalty: float
    cost_penalty: float
    sla_violation_penalty: float
    instability_penalty: float
    total_reward: float
    sla_violated: bool


@dataclass
class PendingReward:
    """A reward observation waiting to be incorporated into n-step returns."""

    decision_log_id: uuid.UUID
    reward: float
    timestamp_tick: int


class RewardService:
    """
    Computes reward signals and persists them to the database.

    Two reward signals:
    1. Immediate reward r_t: computed from current state metrics.
    2. n-step reward R_t: discounted sum over next N ticks (handles provisioning delay).
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._alpha = settings.reward_alpha_latency
        self._beta = settings.reward_beta_cost
        self._gamma_sla = settings.reward_gamma_sla
        self._delta = settings.reward_delta_instability
        self._gamma_discount = settings.rl_gamma  # discount factor for n-step
        self._n_steps = settings.provisioning_delay_ticks

        # Buffer for n-step return computation
        # Keyed by decision_log_id, contains recent rewards for discounting
        self._pending_rewards: deque[PendingReward] = deque(maxlen=self._n_steps * 2)
        self._tick = 0

    def compute_reward(
        self,
        p99_latency_ms: float,
        instance_count: int,
        last_action_delta: int,
        cost_per_instance: float = 0.01,
        sla_threshold_ms: float = _SLA_LATENCY_THRESHOLD_MS,
    ) -> RewardComponents:
        """
        Compute the scalar reward for a single timestep.

        Formula: R = -(α·latency_penalty + β·cost_penalty + γ·sla_violation + δ·instability)

        All weights configurable via env vars. Negative because we maximize reward
        by minimizing penalties. This is a cost-minimization objective.

        Args:
            p99_latency_ms: Current P99 latency.
            instance_count: Current number of running instances.
            last_action_delta: Change in instance count (for instability penalty).
            cost_per_instance: Cost per instance per tick (configurable).
            sla_threshold_ms: P99 latency above this = SLA breach.

        Returns:
            RewardComponents with breakdown for debugging + total_reward.
        """
        # Latency penalty: proportional to how far above target latency we are
        latency_penalty = max(0.0, (p99_latency_ms - sla_threshold_ms) / sla_threshold_ms)

        # Cost penalty: proportional to instances running
        cost_penalty = instance_count * cost_per_instance

        # SLA violation: binary signal (or graded if latency >> threshold)
        sla_violated = p99_latency_ms > sla_threshold_ms
        # Graded: worse violations get stronger signal
        sla_violation_penalty = (
            1.0 + (p99_latency_ms - sla_threshold_ms) / sla_threshold_ms if sla_violated else 0.0
        )

        # Instability penalty: penalize oscillating scale decisions
        instability_penalty = 1.0 if abs(last_action_delta) > 0 and last_action_delta != 0 else 0.0

        total_reward = -(
            self._alpha * latency_penalty
            + self._beta * cost_penalty
            + self._gamma_sla * sla_violation_penalty
            + self._delta * instability_penalty
        )

        return RewardComponents(
            latency_penalty=latency_penalty,
            cost_penalty=cost_penalty,
            sla_violation_penalty=sla_violation_penalty,
            instability_penalty=instability_penalty,
            total_reward=total_reward,
            sla_violated=sla_violated,
        )

    def compute_n_step_reward(self, immediate_reward: float) -> float:
        """
        Compute n-step discounted return using the pending rewards buffer.

        When an immediate reward arrives at tick t:
        R_t = r_t + γ·r_{t-1} + γ²·r_{t-2} + ... + γⁿ·r_{t-n}

        The buffer holds recent rewards. We discount them backwards in time.
        This approximates the true n-step return without needing future rewards.

        Note: Full n-step return requires future rewards (r_{t+1} ... r_{t+n}).
        This implementation uses recent history as a proxy. For true n-step,
        the RL training loop handles this in the replay buffer (Phase 5).
        """
        self._tick += 1

        # Add current reward to pending buffer
        n_step_reward = immediate_reward
        discount = self._gamma_discount

        # Discount and sum recent rewards
        for pending in reversed(list(self._pending_rewards)):
            n_step_reward += discount * pending.reward
            discount *= self._gamma_discount

        # Add to buffer for future discounting
        self._pending_rewards.append(
            PendingReward(
                decision_log_id=uuid.uuid4(),  # placeholder
                reward=immediate_reward,
                timestamp_tick=self._tick,
            )
        )

        return n_step_reward

    async def log_reward(
        self,
        decision_log_id: uuid.UUID,
        reward_components: RewardComponents,
        n_step_reward: float,
        cumulative_reward: float,
        cumulative_regret: float | None,
        baseline_reward: float | None,
        db: AsyncSession,
    ) -> RewardLog:
        """
        Persist reward to database.

        Called asynchronously after environment responds to the decision.
        The RewardLog is the source of truth for drift detection.
        """
        reward_log = RewardLog(
            decision_log_id=decision_log_id,
            reward=reward_components.total_reward,
            n_step_reward=n_step_reward,
            cumulative_reward=cumulative_reward,
            cumulative_regret=cumulative_regret,
            baseline_reward=baseline_reward,
        )
        db.add(reward_log)
        await db.flush()

        logger.info(
            "reward_logged",
            decision_log_id=str(decision_log_id),
            reward=round(reward_components.total_reward, 4),
            n_step=round(n_step_reward, 4),
            sla_violated=reward_components.sla_violated,
            cumulative=round(cumulative_reward, 2),
        )

        return reward_log

    def is_sla_violated(self, p99_latency_ms: float) -> bool:
        """Check if current latency constitutes an SLA violation."""
        return p99_latency_ms > _SLA_LATENCY_THRESHOLD_MS
