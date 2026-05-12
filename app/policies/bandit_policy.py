"""
BanditPolicy — Contextual Bandit with ε-greedy and UCB exploration.

WHY THIS EXISTS:
- Intermediate step between the deterministic BaselinePolicy and full RL.
- Learns online: after every (state, action, reward) tuple, Q-values update immediately.
  No batch training. No replay buffer. No gradient descent. Pure online statistics.
- Provides a warm-up baseline of experience before RL training begins.
- Regret tracking measures how far below optimal we are — used in benchmarking.

THE ALGORITHM:
  Q(a) = running average reward for action a
  ε-greedy: with probability ε → random action (explore)
             with probability 1-ε → argmax Q(a) (exploit)
  UCB:       select action with highest Q̂(a) + c·√(ln(t) / N(a))
             where t = total steps, N(a) = times action a was taken

WHAT BREAKS IF WRONG:
- ε never decays → policy always explores → never converges on best action.
- ε_floor = 0.0 → policy stops exploring entirely → can't adapt to pattern changes.
- No checkpoint → Q-values lost on restart → policy starts from scratch every time.
- Shared Q-values mutated under concurrent reads → race condition in inference.

INDUSTRY PARALLEL:
- Multi-armed bandit is the simplest online reinforcement learning algorithm.
- Used in recommendation systems, ad selection, clinical trials.
- Netflix, Google, and Amazon use bandit variants to test new content before full rollout.

SHADOW MODE:
- When policy_mode = SHADOW, decide() computes action normally but caller does NOT
  commit it to environment. The action is logged with shadow_flag=True for comparison.
- This is Phase 6's shadow gate mechanism, scaffolded here so BanditPolicy is shadow-ready.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import StrEnum

import structlog

from app.policies.base_policy import PolicyCheckpointData, PolicyInterface
from app.schemas.common import ActionType, PolicyMode, PolicyType
from app.schemas.decision import ScalingDecision
from app.schemas.state import SystemState

logger = structlog.get_logger(__name__)

# All 5 discrete actions in a fixed, stable order for indexing
_ACTIONS: list[ActionType] = [
    ActionType.SCALE_UP_3,
    ActionType.SCALE_UP_1,
    ActionType.HOLD,
    ActionType.SCALE_DOWN_1,
    ActionType.SCALE_DOWN_3,
]
_N_ACTIONS = len(_ACTIONS)
_ACTION_IDX: dict[ActionType, int] = {a: i for i, a in enumerate(_ACTIONS)}


class ExplorationStrategy(StrEnum):
    EPSILON_GREEDY = "epsilon_greedy"
    UCB = "ucb"


@dataclass
class BanditState:
    """
    Serializable internal state of the BanditPolicy.

    All mutable state lives here so checkpoint is just: serialize this dataclass.
    """

    q_values: list[float] = field(default_factory=lambda: [0.0] * _N_ACTIONS)
    action_counts: list[int] = field(default_factory=lambda: [0] * _N_ACTIONS)
    total_steps: int = 0
    epsilon: float = 1.0
    cumulative_reward: float = 0.0
    cumulative_regret: float = 0.0


class BanditPolicy(PolicyInterface):
    """
    Contextual Bandit Policy with ε-greedy or UCB exploration.

    Online Q-value update rule (incremental mean):
        Q(a) ← Q(a) + (1/N(a)) · (reward - Q(a))

    This is equivalent to computing the running average, but in O(1) space.
    No need to store all past rewards — only the current estimate and count.

    Parameters (all from config / constructor):
        strategy:    EPSILON_GREEDY or UCB
        epsilon_start: Initial exploration rate (1.0 = always explore)
        epsilon_floor: Minimum exploration rate (never decays to zero)
        epsilon_decay: Multiplicative decay per step
        ucb_c:       UCB confidence parameter (higher = more exploration)
        min_instances / max_instances: Action clipping bounds
    """

    def __init__(
        self,
        strategy: ExplorationStrategy = ExplorationStrategy.EPSILON_GREEDY,
        epsilon_start: float = 1.0,
        epsilon_floor: float = 0.05,
        epsilon_decay: float = 0.995,
        ucb_c: float = 2.0,
        min_instances: int = 1,
        max_instances: int = 20,
    ) -> None:
        self._strategy = strategy
        self._epsilon_floor = epsilon_floor
        self._epsilon_decay = epsilon_decay
        self._ucb_c = ucb_c
        self._min_instances = min_instances
        self._max_instances = max_instances

        self._state = BanditState(epsilon=epsilon_start)
        self.policy_mode = PolicyMode.ACTIVE

        logger.info(
            "bandit_policy_initialized",
            strategy=strategy.value,
            epsilon_start=epsilon_start,
            epsilon_floor=epsilon_floor,
        )

    @property
    def policy_type(self) -> PolicyType:
        return PolicyType.BANDIT

    async def decide(self, state: SystemState, explore: bool = True) -> ScalingDecision:
        """
        Select a scaling action using ε-greedy or UCB.

        Args:
            state: Current system state (S in MDP).
            explore: If False (set by ExplorationGuard during stress), always exploit.

        Returns:
            ScalingDecision with action, instances, confidence, and Q-values for explainer.

        The explore flag integrates directly with ExplorationGuard:
        - Guard passes False when system is under stress → we always pick argmax Q(a).
        - Guard passes True when system is stable → we apply ε-greedy or UCB.
        """
        instances_before = state.instance_count

        # Select action based on strategy
        if not explore:
            # ExplorationGuard suppressed — always exploit
            action_idx = self._exploit()
        elif self._strategy == ExplorationStrategy.EPSILON_GREEDY:
            action_idx = self._epsilon_greedy_select()
        else:
            action_idx = self._ucb_select()

        action = _ACTIONS[action_idx]

        # Compute instances_after with safety clipping
        delta = self._action_delta(action)
        raw_target = instances_before + delta
        instances_after = self.clip_instances(raw_target, self._min_instances, self._max_instances)

        # Adjust action to match clipped result
        actual_delta = instances_after - instances_before
        if actual_delta != delta:
            action = self._delta_to_action(actual_delta)
            action_idx = _ACTION_IDX.get(action, _ACTION_IDX[ActionType.HOLD])

        # Compute confidence spread (max_Q - second_max_Q)
        sorted_q = sorted(self._state.q_values, reverse=True)
        confidence_spread = sorted_q[0] - sorted_q[1] if len(sorted_q) > 1 else 0.0

        # Normalize confidence to [0, 1] — higher = more certain about best action
        q_range = max(abs(q) for q in self._state.q_values) or 1.0
        confidence = min(1.0, abs(confidence_spread) / q_range)

        return ScalingDecision(
            action=action,
            instances_before=instances_before,
            instances_after=instances_after,
            policy_type=self.policy_type,
            policy_mode=self.policy_mode,
            confidence=confidence,
            q_values={a.value: round(self._state.q_values[i], 4) for i, a in enumerate(_ACTIONS)},
        )

    async def update(self, state: SystemState, action: ScalingDecision, reward: float) -> None:
        """
        Online incremental mean update after receiving reward.

        Q(a) ← Q(a) + (1/N(a)) · (reward - Q(a))

        This is mathematically equivalent to computing the running average of all
        rewards observed for action a, but requires only O(1) space.

        Called AFTER the decision is committed and the environment has responded.
        Must be fast — this runs in the request path for shadow updates.
        """
        action_idx = _ACTION_IDX.get(action.action, _ACTION_IDX[ActionType.HOLD])

        # Increment count
        self._state.action_counts[action_idx] += 1
        self._state.total_steps += 1
        n = self._state.action_counts[action_idx]

        # Incremental mean update
        old_q = self._state.q_values[action_idx]
        new_q = old_q + (1.0 / n) * (reward - old_q)
        self._state.q_values[action_idx] = new_q

        # Decay epsilon (ε-greedy only)
        if self._strategy == ExplorationStrategy.EPSILON_GREEDY:
            self._state.epsilon = max(
                self._epsilon_floor,
                self._state.epsilon * self._epsilon_decay,
            )

        # Track cumulative reward and regret
        self._state.cumulative_reward += reward

        # Regret = max_Q - Q(chosen_action), represents opportunity cost
        max_q = max(self._state.q_values)
        regret = max_q - new_q
        self._state.cumulative_regret += max(0.0, regret)

        logger.debug(
            "bandit_updated",
            action=action.action.value,
            reward=round(reward, 3),
            new_q=round(new_q, 4),
            epsilon=round(self._state.epsilon, 4),
            total_steps=self._state.total_steps,
        )

    def get_checkpoint(self) -> PolicyCheckpointData:
        """
        Serialize bandit state for DB persistence.

        Checkpoint is stored as JSONB in policy_checkpoints table.
        Restored on service startup — bandit picks up exactly where it left off.
        """
        return PolicyCheckpointData(
            weights={
                "q_values": self._state.q_values,
                "action_counts": self._state.action_counts,
                "total_steps": self._state.total_steps,
                "epsilon": self._state.epsilon,
                "cumulative_reward": self._state.cumulative_reward,
                "cumulative_regret": self._state.cumulative_regret,
                "strategy": self._strategy.value,
            },
            step_count=self._state.total_steps,
            performance_metric=self._state.cumulative_reward,
        )

    def load_checkpoint(self, checkpoint: PolicyCheckpointData) -> None:
        """
        Restore bandit state from a checkpoint.

        Called on service startup after a crash or restart.
        No data loss — Q-values and epsilon resume exactly.

        Validates that checkpoint has the right structure.
        If validation fails, raises CheckpointError (loud failure, not silent wrong inference).
        """
        from app.core.exceptions import CheckpointError

        weights = checkpoint.weights
        if weights is None:
            raise CheckpointError("Bandit checkpoint has no weights")

        required_keys = {"q_values", "action_counts", "total_steps", "epsilon"}
        missing = required_keys - set(weights.keys())
        if missing:
            raise CheckpointError(f"Bandit checkpoint missing keys: {missing}")

        q_values = weights["q_values"]
        if len(q_values) != _N_ACTIONS:
            raise CheckpointError(
                f"Bandit checkpoint q_values length mismatch: "
                f"expected {_N_ACTIONS}, got {len(q_values)}"
            )

        self._state = BanditState(
            q_values=list(q_values),
            action_counts=list(weights["action_counts"]),
            total_steps=int(weights["total_steps"]),
            epsilon=float(weights["epsilon"]),
            cumulative_reward=float(weights.get("cumulative_reward", 0.0)),
            cumulative_regret=float(weights.get("cumulative_regret", 0.0)),
        )

        logger.info(
            "bandit_checkpoint_loaded",
            total_steps=self._state.total_steps,
            epsilon=round(self._state.epsilon, 4),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _exploit(self) -> int:
        """Select action with highest Q-value (no exploration)."""
        return int(max(range(_N_ACTIONS), key=lambda i: self._state.q_values[i]))

    def _epsilon_greedy_select(self) -> int:
        """
        ε-greedy action selection.

        With probability ε → random action (explore).
        With probability 1-ε → argmax Q(a) (exploit).

        ε decays over time, approaching ε_floor asymptotically.
        ε_floor > 0 ensures the policy never stops adapting to new patterns.
        """
        if random.random() < self._state.epsilon:
            return random.randint(0, _N_ACTIONS - 1)
        return self._exploit()

    def _ucb_select(self) -> int:
        """
        UCB (Upper Confidence Bound) action selection.

        Selects: argmax[Q(a) + c · √(ln(t) / N(a))]

        The confidence bound decreases as we try an action more often (N(a) grows).
        Actions not yet tried get infinite upper bound → tried first.
        No ε parameter to tune — exploration driven purely by uncertainty.

        c (ucb_c) controls exploration-exploitation tradeoff:
        - Higher c → more exploration of uncertain actions.
        - Lower c → more exploitation of known-good actions.
        """
        t = self._state.total_steps + 1  # avoid log(0)

        ucb_values = []
        for i in range(_N_ACTIONS):
            n = self._state.action_counts[i]
            if n == 0:
                # Never tried → infinite upper bound → try it first
                ucb_values.append(float("inf"))
            else:
                confidence_bonus = self._ucb_c * math.sqrt(math.log(t) / n)
                ucb_values.append(self._state.q_values[i] + confidence_bonus)

        return int(max(range(_N_ACTIONS), key=lambda i: ucb_values[i]))

    @staticmethod
    def _action_delta(action: ActionType) -> int:
        deltas = {
            ActionType.SCALE_UP_3: 3,
            ActionType.SCALE_UP_1: 1,
            ActionType.HOLD: 0,
            ActionType.SCALE_DOWN_1: -1,
            ActionType.SCALE_DOWN_3: -3,
        }
        return deltas.get(action, 0)

    @staticmethod
    def _delta_to_action(delta: int) -> ActionType:
        if delta >= 3:
            return ActionType.SCALE_UP_3
        elif delta >= 1:
            return ActionType.SCALE_UP_1
        elif delta <= -3:
            return ActionType.SCALE_DOWN_3
        elif delta <= -1:
            return ActionType.SCALE_DOWN_1
        return ActionType.HOLD

    # ── Read-only properties for monitoring ──────────────────────────────────

    @property
    def epsilon(self) -> float:
        return self._state.epsilon

    @property
    def total_steps(self) -> int:
        return self._state.total_steps

    @property
    def q_values(self) -> list[float]:
        return list(self._state.q_values)

    @property
    def cumulative_reward(self) -> float:
        return self._state.cumulative_reward

    @property
    def cumulative_regret(self) -> float:
        return self._state.cumulative_regret

    @property
    def action_counts(self) -> list[int]:
        return list(self._state.action_counts)

    @property
    def strategy(self) -> ExplorationStrategy:
        return self._strategy
