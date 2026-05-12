"""
PolicyInterface — the abstract base class that all policies implement.

WHY THIS EXISTS:
- DecisionService depends on this abstraction, not concrete policies (Dependency Inversion).
- Swapping DQN for PPO, or Bandit for RL, requires ZERO changes in the service layer.
- CI typecheck enforces: any class claiming to be a policy must implement all 4 methods.

WHAT BREAKS IF WRONG:
- Service imports concrete policy = policy swap requires service code change.
- No ABC enforcement = a policy missing decide() discovered at runtime, not at lint time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.schemas.common import PolicyMode, PolicyType
from app.schemas.decision import ScalingDecision
from app.schemas.state import SystemState


@dataclass
class PolicyCheckpointData:
    """Serializable checkpoint data for persistence."""

    weights: dict  # type: ignore[type-arg]
    step_count: int
    performance_metric: float | None = None


class PolicyInterface(ABC):
    """
    Abstract base class for all RTDE policies.

    Contract:
    - decide(): Given state, return a ScalingDecision. No DB. No Redis. No network.
    - update(): Online learning update after reward feedback. Called async AFTER commit.
    - get_checkpoint(): Serialize internal state for persistence.
    - load_checkpoint(): Restore from serialized state.

    Properties:
    - policy_type: Which policy this is (BASELINE, BANDIT, RL).
    - policy_mode: Whether decisions are committed (ACTIVE) or just logged (SHADOW).
    """

    policy_mode: PolicyMode = PolicyMode.ACTIVE

    @property
    @abstractmethod
    def policy_type(self) -> PolicyType:
        """Return the policy type identifier."""
        ...

    @abstractmethod
    async def decide(self, state: SystemState, explore: bool = True) -> ScalingDecision:
        """
        Select a scaling action given the current system state.

        Args:
            state: Current system state (the S in the MDP).
            explore: Whether exploration is allowed (ExplorationGuard may set False).

        Returns:
            ScalingDecision with action, instances_before/after, policy metadata.

        Rules:
            - MUST return within 10ms for SLO compliance.
            - MUST NOT make any DB, Redis, or network calls.
            - MUST clip instances_after to [min_instances, max_instances].
            - MUST NOT raise exceptions — return HOLD as fallback.
        """
        ...

    @abstractmethod
    async def update(self, state: SystemState, action: ScalingDecision, reward: float) -> None:
        """
        Online learning update after receiving reward feedback.

        Called asynchronously AFTER the decision is committed and reward is computed.
        BaselinePolicy is a no-op. BanditPolicy updates Q-values. RLPolicy stores to buffer.
        """
        ...

    @abstractmethod
    def get_checkpoint(self) -> PolicyCheckpointData:
        """Serialize policy state for persistence across restarts."""
        ...

    @abstractmethod
    def load_checkpoint(self, checkpoint: PolicyCheckpointData) -> None:
        """Restore policy state from a checkpoint."""
        ...

    def clip_instances(self, target: int, min_inst: int, max_inst: int) -> int:
        """
        Hard safety boundary — clip instance count to valid range.

        This is the LAST line of defense before an action reaches the environment.
        If this function is not called, an unsafe action can execute.
        """
        return max(min_inst, min(target, max_inst))
