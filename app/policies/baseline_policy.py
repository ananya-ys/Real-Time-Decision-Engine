"""
BaselinePolicy — the indestructible safety net.

WHY THIS EXISTS:
- This is the rollback target. When RL degrades, the system falls back HERE.
- Deterministic: same input → always same output. No randomness. No state.
- Zero external calls: no DB, no Redis, no network. Returns in < 1ms always.
- No exceptions: all edge cases handled. Returns HOLD as ultimate fallback.

WHAT BREAKS IF WRONG:
- If this crashes, rollback has no safe target. System has no fallback.
- If this makes external calls, it can fail when the system is already degraded.
- If this is non-deterministic, rollback produces unpredictable behavior.

INDUSTRY PARALLEL:
- Every auto-pilot system has a manual override. This is the manual override.
- Airlines call it "reversion to direct law". Same principle.
"""

from __future__ import annotations

import structlog

from app.core.config import get_settings
from app.policies.base_policy import PolicyCheckpointData, PolicyInterface
from app.schemas.common import ActionType, PolicyMode, PolicyType
from app.schemas.decision import ScalingDecision
from app.schemas.state import SystemState

logger = structlog.get_logger(__name__)


class BaselinePolicy(PolicyInterface):
    """
    Rule-based threshold scaling policy.

    Decision logic (4 conditions → 4 actions):
    1. cpu > critical OR latency > critical → SCALE_UP_3
    2. cpu > high OR latency > high → SCALE_UP_1
    3. cpu < low AND latency < low AND instances > min → SCALE_DOWN_1
    4. else → HOLD

    All thresholds from environment variables via Settings.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._cpu_critical = settings.baseline_cpu_critical
        self._cpu_high = settings.baseline_cpu_high
        self._cpu_low = settings.baseline_cpu_low
        self._latency_critical = settings.baseline_latency_critical_ms
        self._latency_high = settings.baseline_latency_high_ms
        self._latency_low = settings.baseline_latency_low_ms
        self._min_instances = settings.min_instances
        self._max_instances = settings.max_instances
        self.policy_mode = PolicyMode.ACTIVE

    @property
    def policy_type(self) -> PolicyType:
        return PolicyType.BASELINE

    async def decide(self, state: SystemState, explore: bool = True) -> ScalingDecision:
        """
        Make a deterministic scaling decision based on thresholds.

        This method NEVER raises an exception.
        explore parameter is ignored — baseline does not explore.
        """
        instances_before = state.instance_count
        action = self._select_action(state)
        delta = self._action_to_delta(action)
        raw_target = instances_before + delta
        instances_after = self.clip_instances(raw_target, self._min_instances, self._max_instances)

        # If clipping changed the action, adjust to match
        actual_delta = instances_after - instances_before
        if actual_delta != delta:
            action = self._delta_to_action(actual_delta)

        return ScalingDecision(
            action=action,
            instances_before=instances_before,
            instances_after=instances_after,
            policy_type=self.policy_type,
            policy_mode=self.policy_mode,
            confidence=1.0,  # baseline is always 100% confident (deterministic)
        )

    async def update(self, state: SystemState, action: ScalingDecision, reward: float) -> None:
        """No-op. Baseline does not learn."""
        pass

    def get_checkpoint(self) -> PolicyCheckpointData:
        """Return empty checkpoint — baseline has no trainable state."""
        return PolicyCheckpointData(
            weights={"type": "baseline", "version": 1},
            step_count=0,
            performance_metric=None,
        )

    def load_checkpoint(self, checkpoint: PolicyCheckpointData) -> None:
        """No-op. Baseline has no state to restore."""
        pass

    def _select_action(self, state: SystemState) -> ActionType:
        """Core decision logic — threshold evaluation."""
        cpu = state.cpu_utilization
        latency = state.p99_latency_ms

        # Condition 1: Critical — aggressive scale up
        if cpu > self._cpu_critical or latency > self._latency_critical:
            return ActionType.SCALE_UP_3

        # Condition 2: High — moderate scale up
        if cpu > self._cpu_high or latency > self._latency_high:
            return ActionType.SCALE_UP_1

        # Condition 3: Low and room to scale down
        if (
            cpu < self._cpu_low
            and latency < self._latency_low
            and state.instance_count > self._min_instances
        ):
            return ActionType.SCALE_DOWN_1

        # Condition 4: Otherwise hold
        return ActionType.HOLD

    @staticmethod
    def _action_to_delta(action: ActionType) -> int:
        """Convert action enum to instance count delta."""
        deltas = {
            ActionType.SCALE_UP_1: 1,
            ActionType.SCALE_UP_3: 3,
            ActionType.SCALE_DOWN_1: -1,
            ActionType.SCALE_DOWN_3: -3,
            ActionType.HOLD: 0,
        }
        return deltas.get(action, 0)

    @staticmethod
    def _delta_to_action(delta: int) -> ActionType:
        """Convert instance count delta back to action enum."""
        if delta >= 3:
            return ActionType.SCALE_UP_3
        elif delta >= 1:
            return ActionType.SCALE_UP_1
        elif delta <= -3:
            return ActionType.SCALE_DOWN_3
        elif delta <= -1:
            return ActionType.SCALE_DOWN_1
        return ActionType.HOLD
