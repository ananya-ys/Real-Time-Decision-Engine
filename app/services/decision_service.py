"""
DecisionService — core decision orchestration.

WHY THIS EXISTS:
- ALL business logic lives here.
- Orchestrates:
  validate → lock → policy.decide() → audit log → commit → metrics.
- Handles fallback automatically.

FIXES INCLUDED:
- Returns decision_log_id in API response
- Refreshes persisted DB object before returning
- Enables rewards/audit/explainability linkage
"""

from __future__ import annotations

import time
import uuid

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import metrics as prom
from app.core.config import get_settings
from app.models.decision_log import DecisionLog
from app.models.environment_state import EnvironmentState
from app.models.scaling_action import ScalingAction
from app.policies.base_policy import PolicyInterface
from app.policies.baseline_policy import BaselinePolicy
from app.safety.exploration_guard import ExplorationGuard, PolicyStats
from app.schemas.common import ActionType, PolicyMode, PolicyType
from app.schemas.decision import DecisionResponse, ScalingDecision
from app.schemas.state import SystemState

logger = structlog.get_logger(__name__)


class DecisionService:
    """
    Orchestrates the full decision cycle.
    """

    def __init__(self) -> None:
        settings = get_settings()

        self._min_instances = settings.min_instances
        self._max_instances = settings.max_instances

        self._baseline = BaselinePolicy()
        self._active_policy: PolicyInterface = self._baseline

        self._guard = ExplorationGuard()
        self._policy_stats = PolicyStats()

        from app.operator.kill_switch import KillSwitch
        from app.operator.manual_override import ManualOverride

        self._kill_switch = KillSwitch()
        self._manual_override = ManualOverride()

        self._active_policy_version_id: str | None = None

    def set_active_policy(
        self,
        policy: PolicyInterface,
        policy_version_id: str | None = None,
    ) -> None:
        logger.info(
            "policy_swapped",
            from_policy=self._active_policy.policy_type.value,
            to_policy=policy.policy_type.value,
            version_id=policy_version_id,
        )

        self._active_policy = policy
        self._active_policy_version_id = policy_version_id

    def get_active_policy_type(self) -> PolicyType:
        return self._active_policy.policy_type

    async def make_decision(
        self,
        state: SystemState,
        trace_id: uuid.UUID,
        db: AsyncSession,
    ) -> DecisionResponse:
        """
        Execute the full decision cycle.
        """

        start_time = time.perf_counter()

        fallback_used = False
        decision: ScalingDecision

        # --------------------------------------------------------------
        # Step 0 — Operator controls
        # --------------------------------------------------------------

        kill_state = await self._kill_switch.get_state()
        baseline_forced = await self._manual_override.is_baseline_forced()

        if kill_state.global_killed or baseline_forced:
            decision = await self._baseline.decide(
                state,
                explore=False,
            )

            fallback_used = True

            logger.warning(
                "decision_operator_override",
                global_killed=kill_state.global_killed,
                baseline_forced=baseline_forced,
                trace_id=str(trace_id),
            )

        else:
            # ----------------------------------------------------------
            # Step 1 — Exploration guard
            # ----------------------------------------------------------

            explore_allowed = self._guard.check_and_log(
                state,
                self._policy_stats,
            )

            explore = (
                explore_allowed
                and kill_state.allow_exploration()
            )

            # ----------------------------------------------------------
            # Step 2 — Active policy
            # ----------------------------------------------------------

            try:
                if not kill_state.is_policy_active(
                    self._active_policy.policy_type
                ):
                    logger.warning(
                        "policy_killed_fallback",
                        policy=self._active_policy.policy_type.value,
                        trace_id=str(trace_id),
                    )

                    decision = await self._baseline.decide(
                        state,
                        explore=False,
                    )

                    fallback_used = True

                else:
                    decision = await self._active_policy.decide(
                        state,
                        explore=explore,
                    )

            except Exception as exc:
                logger.error(
                    "policy_exception_fallback",
                    policy=self._active_policy.policy_type.value,
                    error=str(exc),
                    trace_id=str(trace_id),
                )

                decision = await self._baseline.decide(
                    state,
                    explore=False,
                )

                fallback_used = True

                prom.fallback_total.inc()

        # --------------------------------------------------------------
        # Step 3 — Safety clipping
        # --------------------------------------------------------------

        decision.instances_after = max(
            self._min_instances,
            min(
                decision.instances_after,
                self._max_instances,
            ),
        )

        actual_delta = (
            decision.instances_after
            - decision.instances_before
        )

        if actual_delta != self._action_delta(decision.action):
            decision.action = self._delta_to_action(actual_delta)

        # --------------------------------------------------------------
        # Step 4 — Audit logging
        # --------------------------------------------------------------

        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        import uuid as _uuid

        _pv_id = None

        if self._active_policy_version_id and not fallback_used:
            try:
                _pv_id = _uuid.UUID(
                    self._active_policy_version_id
                )
            except (ValueError, AttributeError):
                _pv_id = None

        decision_log = DecisionLog(
            trace_id=trace_id,
            policy_type=decision.policy_type.value,
            policy_version_id=_pv_id,
            state_snapshot=state.to_snapshot_dict(),
            action=decision.action.value,
            q_values=decision.q_values,
            confidence_spread=None,
            reward=None,
            latency_ms=latency_ms,
            fallback_flag=fallback_used,
            shadow_flag=(
                decision.policy_mode == PolicyMode.SHADOW
            ),
            drift_flag=False,
        )

        db.add(decision_log)

        # --------------------------------------------------------------
        # Step 5 — Scaling action logging
        # --------------------------------------------------------------

        if decision.policy_mode == PolicyMode.ACTIVE:
            env_state = await self._ensure_state_record(
                state,
                db,
            )

            scaling_action = ScalingAction(
                action_type=decision.action.value,
                instances_before=decision.instances_before,
                instances_after=decision.instances_after,
                policy_type=decision.policy_type.value,
                policy_mode=decision.policy_mode.value,
                state_id=env_state.id,
                success_flag=True,
            )

            db.add(scaling_action)

        # --------------------------------------------------------------
        # Step 6 — Commit transaction
        # --------------------------------------------------------------

        await db.commit()

        # IMPORTANT FIX
        await db.refresh(decision_log)

        # --------------------------------------------------------------
        # Step 7 — Metrics
        # --------------------------------------------------------------

        prom.decisions_total.labels(
            policy_type=decision.policy_type.value,
            action=decision.action.value,
            mode=decision.policy_mode.value,
        ).inc()

        prom.decision_latency_ms.labels(
            policy_type=decision.policy_type.value,
        ).observe(latency_ms)

        prom.p99_latency_gauge.labels(
            policy_type=decision.policy_type.value,
        ).set(state.p99_latency_ms)

        prom.instance_count_gauge.labels(
            policy_type=decision.policy_type.value,
        ).set(decision.instances_after)

        if state.p99_latency_ms > 500.0:
            prom.sla_violations_total.labels(
                violation_type="latency"
            ).inc()

        logger.info(
            "decision_made",
            trace_id=str(trace_id),
            decision_log_id=str(decision_log.id),
            policy=decision.policy_type.value,
            action=decision.action.value,
            instances=f"{decision.instances_before}->{decision.instances_after}",
            latency_ms=latency_ms,
            fallback=fallback_used,
        )

        # --------------------------------------------------------------
        # Final API response
        # --------------------------------------------------------------

        return DecisionResponse(
            decision_log_id=decision_log.id,
            trace_id=trace_id,
            action=decision.action,
            instances_before=decision.instances_before,
            instances_after=decision.instances_after,
            policy_type=decision.policy_type,
            policy_mode=decision.policy_mode,
            latency_ms=latency_ms,
            fallback_used=fallback_used,
        )

    async def _ensure_state_record(
        self,
        state: SystemState,
        db: AsyncSession,
    ) -> EnvironmentState:
        """
        Persist state snapshot for FK linkage.
        """

        env_state = EnvironmentState(
            cpu_utilization=state.cpu_utilization,
            request_rate=state.request_rate,
            p99_latency_ms=state.p99_latency_ms,
            instance_count=state.instance_count,
            hour_of_day=state.hour_of_day,
            day_of_week=state.day_of_week,
            traffic_regime=state.traffic_regime.value,
        )

        db.add(env_state)

        await db.flush()

        return env_state

    @staticmethod
    def _action_delta(action: ActionType) -> int:
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
        if delta >= 3:
            return ActionType.SCALE_UP_3
        elif delta >= 1:
            return ActionType.SCALE_UP_1
        elif delta <= -3:
            return ActionType.SCALE_DOWN_3
        elif delta <= -1:
            return ActionType.SCALE_DOWN_1

        return ActionType.HOLD