"""
Structured logging event schemas for RTDE.

WHY THIS EXISTS:
- Every domain event has a defined JSON schema with required fields.
- Caller passes typed objects, not raw strings — typos are caught at import time.
- ELK/Datadog/CloudWatch can index JSON fields, enabling:
    "find all decisions where latency_ms > 300" in one query.
  Plain text logging cannot do this.

EVENT CATEGORIES:
  DecisionEvent     — every scaling decision
  RewardEvent       — every reward computation
  DriftEvent        — every drift detection evaluation
  RollbackEvent     — every policy rollback
  CheckpointEvent   — checkpoint save/load
  PolicyLifecycle   — version created, promoted, retired
  ExplorationEvent  — guard suppression / permission

WHAT BREAKS IF WRONG:
- No trace_id in logs: can't correlate request → decision → reward → drift.
- Wrong field types: Datadog parsing fails, data goes to unstructured bucket.
- Missing fields: dashboard queries return no results.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

logger = structlog.get_logger("rtde.events")


@dataclass
class DecisionEvent:
    """Structured log for every scaling decision."""

    trace_id: str
    policy_type: str
    action: str
    instances_before: int
    instances_after: int
    latency_ms: float
    fallback_used: bool
    shadow_decision: bool
    confidence: float | None = None
    q_value_spread: float | None = None
    explore: bool = True

    def emit(self) -> None:
        logger.info(
            "decision_made",
            trace_id=self.trace_id,
            policy_type=self.policy_type,
            action=self.action,
            instances_before=self.instances_before,
            instances_after=self.instances_after,
            latency_ms=round(self.latency_ms, 2),
            fallback_used=self.fallback_used,
            shadow_decision=self.shadow_decision,
            confidence=round(self.confidence, 4) if self.confidence is not None else None,
            q_value_spread=round(self.q_value_spread, 4)
            if self.q_value_spread is not None
            else None,
            explore=self.explore,
        )


@dataclass
class RewardEvent:
    """Structured log for reward computation."""

    decision_log_id: str
    policy_type: str
    reward: float
    n_step_reward: float
    sla_violated: bool
    latency_penalty: float
    cost_penalty: float
    sla_penalty: float
    instability_penalty: float
    cumulative_reward: float

    def emit(self) -> None:
        logger.info(
            "reward_computed",
            decision_log_id=self.decision_log_id,
            policy_type=self.policy_type,
            reward=round(self.reward, 4),
            n_step_reward=round(self.n_step_reward, 4),
            sla_violated=self.sla_violated,
            latency_penalty=round(self.latency_penalty, 4),
            cost_penalty=round(self.cost_penalty, 4),
            sla_penalty=round(self.sla_penalty, 4),
            instability_penalty=round(self.instability_penalty, 4),
            cumulative_reward=round(self.cumulative_reward, 2),
        )


@dataclass
class DriftEvaluationEvent:
    """Structured log for each drift evaluation run."""

    policy_type: str
    drift_detected: bool
    drift_signal: str | None
    psi_score: float | None
    reward_delta: float | None
    p_value: float | None
    consecutive_degraded_windows: int
    reference_reward_mean: float | None
    current_reward_mean: float | None
    observation_count: int

    def emit(self) -> None:
        level = "critical" if self.drift_detected else "info"
        event = "drift_detected" if self.drift_detected else "drift_evaluation_clean"
        getattr(logger, level)(
            event,
            policy_type=self.policy_type,
            drift_signal=self.drift_signal,
            psi_score=round(self.psi_score, 4) if self.psi_score is not None else None,
            reward_delta=round(self.reward_delta, 4) if self.reward_delta is not None else None,
            consecutive_degraded_windows=self.consecutive_degraded_windows,
            reference_reward_mean=round(self.reference_reward_mean, 4)
            if self.reference_reward_mean is not None
            else None,
            current_reward_mean=round(self.current_reward_mean, 4)
            if self.current_reward_mean is not None
            else None,
            observation_count=self.observation_count,
        )


@dataclass
class RollbackEvent:
    """Structured log for policy rollback execution."""

    policy_from: str
    policy_to: str
    drift_signal: str
    psi_score: float | None
    reward_delta: float | None
    consecutive_windows: int
    rollback_latency_ms: float
    retraining_job_id: str
    success: bool
    error: str | None = None

    def emit(self) -> None:
        level = "critical" if self.success else "error"
        getattr(logger, level)(
            "policy_rollback",
            policy_from=self.policy_from,
            policy_to=self.policy_to,
            drift_signal=self.drift_signal,
            psi_score=round(self.psi_score, 4) if self.psi_score is not None else None,
            reward_delta=round(self.reward_delta, 4) if self.reward_delta is not None else None,
            consecutive_windows=self.consecutive_windows,
            rollback_latency_ms=round(self.rollback_latency_ms, 2),
            retraining_job_id=self.retraining_job_id,
            success=self.success,
            error=self.error,
        )


@dataclass
class CheckpointEvent:
    """Structured log for checkpoint save/load."""

    policy_type: str
    operation: str  # save | load
    step_count: int
    success: bool
    path: str | None = None
    error: str | None = None

    def emit(self) -> None:
        level = "info" if self.success else "error"
        getattr(logger, level)(
            "checkpoint_operation",
            policy_type=self.policy_type,
            operation=self.operation,
            step_count=self.step_count,
            success=self.success,
            path=self.path,
            error=self.error,
        )


@dataclass
class PolicyLifecycleEvent:
    """Structured log for policy version state changes."""

    policy_type: str
    version_id: str
    version_number: int
    transition: str  # created | shadow | promoted | retired
    eval_reward_mean: float | None = None
    eval_seeds: int | None = None
    reason: str | None = None

    def emit(self) -> None:
        logger.info(
            "policy_lifecycle",
            policy_type=self.policy_type,
            version_id=self.version_id,
            version_number=self.version_number,
            transition=self.transition,
            eval_reward_mean=round(self.eval_reward_mean, 4)
            if self.eval_reward_mean is not None
            else None,
            eval_seeds=self.eval_seeds,
            reason=self.reason,
        )


@dataclass
class ExplorationEvent:
    """Structured log for ExplorationGuard decisions."""

    policy_type: str
    explore_allowed: bool
    suppression_reason: str | None
    p99_latency_ms: float
    request_rate: float
    sla_violation_rate: float
    consecutive_violations: int

    def emit(self) -> None:
        level = "warning" if not self.explore_allowed else "debug"
        getattr(logger, level)(
            "exploration_decision",
            policy_type=self.policy_type,
            explore_allowed=self.explore_allowed,
            suppression_reason=self.suppression_reason,
            p99_latency_ms=round(self.p99_latency_ms, 2),
            request_rate=round(self.request_rate, 1),
            sla_violation_rate=round(self.sla_violation_rate, 4),
            consecutive_violations=self.consecutive_violations,
        )


@dataclass
class TrainingEvent:
    """Structured log for RL training steps."""

    policy_type: str
    task_id: str
    step: int
    loss: float | None
    buffer_size: int
    training_steps_total: int

    def emit(self) -> None:
        logger.info(
            "training_step",
            policy_type=self.policy_type,
            task_id=self.task_id,
            step=self.step,
            loss=round(self.loss, 6) if self.loss is not None else None,
            buffer_size=self.buffer_size,
            training_steps_total=self.training_steps_total,
        )
