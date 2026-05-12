"""
DecisionReplayEngine — reconstruct exact system state at any past decision.

WHY THIS EXISTS:
The review: "operators can actually reconstruct what happened fast."
Without this, debugging an incident means:
- Dig through structlog output
- Correlate trace_id across 6 different tables
- Manually reconstruct what the policy saw
- Take 30-90 minutes per incident

With replay:
- GET /api/v1/audit/decisions/{id}/replay
- Get the exact state, exact Q-values, exact policy mode, exact reward outcome
- In under 500ms.

WHAT REPLAY RECONSTRUCTS:
1. Original SystemState (from DecisionLog.state_snapshot)
2. Policy that ran (from DecisionLog.policy_type + policy_version_id)
3. Q-values and confidence (from DecisionLog.q_values + confidence_spread)
4. ExplorationGuard decision at that moment (from ExplorationGuardLog)
5. Reward outcome (from RewardLog)
6. Whether drift was active (from DriftEvent table)
7. Operator overrides at that time (from OperatorEvent)

WHAT REPLAY DOES NOT DO:
- Re-run the policy with current weights (that's backtesting)
- Re-execute the action (that's counterfactual simulation)
- Modify any existing records
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.decision_log import DecisionLog
from app.models.drift_event import DriftEvent
from app.models.exploration_guard_log import ExplorationGuardLog
from app.models.operator_event import OperatorEvent
from app.models.policy_version import PolicyVersion
from app.models.reward_log import RewardLog

logger = structlog.get_logger(__name__)


class ReplayFrame:
    """Complete reconstruction of one decision's context."""

    def __init__(
        self,
        decision: dict[str, Any],
        policy_context: dict[str, Any],
        exploration_context: dict[str, Any],
        reward_outcome: dict[str, Any] | None,
        drift_context: dict[str, Any],
        operator_context: list[dict[str, Any]],
    ) -> None:
        self.decision = decision
        self.policy_context = policy_context
        self.exploration_context = exploration_context
        self.reward_outcome = reward_outcome
        self.drift_context = drift_context
        self.operator_context = operator_context

    def to_dict(self) -> dict[str, Any]:
        return {
            "replay_generated_at": datetime.now(UTC).isoformat(),
            "decision": self.decision,
            "policy_context": self.policy_context,
            "exploration_context": self.exploration_context,
            "reward_outcome": self.reward_outcome,
            "drift_context": self.drift_context,
            "operator_context": self.operator_context,
            "summary": self._generate_summary(),
        }

    def _generate_summary(self) -> str:
        """Human-readable one-paragraph summary of what happened."""
        policy = self.decision.get("policy_type", "UNKNOWN")
        action = self.decision.get("action", "UNKNOWN")
        fallback = self.decision.get("fallback_flag", False)
        reward = self.reward_outcome.get("reward") if self.reward_outcome else None
        drift = self.drift_context.get("drift_active", False)
        explore = self.exploration_context.get("exploration_allowed", True)

        parts = [f"Policy {policy} chose {action}."]

        if fallback:
            parts.append("Baseline fallback was used (policy exception).")
        if not explore:
            parts.append(
                f"Exploration was suppressed by ExplorationGuard "
                f"(reason: {self.exploration_context.get('suppression_reason', 'unknown')})."
            )
        if drift:
            parts.append("Drift was being evaluated at this time.")
        if reward is not None:
            parts.append(f"Reward outcome: {reward:.4f}.")
        if self.operator_context:
            parts.append(
                f"{len(self.operator_context)} operator action(s) were active around this time."
            )

        return " ".join(parts)


class DecisionReplayEngine:
    """
    Reconstructs complete decision context from stored audit records.

    All data comes from DB — no re-execution of any model code.
    """

    async def replay(
        self,
        decision_log_id: uuid.UUID,
        db: AsyncSession,
    ) -> ReplayFrame:
        """
        Reconstruct everything about a past decision.

        Args:
            decision_log_id: UUID of the DecisionLog to replay.
            db: Active DB session.

        Returns:
            ReplayFrame with complete reconstruction.

        Raises:
            ValueError: If decision not found.
        """
        # Load the decision log
        log_result = await db.execute(select(DecisionLog).where(DecisionLog.id == decision_log_id))
        log = log_result.scalar_one_or_none()
        if log is None:
            raise ValueError(f"Decision {decision_log_id} not found")

        decision_dict = {
            "id": str(log.id),
            "trace_id": str(log.trace_id),
            "created_at": log.created_at.isoformat(),
            "policy_type": log.policy_type,
            "action": log.action,
            "state_snapshot": log.state_snapshot,
            "q_values": log.q_values,
            "confidence_spread": log.confidence_spread,
            "fallback_flag": log.fallback_flag,
            "shadow_flag": log.shadow_flag,
            "drift_flag": log.drift_flag,
            "latency_ms": log.latency_ms,
        }

        # Load policy version context
        policy_ctx = await self._load_policy_context(log, db)

        # Load exploration guard context
        explore_ctx = await self._load_exploration_context(log, db)

        # Load reward outcome
        reward = await self._load_reward(log, db)

        # Load drift context around decision time
        drift_ctx = await self._load_drift_context(log.created_at, db)

        # Load operator events around decision time
        operator_ctx = await self._load_operator_events(log.created_at, db)

        logger.info(
            "decision_replayed",
            decision_id=str(decision_log_id),
            policy_type=log.policy_type,
            action=log.action,
        )

        return ReplayFrame(
            decision=decision_dict,
            policy_context=policy_ctx,
            exploration_context=explore_ctx,
            reward_outcome=reward,
            drift_context=drift_ctx,
            operator_context=operator_ctx,
        )

    async def _load_policy_context(self, log: DecisionLog, db: AsyncSession) -> dict[str, Any]:
        """Load policy version info for this decision."""
        if log.policy_version_id is None:
            return {"policy_type": log.policy_type, "version_info": None}

        result = await db.execute(
            select(PolicyVersion).where(PolicyVersion.id == log.policy_version_id)
        )
        version = result.scalar_one_or_none()

        if version is None:
            return {"policy_type": log.policy_type, "version_info": "not_found"}

        return {
            "policy_type": log.policy_type,
            "version_id": str(version.id),
            "version_number": version.version,
            "algorithm": version.algorithm,
            "status_at_decision": version.status,
            "eval_reward_mean": version.eval_reward_mean,
        }

    async def _load_exploration_context(self, log: DecisionLog, db: AsyncSession) -> dict[str, Any]:
        """Load ExplorationGuard decision for this specific decision."""
        result = await db.execute(
            select(ExplorationGuardLog)
            .where(ExplorationGuardLog.decision_log_id == log.id)
            .limit(1)
        )
        guard_log = result.scalar_one_or_none()

        if guard_log is None:
            return {"exploration_allowed": True, "suppression_reason": None}

        return {
            "exploration_allowed": not guard_log.exploration_suppressed,
            "suppression_reason": guard_log.suppression_reason
            if guard_log.exploration_suppressed
            else None,
            "state_at_suppression": guard_log.state_snapshot,
        }

    async def _load_reward(self, log: DecisionLog, db: AsyncSession) -> dict[str, Any] | None:
        """Load reward outcome for this decision."""
        result = await db.execute(
            select(RewardLog).where(RewardLog.decision_log_id == log.id).limit(1)
        )
        reward = result.scalar_one_or_none()

        if reward is None:
            return None

        return {
            "reward": reward.reward,
            "n_step_reward": reward.n_step_reward,
            "cumulative_reward": reward.cumulative_reward,
            "cumulative_regret": reward.cumulative_regret,
            "baseline_reward": reward.baseline_reward,
        }

    async def _load_drift_context(
        self, decision_time: datetime, db: AsyncSession
    ) -> dict[str, Any]:
        """Load drift events within 5 minutes of this decision."""
        window_start = decision_time - timedelta(minutes=5)
        window_end = decision_time + timedelta(minutes=5)

        result = await db.execute(
            select(DriftEvent)
            .where(
                DriftEvent.triggered_at >= window_start,
                DriftEvent.triggered_at <= window_end,
            )
            .order_by(DriftEvent.triggered_at.desc())
            .limit(3)
        )
        drift_events = result.scalars().all()

        return {
            "drift_active": len(drift_events) > 0,
            "recent_drift_events": [
                {
                    "id": str(e.id),
                    "triggered_at": e.triggered_at.isoformat(),
                    "drift_signal": e.drift_signal,
                    "psi_score": e.psi_score,
                    "reward_delta": e.reward_delta,
                }
                for e in drift_events
            ],
        }

    async def _load_operator_events(
        self, decision_time: datetime, db: AsyncSession
    ) -> list[dict[str, Any]]:
        """Load operator events within 10 minutes of this decision."""
        window_start = decision_time - timedelta(minutes=10)
        window_end = decision_time + timedelta(minutes=10)

        result = await db.execute(
            select(OperatorEvent)
            .where(
                OperatorEvent.created_at >= window_start,
                OperatorEvent.created_at <= window_end,
            )
            .order_by(OperatorEvent.created_at.desc())
            .limit(10)
        )
        events = result.scalars().all()

        return [
            {
                "action": e.action,
                "actor": e.actor,
                "reason": e.reason,
                "timestamp": e.created_at.isoformat(),
                "target": e.target,
            }
            for e in events
        ]
