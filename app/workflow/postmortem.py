"""
PostmortemGenerator — auto-generated incident reports after rollback.

WHY THIS EXISTS:
The review: "auto-generated incident postmortem" as a wow feature.
After every rollback:
  1. What triggered it? (drift signal, psi_score, reward_delta)
  2. What was the impact? (SLA violations, fallback count, affected decisions)
  3. What was the root cause? (reward degradation vs input drift)
  4. What was the mitigation? (rollback to baseline + retraining triggered)
  5. What are the next steps? (retrain with fresh data, evaluate in shadow)

GENERATED WITHIN: 5 minutes of rollback completion.
STORED IN: PostgreSQL (PostmortemRecord model) + Redis cache (24h).
ACCESSIBLE AT: GET /api/v1/workflow/postmortems/{incident_id}

FORMAT: Structured JSON (for PagerDuty/OpsGenie) + Markdown text (for humans).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.decision_log import DecisionLog
from app.models.drift_event import DriftEvent
from app.models.operator_event import OperatorEvent

logger = structlog.get_logger(__name__)


class PostmortemReport:
    """A structured incident postmortem."""

    def __init__(
        self,
        incident_id: uuid.UUID,
        drift_event_id: uuid.UUID,
        generated_at: datetime,
        trigger: dict[str, Any],
        impact: dict[str, Any],
        timeline: list[dict[str, Any]],
        root_cause: str,
        mitigation: dict[str, Any],
        next_steps: list[str],
    ) -> None:
        self.incident_id = incident_id
        self.drift_event_id = drift_event_id
        self.generated_at = generated_at
        self.trigger = trigger
        self.impact = impact
        self.timeline = timeline
        self.root_cause = root_cause
        self.mitigation = mitigation
        self.next_steps = next_steps

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": str(self.incident_id),
            "drift_event_id": str(self.drift_event_id),
            "generated_at": self.generated_at.isoformat(),
            "trigger": self.trigger,
            "impact": self.impact,
            "timeline": self.timeline,
            "root_cause": self.root_cause,
            "mitigation": self.mitigation,
            "next_steps": self.next_steps,
        }

    def to_markdown(self) -> str:
        """Human-readable markdown postmortem."""
        lines = [
            f"# Incident Postmortem — {self.generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            f"**Incident ID:** `{self.incident_id}`",
            f"**Generated:** {self.generated_at.isoformat()}",
            "",
            "## Summary",
            "",
            f"Policy rollback triggered by **{self.trigger.get('drift_signal', 'UNKNOWN')} signal**. "
            f"System reverted from {self.trigger.get('policy_from', '?')} to "
            f"{self.trigger.get('policy_to', 'BASELINE')}.",
            "",
            "## Trigger",
            "",
            f"- **Drift signal:** {self.trigger.get('drift_signal', 'N/A')}",
            f"- **PSI score:** {self.trigger.get('psi_score', 'N/A')}",
            f"- **Reward delta:** {self.trigger.get('reward_delta', 'N/A')}",
            f"- **Consecutive degraded windows:** {self.trigger.get('window_count', 'N/A')}",
            "",
            "## Impact",
            "",
            f"- **SLA violations (1h window):** {self.impact.get('sla_violations_1h', 'N/A')}",
            f"- **Fallback decisions:** {self.impact.get('fallback_decisions', 'N/A')}",
            f"- **Total affected decisions:** {self.impact.get('affected_decisions', 'N/A')}",
            "",
            "## Root Cause",
            "",
            self.root_cause,
            "",
            "## Mitigation",
            "",
            f"- **Action taken:** {self.mitigation.get('action', 'Rollback to baseline')}",
            f"- **Rollback at:** {self.mitigation.get('rollback_at', 'N/A')}",
            f"- **Retraining job:** `{self.mitigation.get('retraining_job_id', 'N/A')}`",
            "",
            "## Next Steps",
            "",
        ]
        for i, step in enumerate(self.next_steps, 1):
            lines.append(f"{i}. {step}")

        return "\n".join(lines)


class PostmortemGenerator:
    """
    Generates structured incident postmortems after rollbacks.

    Called automatically by RollbackService after every rollback.
    """

    async def generate(
        self,
        drift_event_id: uuid.UUID,
        db: AsyncSession,
    ) -> PostmortemReport:
        """
        Generate complete postmortem from a DriftEvent.

        Args:
            drift_event_id: The drift event that triggered rollback.
            db: Database session.

        Returns:
            PostmortemReport with complete structured analysis.
        """
        # Load the drift event
        drift_result = await db.execute(select(DriftEvent).where(DriftEvent.id == drift_event_id))
        drift = drift_result.scalar_one_or_none()
        if drift is None:
            raise ValueError(f"DriftEvent {drift_event_id} not found")

        rollback_time = drift.triggered_at
        window_start = rollback_time - timedelta(hours=1)

        # Build trigger section
        trigger = {
            "drift_signal": drift.drift_signal,
            "psi_score": round(drift.psi_score, 4) if drift.psi_score else None,
            "reward_delta": round(drift.reward_delta, 4) if drift.reward_delta else None,
            "window_count": drift.window_count,
            "policy_from": drift.policy_from,
            "policy_to": drift.policy_to,
            "triggered_at": rollback_time.isoformat(),
        }

        # Compute impact
        impact = await self._compute_impact(window_start, rollback_time, db)

        # Build timeline
        timeline = await self._build_timeline(window_start, rollback_time, db)

        # Generate root cause analysis
        root_cause = self._analyze_root_cause(drift)

        # Mitigation
        mitigation = {
            "action": f"Rollback: {drift.policy_from} → {drift.policy_to}",
            "rollback_at": rollback_time.isoformat(),
            "retraining_job_id": str(drift.retraining_job_id) if drift.retraining_job_id else None,
        }

        # Next steps based on drift signal
        next_steps = self._generate_next_steps(drift)

        report = PostmortemReport(
            incident_id=uuid.uuid4(),
            drift_event_id=drift_event_id,
            generated_at=datetime.now(UTC),
            trigger=trigger,
            impact=impact,
            timeline=timeline,
            root_cause=root_cause,
            mitigation=mitigation,
            next_steps=next_steps,
        )

        logger.info(
            "postmortem_generated",
            incident_id=str(report.incident_id),
            drift_event_id=str(drift_event_id),
            drift_signal=drift.drift_signal,
        )

        return report

    async def _compute_impact(
        self, start: datetime, end: datetime, db: AsyncSession
    ) -> dict[str, Any]:
        """Count SLA violations and fallbacks in the incident window."""
        total_result = await db.execute(
            select(func.count(DecisionLog.id)).where(
                DecisionLog.created_at >= start,
                DecisionLog.created_at <= end,
            )
        )
        total = total_result.scalar() or 0

        fallback_result = await db.execute(
            select(func.count(DecisionLog.id)).where(
                DecisionLog.created_at >= start,
                DecisionLog.created_at <= end,
                DecisionLog.fallback_flag.is_(True),
            )
        )
        fallbacks = fallback_result.scalar() or 0

        return {
            "affected_decisions": total,
            "fallback_decisions": fallbacks,
            "fallback_rate": round(fallbacks / max(1, total), 4),
            "sla_violations_1h": "see reward_logs (not pre-computed in this version)",
            "window_hours": 1,
        }

    async def _build_timeline(
        self, start: datetime, end: datetime, db: AsyncSession
    ) -> list[dict[str, Any]]:
        """Build key event timeline for the incident window."""
        events = []

        # Operator events
        op_result = await db.execute(
            select(OperatorEvent)
            .where(OperatorEvent.created_at >= start, OperatorEvent.created_at <= end)
            .order_by(OperatorEvent.created_at)
        )
        for e in op_result.scalars().all():
            events.append(
                {
                    "timestamp": e.created_at.isoformat(),
                    "type": "OPERATOR",
                    "event": f"{e.action} by {e.actor}",
                }
            )

        return sorted(events, key=lambda e: e["timestamp"])

    def _analyze_root_cause(self, drift: DriftEvent) -> str:
        """Generate root cause analysis based on drift signal."""
        if drift.drift_signal == "INPUT_DRIFT":
            return (
                f"Input feature distribution shifted significantly (PSI={drift.psi_score:.4f}). "
                "The model was trained on a different traffic distribution than what it encountered. "
                "The Q-network's value estimates became unreliable for the new input regime. "
                "Likely cause: traffic pattern change (e.g., new client segment, time-of-day shift, "
                "sudden load spike outside training distribution)."
            )
        elif drift.drift_signal == "REWARD_DEGRADATION":
            return (
                f"Reward degradation detected without input distribution shift "
                f"(reward_delta={drift.reward_delta:.4f}). "
                "The model's performance deteriorated on familiar traffic patterns. "
                "Likely cause: model overfit to outdated patterns, or the environment "
                "changed in a way not captured by the input features alone."
            )
        elif drift.drift_signal == "BOTH":
            return (
                "Both input distribution drift AND reward degradation detected simultaneously. "
                "The traffic pattern shifted AND the model failed to adapt. "
                "This is the most severe drift scenario. Retraining with diverse recent data "
                "is required before re-promoting any ML policy."
            )
        return "Unknown drift signal. Manual investigation required."

    def _generate_next_steps(self, drift: DriftEvent) -> list[str]:
        """Generate actionable next steps based on drift type."""
        common_steps = [
            "Verify baseline SLOs are met: GET /api/v1/monitoring/health/slo",
            f"Monitor retraining job: check Celery task {drift.retraining_job_id}",
            "Run backtest on retrained model before shadow promotion",
            "Shadow-run retrained model for minimum 24h at 10% traffic (canary)",
        ]

        if drift.drift_signal == "INPUT_DRIFT":
            return [
                "Analyze PSI breakdown by feature to identify which inputs shifted most",
                "Collect fresh training data from current traffic distribution",
                "Retrain StateNormalizer on recent data before retraining policy",
            ] + common_steps
        elif drift.drift_signal == "REWARD_DEGRADATION":
            return [
                "Review reward function weights — consider recalibration",
                "Check if environment behavior changed (new SLA targets, new cost structure)",
                "Increase replay buffer diversity before retraining",
            ] + common_steps
        elif drift.drift_signal == "BOTH":
            return [
                "URGENT: Collect diverse training data covering new traffic patterns",
                "Retrain StateNormalizer AND policy from scratch",
                "Extend shadow period to 48h minimum before canary",
            ] + common_steps

        return common_steps
