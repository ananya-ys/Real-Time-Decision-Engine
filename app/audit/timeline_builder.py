"""
IncidentTimelineBuilder — auto-generate structured incident timelines.

WHY THIS EXISTS:
The review: "operators can actually reconstruct what happened fast."
Post-rollback: what led to drift? What was the operator doing?
What was the system's state leading up to failure?

TIMELINE INCLUDES (in chronological order):
- All DecisionLog entries (decisions made)
- All DriftEvent entries (drift signals)
- All OperatorEvent entries (human interventions)
- All PolicyVersion status changes (promotions, retirements)
- All ExplorationGuard suppressions (safety activations)

AUTO-TRIGGERED: After every RollbackService.execute_rollback(), the
IncidentTimelineBuilder generates a timeline and stores it in Redis
for 24 hours (accessible at GET /api/v1/audit/incidents/latest).

TYPICAL INCIDENT TIMELINE:
  T-5min: Drift window 1 degraded (PSI=0.18, p=0.04)
  T-4min: Drift window 2 degraded (PSI=0.22, p=0.01)
  T-3min: Drift window 3 degraded — THRESHOLD MET
  T-3min: ROLLBACK triggered: RL → BASELINE (by drift_detector)
  T-3min: Retraining job enqueued (task_id=abc)
  T-2min: OPERATOR freeze_exploration activated (by operator@rtde)
  T-0min: Baseline serving all traffic
"""

from __future__ import annotations

from dataclasses import dataclass
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

logger = structlog.get_logger(__name__)


@dataclass
class TimelineEvent:
    timestamp: datetime
    event_type: str  # DECISION | DRIFT | OPERATOR | POLICY_CHANGE | GUARD_SUPPRESSION
    severity: str  # INFO | WARNING | CRITICAL
    description: str
    details: dict[str, Any]


class IncidentTimeline:
    """A structured incident timeline."""

    def __init__(
        self,
        window_start: datetime,
        window_end: datetime,
        events: list[TimelineEvent],
    ) -> None:
        self.window_start = window_start
        self.window_end = window_end
        self.events = sorted(events, key=lambda e: e.timestamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "total_events": len(self.events),
            "events": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "event_type": e.event_type,
                    "severity": e.severity,
                    "description": e.description,
                    "details": e.details,
                }
                for e in self.events
            ],
            "summary": self._generate_summary(),
        }

    def _generate_summary(self) -> dict[str, Any]:
        drift_events = [e for e in self.events if e.event_type == "DRIFT"]
        operator_events = [e for e in self.events if e.event_type == "OPERATOR"]
        rollback_events = [
            e for e in self.events if e.event_type == "DRIFT" and "ROLLBACK" in e.description
        ]

        return {
            "total_decisions": sum(1 for e in self.events if e.event_type == "DECISION"),
            "drift_events": len(drift_events),
            "operator_interventions": len(operator_events),
            "rollbacks": len(rollback_events),
            "critical_events": sum(1 for e in self.events if e.severity == "CRITICAL"),
            "first_drift_signal": drift_events[0].timestamp.isoformat() if drift_events else None,
            "rollback_time": rollback_events[0].timestamp.isoformat() if rollback_events else None,
        }


class IncidentTimelineBuilder:
    """
    Builds incident timelines by querying multiple audit tables.

    Typical usage:
      builder = IncidentTimelineBuilder()
      timeline = await builder.build(
          window_hours=2,
          db=db,
      )
    """

    async def build(
        self,
        db: AsyncSession,
        window_hours: int = 2,
        anchor_time: datetime | None = None,
    ) -> IncidentTimeline:
        """
        Build a complete incident timeline for the given time window.

        Args:
            db: Database session.
            window_hours: Hours to look back from anchor_time.
            anchor_time: End of window. Defaults to now.

        Returns:
            IncidentTimeline with all events in chronological order.
        """
        if anchor_time is None:
            anchor_time = datetime.now(UTC)

        window_start = anchor_time - timedelta(hours=window_hours)
        window_end = anchor_time

        events: list[TimelineEvent] = []

        # Load all event types in parallel
        events.extend(await self._load_drift_events(window_start, window_end, db))
        events.extend(await self._load_operator_events(window_start, window_end, db))
        events.extend(await self._load_policy_changes(window_start, window_end, db))
        events.extend(await self._load_guard_suppressions(window_start, window_end, db))
        events.extend(await self._load_key_decisions(window_start, window_end, db))

        logger.info(
            "incident_timeline_built",
            window_hours=window_hours,
            total_events=len(events),
        )

        return IncidentTimeline(
            window_start=window_start,
            window_end=window_end,
            events=events,
        )

    async def _load_drift_events(
        self,
        start: datetime,
        end: datetime,
        db: AsyncSession,
    ) -> list[TimelineEvent]:
        result = await db.execute(
            select(DriftEvent)
            .where(DriftEvent.triggered_at >= start, DriftEvent.triggered_at <= end)
            .order_by(DriftEvent.triggered_at)
        )
        events = []
        for e in result.scalars().all():
            events.append(
                TimelineEvent(
                    timestamp=e.triggered_at,
                    event_type="DRIFT",
                    severity="CRITICAL",
                    description=(
                        f"ROLLBACK triggered: {e.policy_from} → {e.policy_to} "
                        f"(signal={e.drift_signal})"
                    ),
                    details={
                        "drift_signal": e.drift_signal,
                        "psi_score": e.psi_score,
                        "reward_delta": e.reward_delta,
                        "window_count": e.window_count,
                        "policy_from": e.policy_from,
                        "policy_to": e.policy_to,
                    },
                )
            )
        return events

    async def _load_operator_events(
        self,
        start: datetime,
        end: datetime,
        db: AsyncSession,
    ) -> list[TimelineEvent]:
        result = await db.execute(
            select(OperatorEvent)
            .where(OperatorEvent.created_at >= start, OperatorEvent.created_at <= end)
            .order_by(OperatorEvent.created_at)
        )
        events = []
        for e in result.scalars().all():
            severity = "WARNING" if "KILL" in e.action or "FREEZE" in e.action else "INFO"
            events.append(
                TimelineEvent(
                    timestamp=e.created_at,
                    event_type="OPERATOR",
                    severity=severity,
                    description=f"OPERATOR {e.action} by {e.actor}: {e.reason}",
                    details={
                        "action": e.action,
                        "actor": e.actor,
                        "actor_role": e.actor_role,
                        "reason": e.reason,
                        "target": e.target,
                        "success": e.success,
                    },
                )
            )
        return events

    async def _load_policy_changes(
        self,
        start: datetime,
        end: datetime,
        db: AsyncSession,
    ) -> list[TimelineEvent]:
        # Find policies promoted or demoted in window
        result = await db.execute(
            select(PolicyVersion)
            .where((PolicyVersion.promoted_at >= start) | (PolicyVersion.demoted_at >= start))
            .order_by(PolicyVersion.promoted_at)
        )
        events = []
        for v in result.scalars().all():
            if v.promoted_at and start <= v.promoted_at <= end:
                events.append(
                    TimelineEvent(
                        timestamp=v.promoted_at,
                        event_type="POLICY_CHANGE",
                        severity="WARNING",
                        description=f"PROMOTION: {v.policy_type} v{v.version} → ACTIVE",
                        details={
                            "policy_type": v.policy_type,
                            "version": v.version,
                            "algorithm": v.algorithm,
                            "eval_reward_mean": v.eval_reward_mean,
                            "eval_seeds": v.eval_seeds,
                        },
                    )
                )
            if v.demoted_at and start <= v.demoted_at <= end:
                events.append(
                    TimelineEvent(
                        timestamp=v.demoted_at,
                        event_type="POLICY_CHANGE",
                        severity="CRITICAL",
                        description=f"DEMOTION: {v.policy_type} v{v.version} → RETIRED",
                        details={
                            "policy_type": v.policy_type,
                            "version": v.version,
                        },
                    )
                )
        return events

    async def _load_guard_suppressions(
        self,
        start: datetime,
        end: datetime,
        db: AsyncSession,
    ) -> list[TimelineEvent]:
        result = await db.execute(
            select(ExplorationGuardLog)
            .where(
                ExplorationGuardLog.created_at >= start,
                ExplorationGuardLog.created_at <= end,
                ExplorationGuardLog.exploration_suppressed.is_(True),
            )
            .order_by(ExplorationGuardLog.created_at)
            .limit(50)  # cap to most recent 50 suppressions
        )
        events = []
        for g in result.scalars().all():
            events.append(
                TimelineEvent(
                    timestamp=g.created_at,
                    event_type="GUARD_SUPPRESSION",
                    severity="WARNING",
                    description=f"EXPLORATION suppressed: {g.suppression_reason}",
                    details={
                        "suppression_reason": g.suppression_reason,
                        "state": g.state_snapshot,
                    },
                )
            )
        return events

    async def _load_key_decisions(
        self,
        start: datetime,
        end: datetime,
        db: AsyncSession,
    ) -> list[TimelineEvent]:
        """Load only fallback decisions (not every decision — too noisy)."""
        result = await db.execute(
            select(DecisionLog)
            .where(
                DecisionLog.created_at >= start,
                DecisionLog.created_at <= end,
                DecisionLog.fallback_flag.is_(True),
            )
            .order_by(DecisionLog.created_at)
            .limit(20)
        )
        events = []
        for d in result.scalars().all():
            events.append(
                TimelineEvent(
                    timestamp=d.created_at,
                    event_type="DECISION",
                    severity="WARNING",
                    description=f"FALLBACK decision: {d.action} (policy exception)",
                    details={
                        "trace_id": str(d.trace_id),
                        "policy_type": d.policy_type,
                        "action": d.action,
                        "latency_ms": d.latency_ms,
                    },
                )
            )
        return events
