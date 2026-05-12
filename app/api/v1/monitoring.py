"""
Monitoring dashboard endpoint — GET /api/v1/monitoring/dashboard.

WHY THIS EXISTS:
- Operations team needs ONE endpoint to answer: "Is the system healthy right now?"
- Aggregates active policy, recent decisions, SLA status, drift status.
- Powers the decision-maker dashboard without requiring Prometheus access.
- Also useful as a structured health probe for automated runbooks.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.dependencies.db import get_db
from app.models.decision_log import DecisionLog
from app.models.drift_event import DriftEvent
from app.models.policy_version import PolicyVersion
from app.models.reward_log import RewardLog
from app.schemas.common import PolicyStatus

router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])
logger = structlog.get_logger(__name__)


@router.get("/dashboard")
async def get_dashboard(
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Aggregate system health dashboard — single endpoint for operations."""
    now = datetime.now(UTC)
    one_hour_ago = now - timedelta(hours=1)

    dashboard: dict[str, Any] = {
        "timestamp": now.isoformat(),
        "system_status": "healthy",
        "app_name": settings.app_name,
        "environment": settings.app_env,
    }

    # Active Policy
    try:
        result = await db.execute(
            select(PolicyVersion)
            .where(PolicyVersion.status == PolicyStatus.ACTIVE.value)
            .order_by(PolicyVersion.promoted_at.desc())
            .limit(1)
        )
        active = result.scalar_one_or_none()
        dashboard["active_policy"] = (
            {
                "policy_type": active.policy_type,
                "version": active.version,
                "algorithm": active.algorithm,
                "promoted_at": active.promoted_at.isoformat() if active.promoted_at else None,
                "eval_reward_mean": active.eval_reward_mean,
            }
            if active
            else {"policy_type": "BASELINE", "version": None, "algorithm": "threshold_v1"}
        )
    except Exception as exc:
        dashboard["active_policy"] = {"error": str(exc)}

    # Decision Statistics
    try:
        count_result = await db.execute(
            select(func.count(DecisionLog.id)).where(DecisionLog.created_at >= one_hour_ago)
        )
        decision_count_1h = count_result.scalar() or 0

        fallback_result = await db.execute(
            select(func.count(DecisionLog.id)).where(
                DecisionLog.created_at >= one_hour_ago,
                DecisionLog.fallback_flag.is_(True),
            )
        )
        fallback_count = fallback_result.scalar() or 0

        latency_result = await db.execute(
            select(func.avg(DecisionLog.latency_ms)).where(DecisionLog.created_at >= one_hour_ago)
        )
        avg_latency = latency_result.scalar()

        recent_result = await db.execute(
            select(DecisionLog).order_by(DecisionLog.created_at.desc()).limit(5)
        )
        recent = recent_result.scalars().all()

        dashboard["decisions_1h"] = {
            "total": decision_count_1h,
            "fallback_count": fallback_count,
            "fallback_rate": round(fallback_count / max(1, decision_count_1h), 4),
            "avg_latency_ms": round(float(avg_latency), 2) if avg_latency else None,
        }
        dashboard["recent_decisions"] = [
            {
                "trace_id": str(d.trace_id),
                "policy_type": d.policy_type,
                "action": d.action,
                "latency_ms": d.latency_ms,
                "fallback": d.fallback_flag,
                "shadow": d.shadow_flag,
                "created_at": d.created_at.isoformat(),
            }
            for d in recent
        ]
    except Exception as exc:
        dashboard["decisions_1h"] = {"error": str(exc)}
        dashboard["recent_decisions"] = []

    # SLO Status
    try:
        breach_result = await db.execute(
            select(func.count(DecisionLog.id)).where(
                DecisionLog.created_at >= one_hour_ago,
                DecisionLog.latency_ms > 300,
            )
        )
        breaches = breach_result.scalar() or 0
        total = dashboard.get("decisions_1h", {}).get("total", 1)
        slo_breach_rate = breaches / max(1, total)

        reward_result = await db.execute(
            select(func.avg(RewardLog.reward)).where(RewardLog.created_at >= one_hour_ago)
        )
        avg_reward = reward_result.scalar()

        dashboard["slo_status"] = {
            "p99_latency_breaches_1h": breaches,
            "slo_breach_rate_1h": round(slo_breach_rate, 4),
            "slo_target_latency_ms": 300,
            "avg_reward_1h": round(float(avg_reward), 4) if avg_reward else None,
            "slo_healthy": slo_breach_rate < 0.05,
        }
        if slo_breach_rate >= 0.05:
            dashboard["system_status"] = "degraded"
    except Exception as exc:
        dashboard["slo_status"] = {"error": str(exc)}

    # Drift Status
    try:
        recent_drift_result = await db.execute(
            select(DriftEvent).order_by(DriftEvent.triggered_at.desc()).limit(1)
        )
        recent_drift = recent_drift_result.scalar_one_or_none()

        drift_24h_result = await db.execute(
            select(func.count(DriftEvent.id)).where(
                DriftEvent.triggered_at >= now - timedelta(hours=24)
            )
        )
        drift_count_24h = drift_24h_result.scalar() or 0

        dashboard["drift_status"] = {
            "drift_events_24h": drift_count_24h,
            "last_drift_event": {
                "triggered_at": recent_drift.triggered_at.isoformat(),
                "drift_signal": recent_drift.drift_signal,
                "psi_score": recent_drift.psi_score,
                "reward_delta": recent_drift.reward_delta,
                "policy_from": recent_drift.policy_from,
                "policy_to": recent_drift.policy_to,
            }
            if recent_drift
            else None,
        }
        if drift_count_24h > 0:
            dashboard["system_status"] = "post_rollback"
    except Exception as exc:
        dashboard["drift_status"] = {"error": str(exc)}

    logger.info(
        "dashboard_served",
        system_status=dashboard["system_status"],
        decision_count_1h=dashboard.get("decisions_1h", {}).get("total", 0),
    )
    return dashboard


@router.get("/health/slo")
async def get_slo_health(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Lightweight SLO pass/fail for automated alerting and runbooks."""
    now = datetime.now(UTC)
    one_hour_ago = now - timedelta(hours=1)
    results: dict[str, Any] = {"timestamp": now.isoformat(), "period": "1h"}

    try:
        total_result = await db.execute(
            select(func.count(DecisionLog.id)).where(DecisionLog.created_at >= one_hour_ago)
        )
        total = total_result.scalar() or 0

        breach_result = await db.execute(
            select(func.count(DecisionLog.id)).where(
                DecisionLog.created_at >= one_hour_ago,
                DecisionLog.latency_ms > 300,
            )
        )
        breaches = breach_result.scalar() or 0

        fallback_result = await db.execute(
            select(func.count(DecisionLog.id)).where(
                DecisionLog.created_at >= one_hour_ago,
                DecisionLog.fallback_flag.is_(True),
            )
        )
        fallbacks = fallback_result.scalar() or 0

        breach_rate = breaches / max(1, total)
        fallback_rate = fallbacks / max(1, total)

        results["decision_latency_slo"] = {
            "target": "< 5% decisions > 300ms",
            "breach_rate": round(breach_rate, 4),
            "passing": breach_rate < 0.05,
            "total": total,
            "breaches": breaches,
        }
        results["fallback_rate_slo"] = {
            "target": "< 1% fallback rate",
            "fallback_rate": round(fallback_rate, 4),
            "passing": fallback_rate < 0.01,
            "fallbacks": fallbacks,
        }
        results["overall_passing"] = all(
            v.get("passing", False)
            for v in results.values()
            if isinstance(v, dict) and "passing" in v
        )
    except Exception as exc:
        results["error"] = str(exc)
        results["overall_passing"] = False

    return results


@router.get("/runbook/drift_response")
async def get_drift_runbook() -> dict[str, Any]:
    """Structured DRIFT_DETECTED runbook for PagerDuty/OpsGenie integration."""
    return {
        "runbook": "DRIFT_DETECTED",
        "severity": "P1",
        "steps": [
            {
                "step": 1,
                "action": "Verify rollback completed",
                "command": "GET /api/v1/monitoring/dashboard",
                "expected": "active_policy.policy_type == BASELINE",
            },
            {
                "step": 2,
                "action": "Check drift signal type",
                "command": "GET /api/v1/monitoring/dashboard",
                "expected": "drift_status.last_drift_event.drift_signal present",
            },
            {
                "step": 3,
                "action": "Verify retraining task enqueued",
                "command": "celery -A app.worker.celery_app inspect active",
                "expected": "rtde.train_rl_policy in active tasks",
            },
            {
                "step": 4,
                "action": "Confirm SLOs recover to healthy",
                "command": "GET /api/v1/monitoring/health/slo",
                "expected": "overall_passing == true",
                "wait": "5 minutes",
            },
            {
                "step": 5,
                "action": "Promote retrained shadow policy when eval_seeds >= 5",
                "command": "POST /api/v1/policies/{version_id}/promote",
            },
        ],
        "escalation": "If unresolved after 15 minutes, escalate to on-call ML engineer.",
    }


@router.get("/slo")
async def get_slo_status(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """SLO pass/fail — alias for /health/slo (test-compatible path)."""
    return await get_slo_health(db=db)


@router.get("/policy-comparison")
async def get_policy_comparison(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Compare active policy performance vs baseline over last 24h."""
    now = datetime.now(UTC)
    window_start = now - timedelta(hours=24)

    result: dict[str, Any] = {"timestamp": now.isoformat(), "window": "24h"}

    try:
        rows = await db.execute(
            select(
                DecisionLog.policy_type,
                func.count(DecisionLog.id).label("count"),
                func.avg(DecisionLog.latency_ms).label("avg_latency_ms"),
                func.sum(func.cast(DecisionLog.fallback_flag, type_=None)).label("fallbacks"),
            )
            .where(DecisionLog.created_at >= window_start)
            .group_by(DecisionLog.policy_type)
        )
        by_policy = []
        for row in rows:
            by_policy.append(
                {
                    "policy_type": row.policy_type,
                    "decision_count": row.count,
                    "avg_latency_ms": round(float(row.avg_latency_ms or 0), 2),
                    "fallback_count": int(row.fallbacks or 0),
                    "fallback_rate": round(int(row.fallbacks or 0) / max(1, row.count), 4),
                }
            )
        result["by_policy"] = by_policy

        # Reward comparison
        reward_rows = await db.execute(
            select(
                RewardLog.policy_type,
                func.avg(RewardLog.reward).label("avg_reward"),
                func.count(RewardLog.id).label("count"),
            )
            .where(RewardLog.created_at >= window_start)
            .group_by(RewardLog.policy_type)
        )
        result["rewards_by_policy"] = [
            {
                "policy_type": row.policy_type,
                "avg_reward": round(float(row.avg_reward or 0), 4),
                "sample_count": row.count,
            }
            for row in reward_rows
        ]
    except Exception as exc:
        result["error"] = str(exc)

    return result
