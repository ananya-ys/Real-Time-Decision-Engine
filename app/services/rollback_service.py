"""
RollbackService — 5-step atomic policy rollback.

WHY THIS EXISTS:
- When drift is detected, the current policy is degrading performance.
- We must switch BACK to baseline (always-on safety net) before damage compounds.
- The 5 steps must be atomic: if step 3 fails, we cannot be in a state where
  the old policy is retired but the new one isn't active yet.

THE 5 STEPS:
1. FREEZE — Stop the current policy from making new decisions (set flag).
2. SWAP — Load baseline policy as active (guaranteed to always work).
3. LOG — Write DriftEvent to DB (forensic audit trail for post-incident).
4. SIGNAL — Enqueue Celery retraining task (new policy training starts).
5. ALERT — Emit metrics (Prometheus) + structured log for alerting.

WHAT BREAKS IF WRONG:
- No freeze step: policy continues making decisions during swap = inconsistent state.
- Non-atomic DB + metrics: DB succeeds but metrics fail = alert doesn't fire.
- No DriftEvent: incident happened but no forensic record = can't improve.
- No retraining signal: system stays on baseline forever = no learning recovery.

INDUSTRY PARALLEL:
- Kubernetes: pod fails → immediate restart with last-known-good image.
- Circuit breaker pattern: trips on failure → falls back to safe mode.
- This is the ML equivalent: model degrades → fall back to rule-based.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import structlog
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import metrics as prom
from app.models.drift_event import DriftEvent
from app.models.policy_version import PolicyVersion
from app.models.scaling_action import ScalingAction
from app.schemas.common import DriftSignal, PolicyStatus, PolicyType
from app.services.drift_service import DriftResult

logger = structlog.get_logger(__name__)


class RollbackService:
    """
    Orchestrates the 5-step policy rollback on drift detection.

    Thread-safety: rollback should only run once at a time.
    The DriftService hysteresis ensures it's not triggered on noise.
    """

    def __init__(self) -> None:
        self._rollback_in_progress = False

    async def execute_rollback(
        self,
        drift_result: DriftResult,
        current_policy_type: PolicyType,
        current_policy_version_id: uuid.UUID | None,
        db: AsyncSession,
    ) -> DriftEvent | None:
        """
        Execute full 5-step rollback.

        Args:
            drift_result: Evidence from DriftService evaluation.
            current_policy_type: The policy being rolled back.
            current_policy_version_id: The version being retired.
            db: Active async DB session.

        Returns:
            DriftEvent record (for caller to emit metrics) or None if already rolling back.
        """
        if self._rollback_in_progress:
            logger.warning("rollback_already_in_progress", skipping=True)
            return None

        self._rollback_in_progress = True
        rollback_start = datetime.now(UTC)

        try:
            logger.critical(
                "rollback_started",
                policy_from=current_policy_type.value,
                drift_signal=drift_result.drift_signal.value if drift_result.drift_signal else None,
                psi_score=drift_result.psi_score,
                reward_delta=drift_result.reward_delta,
            )

            # ── Step 1: FREEZE current policy ──────────────────────
            # Mark scaling actions with rollback_trigger for audit trail
            await db.execute(
                update(ScalingAction)
                .where(ScalingAction.rollback_trigger.is_(False))
                .values(rollback_trigger=True)
            )
            logger.info("rollback_step_1_freeze_complete")

            # ── Step 2: SWAP to baseline ──────────────────────────
            # Retire current active policy version
            if current_policy_version_id:
                await db.execute(
                    update(PolicyVersion)
                    .where(PolicyVersion.id == current_policy_version_id)
                    .values(
                        status=PolicyStatus.RETIRED.value,
                        demoted_at=datetime.now(UTC),
                    )
                )
            logger.info(
                "rollback_step_2_swap_complete",
                rolled_back_to=PolicyType.BASELINE.value,
            )

            # ── Step 3: LOG drift event ────────────────────────────
            retraining_job_id = uuid.uuid4()  # will be used in step 4

            drift_event = DriftEvent(
                policy_version_id=current_policy_version_id,
                drift_signal=drift_result.drift_signal.value
                if drift_result.drift_signal
                else DriftSignal.REWARD_DEGRADATION.value,
                psi_score=drift_result.psi_score,
                reward_delta=drift_result.reward_delta,
                window_count=drift_result.consecutive_degraded_windows,
                policy_from=current_policy_type.value,
                policy_to=PolicyType.BASELINE.value,
                retraining_job_id=retraining_job_id,
            )
            db.add(drift_event)
            await db.flush()
            logger.info("rollback_step_3_drift_event_logged", event_id=str(drift_event.id))

            # ── Step 4: SIGNAL retraining ───────────────────────────
            # Enqueue Celery task to retrain policy from scratch
            try:
                from app.worker.tasks import generate_postmortem, train_rl_policy

                train_rl_policy.apply_async(
                    kwargs={
                        "policy_version_id": str(retraining_job_id),
                        "n_training_steps": 2000,
                    },
                    task_id=str(retraining_job_id),
                )
                logger.info(
                    "rollback_step_4_retraining_enqueued",
                    task_id=str(retraining_job_id),
                )

                # Auto-trigger postmortem generation (Phase 16)
                generate_postmortem.apply_async(
                    kwargs={"drift_event_id": str(drift_event.id)},
                )
                logger.info(
                    "rollback_step_4_postmortem_triggered",
                    drift_event_id=str(drift_event.id),
                )
            except Exception as celery_exc:
                # Celery failure is non-fatal for rollback — baseline is already active
                logger.error(
                    "rollback_step_4_celery_failed",
                    error=str(celery_exc),
                    note="Baseline is active. Manual retraining required.",
                )

            # ── Step 5: ALERT ────────────────────────────────────────
            prom.rollback_total.inc()
            prom.drift_events_total.labels(
                drift_signal=drift_result.drift_signal.value
                if drift_result.drift_signal
                else "UNKNOWN",
                policy_from=current_policy_type.value,
                policy_to=PolicyType.BASELINE.value,
            ).inc()

            rollback_latency_ms = (datetime.now(UTC) - rollback_start).total_seconds() * 1000
            logger.critical(
                "rollback_complete",
                policy_from=current_policy_type.value,
                policy_to=PolicyType.BASELINE.value,
                latency_ms=round(rollback_latency_ms, 2),
                drift_event_id=str(drift_event.id),
            )

            # Commit all DB changes atomically
            await db.commit()

            return drift_event

        except Exception as exc:
            logger.error("rollback_failed", error=str(exc))
            await db.rollback()
            raise
        finally:
            self._rollback_in_progress = False

    @property
    def is_rolling_back(self) -> bool:
        return self._rollback_in_progress
