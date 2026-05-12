"""
Celery application — async task worker.

Implemented in Phase 5. Skeleton here for docker-compose service reference.
"""

from __future__ import annotations

from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "rtde_worker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    worker_send_task_events=True,
    # Celery Beat schedule
    beat_schedule={
        "evaluate-drift-every-60s": {
            "task": "rtde.evaluate_and_rollback_if_drift",
            "schedule": 60.0,
            "options": {"queue": "drift_eval"},
        },
        "compute-trust-scores-every-60s": {
            "task": "rtde.compute_trust_scores",
            "schedule": 60.0,
            "options": {"queue": "celery"},
        },
        "auto-canary-health-check-every-60s": {
            "task": "rtde.auto_canary_health_check",
            "schedule": 60.0,
            "options": {"queue": "celery"},
        },
    },
)
