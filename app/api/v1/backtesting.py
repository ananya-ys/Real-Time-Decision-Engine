"""
Backtesting API — replay historical trace through policies.

ENDPOINTS:
  POST /api/v1/backtest/run                       → run backtest (Celery task)
  GET  /api/v1/backtest/quick                     → quick in-process backtest (small window)
  POST /api/v1/backtest/counterfactual/{id}       → counterfactual for one decision
  GET  /api/v1/backtest/results/{run_id}          → get backtest results from cache
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog
from fastapi import APIRouter, Body, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.backtesting.engine import BacktestingEngine
from app.core.auth import CurrentUser, Role
from app.dependencies.auth import require_role
from app.dependencies.db import get_db
from app.policies.bandit_policy import BanditPolicy
from app.policies.baseline_policy import BaselinePolicy

router = APIRouter(prefix="/api/v1/backtest", tags=["backtesting"])
logger = structlog.get_logger(__name__)

_engine = BacktestingEngine()


@router.post("/quick")
async def quick_backtest(
    window_hours: int = Body(default=6, embed=True, ge=1, le=24),
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> dict[str, Any]:
    """
    Quick in-process backtest: Baseline vs Bandit on recent history.
    Runs synchronously for small windows (≤6h).
    For larger windows, use /backtest/run (Celery task).
    """
    logger.info("quick_backtest_started", window_hours=window_hours, actor=user.user_id)

    policies = {
        "baseline": BaselinePolicy(),
        "bandit": BanditPolicy(epsilon_start=0.0),  # exploit-only for fair comparison
    }

    report = await _engine.run_backtest(
        candidate_policies=policies,
        window_hours=window_hours,
        db=db,
    )

    logger.info(
        "quick_backtest_complete",
        run_id=str(report.run_id),
        n_states=report.n_historical_decisions,
    )
    return report.to_dict()


@router.post("/counterfactual/{decision_id}")
async def compute_counterfactual(
    decision_id: uuid.UUID,
    counterfactual_delta: int = Body(
        ...,
        embed=True,
        ge=-3,
        le=3,
        description="Instance count delta for the what-if scenario",
    ),
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(require_role(Role.VIEWER)),
) -> dict[str, Any]:
    """
    Counterfactual: 'What if we had chosen a different action?'

    Computes: what reward would we have gotten with counterfactual_delta
    vs what reward we actually got with the chosen action.
    Does NOT re-run any model. Pure reward function evaluation.
    """
    try:
        result = await _engine.compute_counterfactual(
            decision_log_id=decision_id,
            counterfactual_action_delta=counterfactual_delta,
            db=db,
        )
    except ValueError as exc:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return result


@router.post("/run")
async def enqueue_backtest(
    window_hours: int = Body(default=24, embed=True, ge=1, le=168),
    user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> dict[str, Any]:
    """
    Enqueue a full backtest as a Celery task (for large windows).
    Returns task_id to poll for results.
    """
    from app.worker.tasks import run_backtest_task

    task_id = str(uuid.uuid4())
    try:
        run_backtest_task.apply_async(
            kwargs={"window_hours": window_hours, "task_id": task_id},
            task_id=task_id,
        )
    except Exception as exc:
        # Celery broker (Redis) is unavailable — return 503, not 500
        from fastapi import HTTPException

        raise HTTPException(
            status_code=503,
            detail=(
                f"Backtest task queue unavailable: {exc}. "
                "Celery broker (Redis) may be down. "
                "Use POST /api/v1/backtest/quick for synchronous backtesting."
            ),
        ) from exc

    return {
        "status": "enqueued",
        "task_id": task_id,
        "window_hours": window_hours,
        "actor": user.user_id,
    }
