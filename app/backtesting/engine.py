"""
BacktestingEngine — replay historical decisions through any policy.

WHY THIS EXISTS:
The review: "backtesting is one of the biggest missing credibility layers."
Without backtesting, policy promotion is based entirely on:
  - Shadow mode (logs but doesn't commit — no real pressure)
  - Synthetic benchmarks (nice, but not real traffic patterns)

Backtesting adds:
  - Replay actual historical trace through candidate policy
  - Compare: what did RL choose vs what Bandit would have chosen?
  - Counterfactual: what if SCALE_UP_3 was taken at T=15min?
  - Side-by-side reward comparison on identical inputs

USAGE:
  engine = BacktestingEngine()
  report = await engine.run_backtest(
      policy=RLPolicy(normalizer=fitted_normalizer),
      window_hours=24,
      db=db,
  )
  # report.winner, report.reward_comparison, report.sla_comparison

WHAT BACKTESTING IS NOT:
- Re-executing real scaling actions (all actions are simulated)
- Training the policy (weights frozen during backtest)
- Real-time (it's offline, runs in Celery worker)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.decision_log import DecisionLog
from app.policies.base_policy import PolicyInterface
from app.policies.baseline_policy import BaselinePolicy
from app.schemas.common import TrafficRegime
from app.schemas.state import SystemState
from app.services.reward_service import RewardService

logger = structlog.get_logger(__name__)


@dataclass
class BacktestResult:
    """Results from a single policy backtest run."""

    policy_name: str
    window_start: datetime
    window_end: datetime
    n_decisions: int
    cumulative_reward: float
    avg_reward_per_decision: float
    sla_violations: int
    sla_violation_rate: float
    actions_taken: dict[str, int] = field(default_factory=dict)
    reward_series: list[float] = field(default_factory=list)


@dataclass
class BacktestReport:
    """Complete backtest comparison across policies."""

    run_id: uuid.UUID
    generated_at: datetime
    window_hours: int
    n_historical_decisions: int
    results: dict[str, BacktestResult]

    def to_dict(self) -> dict[str, Any]:
        results_dict = {}
        for name, r in self.results.items():
            results_dict[name] = {
                "policy_name": r.policy_name,
                "n_decisions": r.n_decisions,
                "cumulative_reward": round(r.cumulative_reward, 4),
                "avg_reward_per_decision": round(r.avg_reward_per_decision, 4),
                "sla_violations": r.sla_violations,
                "sla_violation_rate": round(r.sla_violation_rate, 4),
                "actions_taken": r.actions_taken,
            }

        # Determine winner by cumulative reward
        winner = max(self.results.items(), key=lambda x: x[1].cumulative_reward)[0]

        return {
            "run_id": str(self.run_id),
            "generated_at": self.generated_at.isoformat(),
            "window_hours": self.window_hours,
            "n_historical_decisions": self.n_historical_decisions,
            "results": results_dict,
            "winner": winner,
            "winner_margin": self._compute_winner_margin(winner),
        }

    def _compute_winner_margin(self, winner: str) -> float | None:
        """How much better is the winner vs the second-best policy?"""
        if len(self.results) < 2:
            return None
        rewards = sorted([r.cumulative_reward for r in self.results.values()], reverse=True)
        if rewards[0] == 0:
            return None
        return round((rewards[0] - rewards[1]) / abs(rewards[0]), 4)


class BacktestingEngine:
    """
    Replays historical states through candidate policies.

    Steps:
    1. Load historical DecisionLog entries from the backtest window.
    2. Reconstruct SystemState from state_snapshot JSONB.
    3. For each historical state: call candidate_policy.decide().
    4. Compute reward for candidate's action.
    5. Compare candidate reward vs original reward vs baseline reward.
    """

    def __init__(self) -> None:
        self._reward_service = RewardService()

    async def run_backtest(
        self,
        candidate_policies: dict[str, PolicyInterface],
        window_hours: int = 24,
        db: AsyncSession = None,  # type: ignore[assignment]
        anchor_time: datetime | None = None,
    ) -> BacktestReport:
        """
        Run backtest for all candidate policies against the same historical trace.

        Args:
            candidate_policies: Dict of {name: policy} to evaluate.
            window_hours: How many hours of history to replay.
            db: Database session.
            anchor_time: End of backtest window. Defaults to now.

        Returns:
            BacktestReport with results for each candidate policy.
        """
        if anchor_time is None:
            anchor_time = datetime.now(UTC)

        window_start = anchor_time - timedelta(hours=window_hours)

        # Load historical decisions
        historical = await self._load_historical_states(window_start, anchor_time, db)

        if not historical:
            logger.warning("backtest_no_historical_data", window_hours=window_hours)
            return BacktestReport(
                run_id=uuid.uuid4(),
                generated_at=datetime.now(UTC),
                window_hours=window_hours,
                n_historical_decisions=0,
                results={},
            )

        logger.info(
            "backtest_started",
            n_states=len(historical),
            window_hours=window_hours,
            policies=list(candidate_policies.keys()),
        )

        # Always include baseline as reference
        if "baseline" not in candidate_policies:
            candidate_policies["baseline"] = BaselinePolicy()

        # Run each policy through the historical trace
        results = {}
        for policy_name, policy in candidate_policies.items():
            result = await self._run_single_policy(
                policy_name=policy_name,
                policy=policy,
                historical_states=historical,
                window_start=window_start,
                window_end=anchor_time,
            )
            results[policy_name] = result

        report = BacktestReport(
            run_id=uuid.uuid4(),
            generated_at=datetime.now(UTC),
            window_hours=window_hours,
            n_historical_decisions=len(historical),
            results=results,
        )

        logger.info(
            "backtest_complete",
            run_id=str(report.run_id),
            n_states=len(historical),
            winner=max(results.items(), key=lambda x: x[1].cumulative_reward)[0]
            if results
            else None,
        )

        return report

    async def _load_historical_states(
        self,
        start: datetime,
        end: datetime,
        db: AsyncSession,
    ) -> list[SystemState]:
        """Reconstruct SystemState objects from historical DecisionLog snapshots."""
        result = await db.execute(
            select(DecisionLog)
            .where(
                DecisionLog.created_at >= start,
                DecisionLog.created_at <= end,
            )
            .order_by(DecisionLog.created_at)
            .limit(5000)  # cap to prevent memory issues
        )
        logs = result.scalars().all()

        states = []
        for log in logs:
            snapshot = log.state_snapshot
            if not snapshot:
                continue
            try:
                state = SystemState(
                    cpu_utilization=float(snapshot.get("cpu_utilization", 0.5)),
                    request_rate=float(snapshot.get("request_rate", 1000.0)),
                    p99_latency_ms=float(snapshot.get("p99_latency_ms", 200.0)),
                    instance_count=max(1, int(snapshot.get("instance_count", 5))),
                    hour_of_day=int(snapshot.get("hour_of_day", 12)),
                    day_of_week=int(snapshot.get("day_of_week", 1)),
                    traffic_regime=TrafficRegime(snapshot.get("traffic_regime", "UNKNOWN")),
                )
                states.append(state)
            except Exception as exc:
                logger.warning("backtest_state_parse_error", error=str(exc))
                continue

        return states

    async def _run_single_policy(
        self,
        policy_name: str,
        policy: PolicyInterface,
        historical_states: list[SystemState],
        window_start: datetime,
        window_end: datetime,
    ) -> BacktestResult:
        """Run one policy through the entire historical trace."""
        cumulative_reward = 0.0
        sla_violations = 0
        actions: dict[str, int] = {}
        rewards: list[float] = []
        last_delta = 0

        for state in historical_states:
            try:
                # Weights are FROZEN — no training during backtest
                decision = await policy.decide(state, explore=False)
            except Exception as exc:
                logger.debug("backtest_policy_error", policy=policy_name, error=str(exc))
                continue

            last_delta = decision.instances_after - decision.instances_before
            reward_components = self._reward_service.compute_reward(
                p99_latency_ms=state.p99_latency_ms,
                instance_count=decision.instances_after,
                last_action_delta=last_delta,
            )

            cumulative_reward += reward_components.total_reward
            rewards.append(reward_components.total_reward)

            if reward_components.sla_violated:
                sla_violations += 1

            action_str = decision.action.value
            actions[action_str] = actions.get(action_str, 0) + 1

        n = len(historical_states)
        return BacktestResult(
            policy_name=policy_name,
            window_start=window_start,
            window_end=window_end,
            n_decisions=n,
            cumulative_reward=cumulative_reward,
            avg_reward_per_decision=cumulative_reward / max(1, n),
            sla_violations=sla_violations,
            sla_violation_rate=sla_violations / max(1, n),
            actions_taken=actions,
            reward_series=rewards,
        )

    async def compute_counterfactual(
        self,
        decision_log_id: uuid.UUID,
        counterfactual_action_delta: int,
        db: AsyncSession,
    ) -> dict[str, Any]:
        """
        Counterfactual: what would have happened if a different action was taken?

        Computes reward for the counterfactual action vs the actual action.
        Does NOT re-run any model — uses stored state snapshot.

        Args:
            decision_log_id: The past decision to counterfactual.
            counterfactual_action_delta: Instance count delta for the what-if action.
            db: Database session.

        Returns:
            Dict comparing actual vs counterfactual reward.
        """
        result = await db.execute(select(DecisionLog).where(DecisionLog.id == decision_log_id))
        log = result.scalar_one_or_none()
        if log is None:
            raise ValueError(f"Decision {decision_log_id} not found")

        snapshot = log.state_snapshot
        state = SystemState(
            cpu_utilization=float(snapshot.get("cpu_utilization", 0.5)),
            request_rate=float(snapshot.get("request_rate", 1000.0)),
            p99_latency_ms=float(snapshot.get("p99_latency_ms", 200.0)),
            instance_count=max(1, int(snapshot.get("instance_count", 5))),
        )

        # Actual outcome
        actual_delta = self._action_delta(log.action)
        actual_instances = max(1, min(20, state.instance_count + actual_delta))
        actual_reward = self._reward_service.compute_reward(
            p99_latency_ms=state.p99_latency_ms,
            instance_count=actual_instances,
            last_action_delta=actual_delta,
        )

        # Counterfactual outcome
        cf_instances = max(1, min(20, state.instance_count + counterfactual_action_delta))
        cf_reward = self._reward_service.compute_reward(
            p99_latency_ms=state.p99_latency_ms,
            instance_count=cf_instances,
            last_action_delta=counterfactual_action_delta,
        )

        reward_diff = cf_reward.total_reward - actual_reward.total_reward

        return {
            "decision_id": str(decision_log_id),
            "actual": {
                "action": log.action,
                "instances": actual_instances,
                "reward": round(actual_reward.total_reward, 4),
                "sla_violated": actual_reward.sla_violated,
            },
            "counterfactual": {
                "instance_delta": counterfactual_action_delta,
                "instances": cf_instances,
                "reward": round(cf_reward.total_reward, 4),
                "sla_violated": cf_reward.sla_violated,
            },
            "reward_difference": round(reward_diff, 4),
            "counterfactual_is_better": reward_diff > 0,
        }

    @staticmethod
    def _action_delta(action_str: str) -> int:
        return {
            "SCALE_UP_3": 3,
            "SCALE_UP_1": 1,
            "HOLD": 0,
            "SCALE_DOWN_1": -1,
            "SCALE_DOWN_3": -3,
        }.get(action_str, 0)
