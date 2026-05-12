"""
ExplorationGuard — proactive exploration safety.

WHY THIS EXISTS:
- Bandit and RL policies explore: they intentionally take suboptimal actions to learn.
- Exploration during a traffic spike = intentionally scale DOWN while load is high.
- Result: SLA breach, cascading failure, rollback fires — but damage is already done.
- The guard prevents this BEFORE the decision is made, not after.

THE FIX:
- Check system health BEFORE calling policy.decide().
- If system is under stress: suppress exploration, force exploitation of best-known action.
- Every suppression is logged to ExplorationGuardLog for threshold tuning.

WHAT BREAKS IF WRONG:
- No guard: RL explores SCALE_DOWN_3 during a burst → SLA breach → rollback.
  The rollback fires correctly, but you still violated the SLA.
- Guard too aggressive: policy never explores → never learns new traffic patterns.
- Guard too permissive: same as no guard.

INDUSTRY PARALLEL:
- Safe RL / constrained exploration. Standard in robotics, autonomous vehicles.
- Production ML systems constrain exploration when risk is high.
  You only run A/B tests when traffic is stable. Same principle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from app.core import metrics as prom
from app.core.config import get_settings
from app.schemas.common import SuppressionReason
from app.schemas.state import SystemState

logger = structlog.get_logger(__name__)


@dataclass
class PolicyStats:
    """Runtime stats the guard uses to assess system health."""

    sla_violation_rate_5min: float = 0.0
    consecutive_violations: int = 0
    total_decisions: int = 0
    total_violations: int = 0
    recent_rewards: list[float] = field(default_factory=list)


class ExplorationGuard:
    """
    Proactive safety layer that suppresses exploration during system stress.

    Decision: should_explore() returns True only when the system is in a
    stable, safe state. Called BEFORE policy.decide().

    All thresholds configurable via env vars.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._latency_warning_ms = settings.exploration_latency_warning_ms
        self._sla_warning_rate = settings.exploration_sla_warning_rate
        self._high_load_rps = settings.exploration_high_load_rps
        self._max_consecutive_violations = settings.exploration_max_consecutive_violations

    def should_explore(
        self,
        state: SystemState,
        policy_stats: PolicyStats,
    ) -> tuple[bool, SuppressionReason | None]:
        """
        Determine if exploration is safe given current system state.

        Returns:
            (True, None) — exploration allowed, system is stable.
            (False, reason) — exploration suppressed, reason explains why.

        Suppression conditions (checked in priority order):
        1. p99 latency already degraded → system can't afford more risk.
        2. SLA violation rate > threshold → active SLA breach in flight.
        3. Request rate > high-load threshold → traffic spike in progress.
        4. Consecutive violations streak → system showing instability pattern.
        """
        # Condition 1: Latency already degraded
        if state.p99_latency_ms > self._latency_warning_ms:
            return False, SuppressionReason.HIGH_LATENCY

        # Condition 2: Active SLA violations
        if policy_stats.sla_violation_rate_5min > self._sla_warning_rate:
            return False, SuppressionReason.SLA_VIOLATION_STREAK

        # Condition 3: Traffic spike
        if state.request_rate > self._high_load_rps:
            return False, SuppressionReason.HIGH_LOAD

        # Condition 4: Consecutive violation streak
        if policy_stats.consecutive_violations >= self._max_consecutive_violations:
            return False, SuppressionReason.SLA_VIOLATION_STREAK

        return True, None

    def check_and_log(
        self,
        state: SystemState,
        policy_stats: PolicyStats,
    ) -> bool:
        """
        Check if exploration is safe, log suppression if not.

        Returns True if exploration is allowed, False if suppressed.
        """
        allowed, reason = self.should_explore(state, policy_stats)

        if not allowed and reason is not None:
            logger.warning(
                "exploration_suppressed",
                reason=reason.value,
                p99_latency_ms=state.p99_latency_ms,
                request_rate=state.request_rate,
                sla_violation_rate=policy_stats.sla_violation_rate_5min,
                consecutive_violations=policy_stats.consecutive_violations,
            )
            prom.exploration_suppressed_total.labels(reason=reason.value).inc()

        return allowed

    def update_policy_stats(
        self,
        stats: PolicyStats,
        reward: float,
        sla_violated: bool,
    ) -> PolicyStats:
        """
        Update runtime stats after a decision outcome.

        Called after reward is computed to keep stats current.
        """
        stats.total_decisions += 1
        stats.recent_rewards.append(reward)

        # Keep only the last 300 rewards (5-minute window at 1 decision/sec)
        if len(stats.recent_rewards) > 300:
            stats.recent_rewards = stats.recent_rewards[-300:]

        if sla_violated:
            stats.total_violations += 1
            stats.consecutive_violations += 1
            # Compute 5-min violation rate
            recent_window = stats.recent_rewards[-300:]
            if recent_window:
                # Approximate: count rewards below SLA threshold
                stats.sla_violation_rate_5min = stats.consecutive_violations / max(
                    1, min(300, stats.total_decisions)
                )
        else:
            stats.consecutive_violations = 0
            stats.sla_violation_rate_5min = max(
                0.0,
                stats.sla_violation_rate_5min * 0.95,  # exponential decay on recovery
            )

        return stats
