"""
MetricsCollector — centralized Prometheus metric emission.

WHY THIS EXISTS:
- Metrics scattered across modules = impossible to audit what's being tracked.
- Every SLO (P99 < 300ms, SLA violation rate < 5%, drift MTTR < 5min) needs
  a corresponding metric. Missing metric = invisible SLO breach.
- Structured metric emission with full label sets for Grafana dashboard slicing.

METRICS EMITTED (16 total):
  decisions_total              [policy_type, action, mode]
  decision_latency_ms          [policy_type] histogram
  reward_current               [policy_type] gauge
  reward_cumulative            [policy_type] gauge
  sla_violations_total         [violation_type]
  drift_events_total           [drift_signal, policy_from, policy_to]
  rollbacks_total              counter
  exploration_suppressed_total [reason]
  active_policy_info           [policy_type, policy_version] gauge
  fallback_total               counter
  action_boundary_violations   counter
  buffer_size                  [policy_type] gauge
  training_steps               [policy_type] gauge
  normalizer_fitted            [policy_type, version] gauge
  checkpoint_saves_total       [policy_type, status]
  api_request_latency_ms       [endpoint, status_code] histogram

WHAT BREAKS IF WRONG:
- Missing labels: can't filter by policy_type in Grafana.
- Wrong histogram buckets: SLO alerting fires at wrong threshold.
- Metric defined but never emitted: dashboard shows zero always.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager

import structlog
from prometheus_client import Gauge, Histogram

# Import shared metrics from core (prevents duplicate registration)
from app.core.metrics import (
    action_boundary_violations,
    active_policy_info,
    checkpoint_saves_total,
    cumulative_reward_gauge,
    decision_latency_ms,
    decisions_total,
    drift_events_total,
    exploration_suppressed_total,
    fallback_total,
    reward_gauge,
    rollback_total,
    sla_violations_total,
)
from app.core.metrics import (
    replay_buffer_size as buffer_size_gauge,
)
from app.core.metrics import (
    training_loss_gauge as training_steps_gauge,
)

logger = structlog.get_logger(__name__)

# ── Additional Metrics (not in core/metrics.py) ───────────────────────────────

drift_detection_latency_ms = Histogram(
    "rtde_drift_detection_latency_ms",
    "Time to complete drift evaluation in milliseconds",
    buckets=[10, 50, 100, 500, 1000, 5000],
)

# (imported from app.core.metrics)

# (imported from app.core.metrics)

# (imported from app.core.metrics)

normalizer_fitted_gauge = Gauge(
    "rtde_normalizer_fitted",
    "Whether state normalizer is fitted — 1 = yes",
    ["policy_type", "version_id"],
)

api_request_latency_ms = Histogram(
    "rtde_api_request_latency_ms",
    "API endpoint request latency in milliseconds",
    ["endpoint", "status_code"],
    buckets=[5, 10, 25, 50, 100, 200, 300, 500, 1000, 2000],
)

# ── Context Managers for SLO Tracking ────────────────────────────────────────


@contextmanager
def track_decision_latency(policy_type: str) -> Generator[None, None, None]:
    """Context manager to time and record decision latency."""
    start = time.perf_counter()
    try:
        yield
    finally:
        latency = (time.perf_counter() - start) * 1000
        decision_latency_ms.labels(policy_type=policy_type).observe(latency)


@contextmanager
def track_drift_evaluation() -> Generator[None, None, None]:
    """Context manager to time drift evaluation."""
    start = time.perf_counter()
    try:
        yield
    finally:
        latency = (time.perf_counter() - start) * 1000
        drift_detection_latency_ms.observe(latency)


@contextmanager
def track_api_latency(endpoint: str, status_code: int = 200) -> Generator[None, None, None]:
    """Context manager to time API request latency."""
    start = time.perf_counter()
    try:
        yield
    finally:
        latency = (time.perf_counter() - start) * 1000
        api_request_latency_ms.labels(endpoint=endpoint, status_code=str(status_code)).observe(
            latency
        )


class MetricsCollector:
    """
    Facade for emitting all RTDE metrics.

    Centralizes all metric emission in one place.
    Call methods on this class — never import prometheus_client directly.
    """

    @staticmethod
    def record_decision(
        policy_type: str,
        action: str,
        mode: str,
        latency_ms: float,
        fallback_used: bool,
    ) -> None:
        """Record a completed scaling decision."""
        decisions_total.labels(
            policy_type=policy_type,
            action=action,
            mode=mode,
        ).inc()
        decision_latency_ms.labels(policy_type=policy_type).observe(latency_ms)
        if fallback_used:
            fallback_total.inc()

    @staticmethod
    def record_reward(
        policy_type: str,
        reward: float,
        cumulative: float,
        sla_violated: bool,
    ) -> None:
        """Record reward and SLA status."""
        reward_gauge.labels(policy_type=policy_type).set(reward)
        cumulative_reward_gauge.labels(policy_type=policy_type).set(cumulative)
        if sla_violated:
            sla_violations_total.labels(violation_type="latency").inc()

    @staticmethod
    def record_drift_event(
        drift_signal: str,
        policy_from: str,
        policy_to: str,
    ) -> None:
        """Record a drift detection event and rollback."""
        drift_events_total.labels(
            drift_signal=drift_signal,
            policy_from=policy_from,
            policy_to=policy_to,
        ).inc()
        rollback_total.inc()

    @staticmethod
    def record_exploration_suppression(reason: str) -> None:
        """Record an ExplorationGuard suppression event."""
        exploration_suppressed_total.labels(reason=reason).inc()

    @staticmethod
    def set_active_policy(policy_type: str, policy_version: str) -> None:
        """Update active policy gauge."""
        active_policy_info.labels(
            policy_type=policy_type,
            policy_version=policy_version,
        ).set(1)

    @staticmethod
    def record_checkpoint(policy_type: str, success: bool) -> None:
        """Record checkpoint save outcome."""
        status = "success" if success else "failure"
        checkpoint_saves_total.labels(policy_type=policy_type, status=status).inc()

    @staticmethod
    def update_training_state(
        policy_type: str,
        buffer_size: int,
        training_steps: int,
    ) -> None:
        """Update training progress gauges."""
        buffer_size_gauge.labels(policy_type=policy_type).set(buffer_size)
        training_steps_gauge.labels(policy_type=policy_type).set(training_steps)

    @staticmethod
    def set_normalizer_fitted(policy_type: str, version_id: str, fitted: bool) -> None:
        """Record normalizer fit status."""
        normalizer_fitted_gauge.labels(
            policy_type=policy_type,
            version_id=version_id,
        ).set(1 if fitted else 0)

    @staticmethod
    def record_action_clip(direction: str) -> None:
        """Record when an action required safety clipping."""
        action_boundary_violations.labels(direction=direction).inc()
