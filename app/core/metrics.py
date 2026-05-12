"""
Prometheus metric definitions for RTDE.

WHY THIS EXISTS:
- Prometheus pull model: /metrics endpoint scraped every 15s. No push needed.
- Counters, histograms, and gauges defined in one place.
- SLO alerting requires metrics. Without metrics, SLO violations are invisible.

METRIC NAMING CONVENTION:
- rtde_<subsystem>_<metric>_<unit>
- All durations in milliseconds (ms) for consistency with P99 SLO
- Counters: _total suffix (Prometheus convention)
- Labels on counters: kept sparse to avoid cardinality explosion

WHAT BREAKS IF WRONG:
- No metrics = no dashboards = no alerts = silent failures.
- Metrics scattered across modules = impossible to find what's tracked.
- Too many label cardinalities = Prometheus OOM.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ── Decision Metrics ────────────────────────────────────────────
decisions_total = Counter(
    "rtde_decisions_total",
    "Total scaling decisions made",
    ["policy_type", "action", "mode"],  # mode: active or shadow
)

decision_latency_ms = Histogram(
    "rtde_decision_latency_ms",
    "Decision inference latency in milliseconds",
    ["policy_type"],
    buckets=[5, 10, 25, 50, 100, 150, 200, 250, 300, 500, 1000],
)

# ── Reward Metrics ──────────────────────────────────────────────
reward_gauge = Gauge(
    "rtde_reward_current",
    "Most recent reward value",
    ["policy_type"],
)

cumulative_reward_gauge = Gauge(
    "rtde_reward_cumulative",
    "Cumulative reward since policy activation",
    ["policy_type"],
)

# ── SLA Metrics ─────────────────────────────────────────────────
sla_violations_total = Counter(
    "rtde_sla_violations_total",
    "Total SLA violation events",
    ["violation_type"],  # latency, error_rate
)

# P99 latency gauge (updated per decision for real-time SLO tracking)
p99_latency_gauge = Gauge(
    "rtde_p99_latency_ms",
    "Current P99 latency observed by decision service",
    ["policy_type"],
)

instance_count_gauge = Gauge(
    "rtde_instance_count",
    "Current number of scaling instances",
    ["policy_type"],
)

# ── Drift & Rollback Metrics ───────────────────────────────────
drift_events_total = Counter(
    "rtde_drift_events_total",
    "Total drift detection events",
    ["drift_signal", "policy_from", "policy_to"],
)

rollback_total = Counter(
    "rtde_rollbacks_total",
    "Total policy rollbacks executed",
)

drift_psi_gauge = Gauge(
    "rtde_drift_psi_score",
    "Most recent PSI score from drift detector",
)

drift_reward_delta_gauge = Gauge(
    "rtde_drift_reward_delta",
    "Most recent reward delta from drift detector",
)

# ── Training Metrics ────────────────────────────────────────────
training_steps_total = Counter(
    "rtde_training_steps_total",
    "Total DQN training gradient steps executed",
    ["policy_type"],
)

training_loss_gauge = Gauge(
    "rtde_training_loss",
    "Most recent DQN training loss",
    ["policy_type"],
)

replay_buffer_size = Gauge(
    "rtde_replay_buffer_size",
    "Current number of transitions in the replay buffer",
    ["policy_type"],
)

# Tracks epsilon for bandit policy decay monitoring
bandit_epsilon_gauge = Gauge(
    "rtde_bandit_epsilon",
    "Current epsilon (exploration rate) for bandit policy",
)

# ── Exploration Guard Metrics ──────────────────────────────────
exploration_suppressed_total = Counter(
    "rtde_exploration_suppressed_total",
    "Times ExplorationGuard suppressed exploration",
    ["reason"],
)

# ── Infrastructure Metrics ─────────────────────────────────────
active_policy_info = Gauge(
    "rtde_active_policy_info",
    "Currently active policy (1 = active)",
    ["policy_type", "policy_version"],
)

fallback_total = Counter(
    "rtde_fallback_total",
    "Times inference fell back to baseline due to error",
)

action_boundary_violations = Counter(
    "rtde_action_boundary_violations_total",
    "Actions that violated instance count boundaries",
    ["direction"],
)

# ── HTTP / API Metrics ─────────────────────────────────────────
http_requests_total = Counter(
    "rtde_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)

http_request_duration_ms = Histogram(
    "rtde_http_request_duration_ms",
    "HTTP request duration in milliseconds",
    ["method", "path"],
    buckets=[10, 25, 50, 100, 200, 300, 500, 1000, 2000],
)

# ── DB Pool Metrics ────────────────────────────────────────────
db_pool_size_gauge = Gauge(
    "rtde_db_pool_size",
    "Current database connection pool size",
)

db_pool_checked_out = Gauge(
    "rtde_db_pool_checked_out",
    "Database connections currently checked out",
)

# ── Checkpoint Metrics ─────────────────────────────────────────
checkpoint_saves_total = Counter(
    "rtde_checkpoint_saves_total",
    "Total checkpoint save operations",
    ["policy_type", "status"],  # status: success, failure
)

checkpoint_loads_total = Counter(
    "rtde_checkpoint_loads_total",
    "Total checkpoint load operations",
    ["policy_type", "status"],
)
