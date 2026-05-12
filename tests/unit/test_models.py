"""
Model import and structure tests — Phase 1 gate.

Verifies:
- All 8 models import without error
- Table names are correct
- Critical columns exist on each model
- Base.metadata has all 8 tables registered
"""

from __future__ import annotations

import pytest

from app.core.database import Base
from app.models import (
    DecisionLog,
    DriftEvent,
    EnvironmentState,
    ExplorationGuardLog,
    PolicyCheckpoint,
    PolicyVersion,
    RewardLog,
    ScalingAction,
)


@pytest.mark.unit
class TestModelRegistry:
    """Verify all models are registered in Base.metadata."""

    def test_all_8_tables_registered(self) -> None:
        """Base.metadata must contain exactly 8 tables."""
        table_names = set(Base.metadata.tables.keys())
        expected = {
            "environment_states",
            "scaling_actions",
            "decision_logs",
            "reward_logs",
            "policy_versions",
            "policy_checkpoints",
            "drift_events",
            "exploration_guard_logs",
        }
        assert expected.issubset(table_names), f"Missing: {expected - table_names}"

    def test_environment_state_table_name(self) -> None:
        assert EnvironmentState.__tablename__ == "environment_states"

    def test_scaling_action_table_name(self) -> None:
        assert ScalingAction.__tablename__ == "scaling_actions"

    def test_decision_log_table_name(self) -> None:
        assert DecisionLog.__tablename__ == "decision_logs"

    def test_reward_log_table_name(self) -> None:
        assert RewardLog.__tablename__ == "reward_logs"

    def test_policy_version_table_name(self) -> None:
        assert PolicyVersion.__tablename__ == "policy_versions"

    def test_policy_checkpoint_table_name(self) -> None:
        assert PolicyCheckpoint.__tablename__ == "policy_checkpoints"

    def test_drift_event_table_name(self) -> None:
        assert DriftEvent.__tablename__ == "drift_events"

    def test_exploration_guard_log_table_name(self) -> None:
        assert ExplorationGuardLog.__tablename__ == "exploration_guard_logs"


@pytest.mark.unit
class TestModelColumns:
    """Verify critical columns exist on each model."""

    def test_environment_state_has_version(self) -> None:
        """version field required for optimistic concurrency."""
        columns = {c.name for c in EnvironmentState.__table__.columns}
        assert "version" in columns

    def test_environment_state_has_traffic_regime(self) -> None:
        """traffic_regime required for two-signal drift detector."""
        columns = {c.name for c in EnvironmentState.__table__.columns}
        assert "traffic_regime" in columns

    def test_decision_log_has_trace_id(self) -> None:
        """trace_id required for end-to-end request tracing."""
        columns = {c.name for c in DecisionLog.__table__.columns}
        assert "trace_id" in columns

    def test_decision_log_has_q_values(self) -> None:
        """q_values JSONB required for Decision Explainer."""
        columns = {c.name for c in DecisionLog.__table__.columns}
        assert "q_values" in columns

    def test_decision_log_has_shadow_flag(self) -> None:
        """shadow_flag required to distinguish active from shadow decisions."""
        columns = {c.name for c in DecisionLog.__table__.columns}
        assert "shadow_flag" in columns

    def test_reward_log_has_n_step(self) -> None:
        """n_step_reward required for temporal credit assignment fix."""
        columns = {c.name for c in RewardLog.__table__.columns}
        assert "n_step_reward" in columns

    def test_reward_log_has_baseline_reward(self) -> None:
        """baseline_reward required for drift comparison."""
        columns = {c.name for c in RewardLog.__table__.columns}
        assert "baseline_reward" in columns

    def test_policy_version_has_normalizer_path(self) -> None:
        """normalizer_path required — mismatch with weights = silent wrong inference."""
        columns = {c.name for c in PolicyVersion.__table__.columns}
        assert "normalizer_path" in columns

    def test_policy_version_has_eval_seeds(self) -> None:
        """eval_seeds required — must be >= 5 for promotion."""
        columns = {c.name for c in PolicyVersion.__table__.columns}
        assert "eval_seeds" in columns

    def test_drift_event_has_psi_score(self) -> None:
        """psi_score required for two-signal drift detection."""
        columns = {c.name for c in DriftEvent.__table__.columns}
        assert "psi_score" in columns

    def test_drift_event_has_drift_signal(self) -> None:
        """drift_signal enum required to distinguish input vs reward drift."""
        columns = {c.name for c in DriftEvent.__table__.columns}
        assert "drift_signal" in columns

    def test_exploration_guard_has_reason(self) -> None:
        """suppression_reason required for guard tuning analysis."""
        columns = {c.name for c in ExplorationGuardLog.__table__.columns}
        assert "suppression_reason" in columns

    def test_scaling_action_has_policy_mode(self) -> None:
        """policy_mode required to distinguish active from shadow commits."""
        columns = {c.name for c in ScalingAction.__table__.columns}
        assert "policy_mode" in columns
