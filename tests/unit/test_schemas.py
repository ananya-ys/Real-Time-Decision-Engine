"""
Schema validation tests — Phase 1 gate.

Verifies:
- SystemState rejects invalid values (cpu > 1.0, negative latency)
- SystemState produces correct feature vectors
- Create/Read separation works correctly
- All enums have expected values
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.common import (
    ActionType,
    DriftSignal,
    PolicyMode,
    PolicyStatus,
    PolicyType,
    SuppressionReason,
    TrafficRegime,
)
from app.schemas.decision import DecisionRequest, ScalingDecision
from app.schemas.reward import RewardWeights
from app.schemas.state import SystemState


@pytest.mark.unit
class TestSystemState:
    """Validate SystemState — the S in the MDP."""

    def test_valid_state(self) -> None:
        """Valid state should construct without error."""
        state = SystemState(
            cpu_utilization=0.75,
            request_rate=1500.0,
            p99_latency_ms=250.0,
            instance_count=5,
        )
        assert state.cpu_utilization == 0.75
        assert state.instance_count == 5

    def test_cpu_above_one_rejected(self) -> None:
        """cpu_utilization > 1.0 must be rejected."""
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            SystemState(
                cpu_utilization=1.5,
                request_rate=100.0,
                p99_latency_ms=50.0,
                instance_count=2,
            )

    def test_cpu_below_zero_rejected(self) -> None:
        """cpu_utilization < 0.0 must be rejected."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            SystemState(
                cpu_utilization=-0.1,
                request_rate=100.0,
                p99_latency_ms=50.0,
                instance_count=2,
            )

    def test_negative_latency_rejected(self) -> None:
        """Negative latency must be rejected."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            SystemState(
                cpu_utilization=0.5,
                request_rate=100.0,
                p99_latency_ms=-10.0,
                instance_count=2,
            )

    def test_zero_instances_rejected(self) -> None:
        """instance_count < 1 must be rejected."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            SystemState(
                cpu_utilization=0.5,
                request_rate=100.0,
                p99_latency_ms=50.0,
                instance_count=0,
            )

    def test_feature_vector_length(self) -> None:
        """Feature vector must have 6 elements."""
        state = SystemState(
            cpu_utilization=0.5,
            request_rate=100.0,
            p99_latency_ms=50.0,
            instance_count=3,
            hour_of_day=14,
            day_of_week=2,
        )
        fv = state.to_feature_vector()
        assert len(fv) == 6
        assert fv[0] == 0.5  # cpu
        assert fv[4] == pytest.approx(14.0 / 23.0)  # hour normalized

    def test_snapshot_dict(self) -> None:
        """Snapshot dict must include all fields."""
        state = SystemState(
            cpu_utilization=0.8,
            request_rate=2000.0,
            p99_latency_ms=300.0,
            instance_count=10,
            traffic_regime=TrafficRegime.BURST,
        )
        snapshot = state.to_snapshot_dict()
        assert snapshot["cpu_utilization"] == 0.8
        assert snapshot["traffic_regime"] == "BURST"
        assert "instance_count" in snapshot

    def test_defaults(self) -> None:
        """Default hour, day, traffic regime should be set."""
        state = SystemState(
            cpu_utilization=0.5,
            request_rate=100.0,
            p99_latency_ms=50.0,
            instance_count=2,
        )
        assert state.hour_of_day == 0
        assert state.day_of_week == 0
        assert state.traffic_regime == TrafficRegime.UNKNOWN

    def test_hour_out_of_range_rejected(self) -> None:
        """hour_of_day > 23 must be rejected."""
        with pytest.raises(ValidationError, match="less than or equal to 23"):
            SystemState(
                cpu_utilization=0.5,
                request_rate=100.0,
                p99_latency_ms=50.0,
                instance_count=2,
                hour_of_day=24,
            )


@pytest.mark.unit
class TestScalingDecision:
    """Validate ScalingDecision — the A in the MDP."""

    def test_valid_decision(self) -> None:
        """Valid decision should construct without error."""
        decision = ScalingDecision(
            action=ActionType.SCALE_UP_1,
            instances_before=3,
            instances_after=4,
            policy_type=PolicyType.BASELINE,
        )
        assert decision.action == ActionType.SCALE_UP_1

    def test_confidence_out_of_range(self) -> None:
        """Confidence > 1.0 must be rejected."""
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            ScalingDecision(
                action=ActionType.HOLD,
                instances_before=3,
                instances_after=3,
                policy_type=PolicyType.RL,
                confidence=1.5,
            )


@pytest.mark.unit
class TestDecisionRequest:
    """Validate DecisionRequest — the API input contract."""

    def test_valid_request(self) -> None:
        """Valid request wraps a SystemState correctly."""
        req = DecisionRequest(
            state=SystemState(
                cpu_utilization=0.7,
                request_rate=1000.0,
                p99_latency_ms=200.0,
                instance_count=5,
            )
        )
        assert req.state.cpu_utilization == 0.7

    def test_invalid_state_rejected(self) -> None:
        """Request with invalid state must be rejected."""
        with pytest.raises(ValidationError):
            DecisionRequest(
                state=SystemState(
                    cpu_utilization=2.0,  # invalid
                    request_rate=100.0,
                    p99_latency_ms=50.0,
                    instance_count=2,
                )
            )


@pytest.mark.unit
class TestRewardWeights:
    """Validate RewardWeights — the reward function configuration."""

    def test_defaults(self) -> None:
        """Default weights should match .env.example values."""
        w = RewardWeights()
        assert w.alpha_latency == 1.0
        assert w.beta_cost == 0.5
        assert w.gamma_sla == 2.0
        assert w.delta_instability == 0.3

    def test_negative_weight_rejected(self) -> None:
        """Negative weights must be rejected."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            RewardWeights(alpha_latency=-1.0)


@pytest.mark.unit
class TestEnumValues:
    """Verify all enums have expected values."""

    def test_action_types(self) -> None:
        assert len(ActionType) == 5
        assert ActionType.HOLD.value == "HOLD"

    def test_policy_types(self) -> None:
        assert len(PolicyType) == 3
        assert PolicyType.RL.value == "RL"

    def test_policy_status(self) -> None:
        assert len(PolicyStatus) == 4
        assert PolicyStatus.SHADOW.value == "SHADOW"

    def test_traffic_regime(self) -> None:
        assert len(TrafficRegime) == 4
        assert TrafficRegime.BURST.value == "BURST"

    def test_drift_signal(self) -> None:
        assert len(DriftSignal) == 3
        assert DriftSignal.BOTH.value == "BOTH"

    def test_suppression_reason(self) -> None:
        assert len(SuppressionReason) == 4
        assert SuppressionReason.HIGH_LATENCY.value == "HIGH_LATENCY"

    def test_policy_mode(self) -> None:
        assert len(PolicyMode) == 2
        assert PolicyMode.SHADOW.value == "SHADOW"
