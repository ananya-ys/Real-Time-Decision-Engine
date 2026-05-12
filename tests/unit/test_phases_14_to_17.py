"""
Phase 14-17 tests — Cost, Circuit Breaker, Postmortem, Trust Score.
"""

from __future__ import annotations

import pytest

from app.circuit_breaker.db_breaker import CircuitBreakerOpenError, RedisCircuitBreaker
from app.cost.cost_tracker import CostTracker


@pytest.mark.unit
class TestDecisionCost:
    """Verify cost computation logic."""

    @pytest.fixture
    def tracker(self) -> CostTracker:
        return CostTracker(hourly_budget_usd=100.0)

    def test_cost_increases_with_inference_time(self, tracker: CostTracker) -> None:
        """Longer inference = higher compute cost."""
        fast = tracker.compute_decision_cost(inference_ms=10.0, instance_count=5)
        slow = tracker.compute_decision_cost(inference_ms=100.0, instance_count=5)
        assert slow.compute_cost_usd > fast.compute_cost_usd

    def test_cost_increases_with_instance_count(self, tracker: CostTracker) -> None:
        """More instances = higher infrastructure cost."""
        few = tracker.compute_decision_cost(inference_ms=10.0, instance_count=2)
        many = tracker.compute_decision_cost(inference_ms=10.0, instance_count=20)
        assert many.instance_cost_usd > few.instance_cost_usd

    def test_total_cost_is_sum(self, tracker: CostTracker) -> None:
        """Total cost = compute + instance."""
        cost = tracker.compute_decision_cost(inference_ms=50.0, instance_count=10)
        assert cost.total_cost_usd == pytest.approx(cost.compute_cost_usd + cost.instance_cost_usd)

    def test_cost_to_dict(self, tracker: CostTracker) -> None:
        cost = tracker.compute_decision_cost(inference_ms=42.0, instance_count=7)
        d = cost.to_dict()
        assert "inference_ms" in d
        assert "total_cost_usd" in d
        assert d["inference_ms"] == 42.0
        assert d["instance_count"] == 7

    def test_zero_inference_gives_minimal_cost(self, tracker: CostTracker) -> None:
        """Zero inference time should give very low (but positive) cost."""
        cost = tracker.compute_decision_cost(inference_ms=0.0, instance_count=1)
        assert cost.compute_cost_usd == 0.0
        assert cost.instance_cost_usd > 0  # instance cost still applies
        assert cost.total_cost_usd > 0


@pytest.mark.unit
class TestCircuitBreakerLogic:
    """Verify circuit breaker state machine logic (no Redis)."""

    def test_circuit_breaker_open_error_message(self) -> None:
        exc = CircuitBreakerOpenError("postgres", "OPEN")
        assert "postgres" in str(exc)
        assert "OPEN" in str(exc)

    def test_circuit_breaker_open_error_is_exception(self) -> None:
        exc = CircuitBreakerOpenError("redis", "HALF_OPEN")
        assert isinstance(exc, Exception)

    @pytest.mark.asyncio
    async def test_circuit_breaker_calls_pass_on_success(self) -> None:
        """Successful calls should not open circuit."""
        breaker = RedisCircuitBreaker("test_success", failure_threshold=3)

        async def success_coro() -> str:
            return "ok"

        from unittest.mock import AsyncMock, patch

        with (
            patch.object(
                breaker,
                "_get_state",
                AsyncMock(return_value={"state": "CLOSED", "failure_count": 0, "success_count": 0}),
            ),
            patch.object(breaker, "_save_state", AsyncMock()),
        ):
            result = await breaker.call(success_coro())
            assert result == "ok"

    @pytest.mark.asyncio
    async def test_open_circuit_raises_immediately(self) -> None:
        """Open circuit must raise CircuitBreakerOpenError without calling coro."""
        from datetime import UTC, datetime, timedelta
        from unittest.mock import AsyncMock, patch

        breaker = RedisCircuitBreaker("test_open", recovery_timeout_seconds=3600)

        # Set state to OPEN with recent failure (timeout not elapsed)
        with patch.object(
            breaker,
            "_get_state",
            AsyncMock(
                return_value={
                    "state": "OPEN",
                    "failure_count": 5,
                    "success_count": 0,
                    "opened_at": (datetime.now(UTC) - timedelta(seconds=10)).isoformat(),
                }
            ),
        ):
            called = False

            async def should_not_be_called() -> str:
                nonlocal called
                called = True
                return "should_not_reach"

            import warnings

            with pytest.raises(CircuitBreakerOpenError), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                await breaker.call(should_not_be_called())

            assert not called, "Circuit was open but underlying coroutine was called"


@pytest.mark.unit
class TestPostmortemGeneration:
    """Verify postmortem analysis logic (without DB)."""

    @pytest.fixture
    def generator(self):
        from app.workflow.postmortem import PostmortemGenerator

        return PostmortemGenerator()

    def test_root_cause_for_input_drift(self, generator) -> None:

        mock_drift = type(
            "DriftEvent",
            (),
            {
                "drift_signal": "INPUT_DRIFT",
                "psi_score": 0.35,
                "reward_delta": -0.5,
            },
        )()
        result = generator._analyze_root_cause(mock_drift)
        assert "PSI" in result or "distribution" in result.lower()
        assert len(result) > 50

    def test_root_cause_for_reward_degradation(self, generator) -> None:
        mock_drift = type(
            "DriftEvent",
            (),
            {
                "drift_signal": "REWARD_DEGRADATION",
                "psi_score": 0.05,
                "reward_delta": -4.5,
            },
        )()
        result = generator._analyze_root_cause(mock_drift)
        assert "degradation" in result.lower() or "reward" in result.lower()

    def test_root_cause_for_both(self, generator) -> None:
        mock_drift = type(
            "DriftEvent",
            (),
            {
                "drift_signal": "BOTH",
                "psi_score": 0.25,
                "reward_delta": -3.0,
            },
        )()
        result = generator._analyze_root_cause(mock_drift)
        assert "both" in result.lower() or "severe" in result.lower()

    def test_next_steps_for_input_drift(self, generator) -> None:
        mock_drift = type(
            "DriftEvent",
            (),
            {
                "drift_signal": "INPUT_DRIFT",
                "retraining_job_id": None,
            },
        )()
        steps = generator._generate_next_steps(mock_drift)
        assert len(steps) >= 5
        assert any("normalizer" in s.lower() or "distribution" in s.lower() for s in steps)

    def test_next_steps_always_include_slo_check(self, generator) -> None:
        for signal in ["INPUT_DRIFT", "REWARD_DEGRADATION", "BOTH"]:
            mock_drift = type(
                "DriftEvent",
                (),
                {
                    "drift_signal": signal,
                    "retraining_job_id": None,
                },
            )()
            steps = generator._generate_next_steps(mock_drift)
            assert any("slo" in s.lower() or "health" in s.lower() for s in steps), (
                f"Next steps for {signal} missing SLO check"
            )


@pytest.mark.unit
class TestTrustScoreComponents:
    """Verify trust score component calculations."""

    @pytest.fixture
    def computer(self):
        from app.trust.policy_trust_score import TrustScoreComputer

        return TrustScoreComputer(window_hours=1)

    def test_make_recommendation_promote(self, computer) -> None:
        assert computer._make_recommendation(0.9) == "PROMOTE"
        assert computer._make_recommendation(0.81) == "PROMOTE"

    def test_make_recommendation_monitor(self, computer) -> None:
        assert computer._make_recommendation(0.79) == "MONITOR"
        assert computer._make_recommendation(0.51) == "MONITOR"

    def test_make_recommendation_warn(self, computer) -> None:
        assert computer._make_recommendation(0.49) == "WARN"
        assert computer._make_recommendation(0.25) == "WARN"

    def test_make_recommendation_rollback(self, computer) -> None:
        assert computer._make_recommendation(0.19) == "ROLLBACK"
        assert computer._make_recommendation(0.0) == "ROLLBACK"

    def test_recency_score_fresh_data(self, computer) -> None:
        from datetime import UTC, datetime, timedelta

        now = datetime.now(UTC)
        recent = type("Log", (), {"created_at": now - timedelta(seconds=30)})()
        score = computer._recency_score([recent], now)
        assert score > 0.95, "Very recent data should give near-perfect recency score"

    def test_recency_score_stale_data(self, computer) -> None:
        from datetime import UTC, datetime, timedelta

        now = datetime.now(UTC)
        stale = type("Log", (), {"created_at": now - timedelta(hours=1)})()
        score = computer._recency_score([stale], now)
        assert score < 0.05, "One-hour-old data should give near-zero recency score"

    def test_recency_score_no_data(self, computer) -> None:
        from datetime import UTC, datetime

        score = computer._recency_score([], datetime.now(UTC))
        assert score == 0.0

    def test_no_data_score_returns_neutral(self, computer) -> None:
        from datetime import UTC, datetime

        from app.schemas.common import PolicyType

        score = computer._no_data_score(PolicyType.RL, datetime.now(UTC))
        assert score.composite_score == 0.5
        assert score.recommendation == "MONITOR"
        assert score.n_decisions == 0
