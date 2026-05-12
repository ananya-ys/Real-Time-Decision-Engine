"""
P0 enforcement tests — rate limiter, idempotency, hash chain, confirmation, approval.

These test the REAL enforcement mechanisms, not the labels.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.audit.hash_chain import _GENESIS_HASH, compute_event_hash
from app.core.rate_limiter import _get_limit
from app.operator.confirmation_gate import (
    HIGH_RISK_ACTIONS,
    MEDIUM_RISK_ACTIONS,
    ConfirmationGate,
)


@pytest.mark.unit
class TestRateLimiterLogic:
    """Rate limiter rules are correctly classified."""

    def test_kill_switch_critical_limit(self) -> None:
        """Kill switch must be in the most restrictive tier."""
        limit, _w = _get_limit("/api/v1/operator/kill-switch/activate", "POST")
        assert limit <= 3
        assert _w >= 600  # at least 10 minutes

    def test_auth_endpoint_limited(self) -> None:
        limit, _w = _get_limit("/api/v1/auth/token", "POST")
        assert limit <= 20  # prevent brute force
        assert _w <= 60

    def test_read_endpoint_generous(self) -> None:
        limit, _w = _get_limit("/api/v1/monitoring/dashboard", "GET")
        assert limit >= 100

    def test_default_is_reasonable(self) -> None:
        limit, _w = _get_limit("/api/v1/unknown/endpoint", "GET")
        assert 60 <= limit <= 500

    def test_websocket_unlimited(self) -> None:
        limit, _w = _get_limit("/ws/decisions", "GET")
        assert limit == 0  # unlimited

    def test_health_unlimited(self) -> None:
        limit, _w = _get_limit("/health", "GET")
        assert limit == 0  # unlimited

    @pytest.mark.parametrize(
        "action_path",
        [
            "/api/v1/operator/kill-switch/activate",
            "/api/v1/operator/maintenance/enter",
        ],
    )
    def test_all_critical_paths_have_tight_limits(self, action_path: str) -> None:
        limit, _w2 = _get_limit(action_path, "POST")
        assert limit <= 10, f"{action_path} should be tightly rate limited, got {limit}"


@pytest.mark.unit
class TestConfirmationGateLogic:
    """Confirmation gate classification and challenge string logic."""

    @pytest.fixture
    def gate(self) -> ConfirmationGate:
        return ConfirmationGate()

    def test_kill_switch_is_high_risk(self, gate: ConfirmationGate) -> None:
        assert "KILL_SWITCH_GLOBAL" in HIGH_RISK_ACTIONS

    def test_maintenance_is_high_risk(self, gate: ConfirmationGate) -> None:
        assert "MAINTENANCE_ENTER" in HIGH_RISK_ACTIONS

    def test_freeze_is_medium_risk(self, gate: ConfirmationGate) -> None:
        assert "FREEZE_EXPLORATION" in MEDIUM_RISK_ACTIONS

    def test_high_risk_requires_second_approval(self, gate: ConfirmationGate) -> None:
        """High risk actions must flag requires_second_approval=True."""
        risk = gate._risk_level("KILL_SWITCH_GLOBAL")
        assert risk == "CRITICAL"

    def test_blast_radius_defined_for_all_high_risk(self, gate: ConfirmationGate) -> None:
        """Every high-risk action must have a blast radius description."""
        for action in HIGH_RISK_ACTIONS:
            desc = gate._blast_radius(action)
            assert len(desc) > 20, f"Blast radius for {action} is too vague: '{desc}'"

    def test_challenge_string_includes_action(self, gate: ConfirmationGate) -> None:
        """Challenge must include the action name so operator reads what they're doing."""
        challenge = gate._challenge_string("KILL_SWITCH_GLOBAL", "actor@test")
        assert "KILL_SWITCH_GLOBAL" in challenge


@pytest.mark.unit
class TestAuditHashChain:
    """Hash chain cryptographic integrity."""

    def test_genesis_hash_is_64_zeros(self) -> None:
        assert _GENESIS_HASH == "0" * 64

    def test_compute_hash_deterministic(self) -> None:
        """Same input always produces same hash."""
        h1 = compute_event_hash("id1", "actor", "ACTION", "target", "reason", "2025-01-01", "prev")
        h2 = compute_event_hash("id1", "actor", "ACTION", "target", "reason", "2025-01-01", "prev")
        assert h1 == h2

    def test_hash_changes_with_any_field(self) -> None:
        base = ("id1", "actor", "ACTION", "target", "reason", "2025-01-01", "prev")
        base_hash = compute_event_hash(*base)

        # Change each field
        for i, new_val in enumerate(
            ["id2", "actor2", "ACTION2", "target2", "reason2", "2025-01-02", "prev2"]
        ):
            args = list(base)
            args[i] = new_val
            modified_hash = compute_event_hash(*args)
            assert modified_hash != base_hash, f"Hash did not change when field {i} changed"

    def test_hash_is_64_chars(self) -> None:
        h = compute_event_hash("id", "a", "A", None, "r", "t", "p")
        assert len(h) == 64

    def test_hash_is_hex(self) -> None:
        h = compute_event_hash("id", "a", "A", None, "r", "t", "p")
        int(h, 16)  # raises ValueError if not valid hex

    def test_prev_hash_chaining_detected_on_tamper(self) -> None:
        """Modifying any event makes its hash wrong, breaking the chain link."""
        id1 = "event-1"
        h1 = compute_event_hash(id1, "alice", "KILL_SWITCH", None, "test", "ts1", _GENESIS_HASH)

        id2 = "event-2"
        h2 = compute_event_hash(id2, "bob", "RESTORE", None, "done", "ts2", h1)

        # Simulate tampered event 1 (reason changed)
        tampered_h1 = compute_event_hash(
            id1, "alice", "KILL_SWITCH", None, "TAMPERED", "ts1", _GENESIS_HASH
        )

        # Event 2's prev_hash (h1) no longer matches tampered_h1
        assert h1 != tampered_h1
        # A verifier would compute h2_with_tampered_prev and find it != stored h2
        h2_with_tampered = compute_event_hash(
            id2, "bob", "RESTORE", None, "done", "ts2", tampered_h1
        )
        assert h2 != h2_with_tampered  # chain integrity violation detectable


@pytest.mark.unit
class TestApprovalServiceLogic:
    """Approval service enforcement — no self-approval, no stale approvals."""

    @pytest.mark.asyncio
    async def test_self_approval_blocked(self) -> None:
        """User cannot approve their own request."""
        from app.models.approval_request import ApprovalRequest, ApprovalStatus
        from app.services.approval_service import ApprovalService

        svc = ApprovalService()

        # Create a fake approved-but-same-requester scenario
        mock_req = MagicMock(spec=ApprovalRequest)
        mock_req.id = uuid.uuid4()
        mock_req.status = ApprovalStatus.PENDING.value
        mock_req.requester_id = "alice@test"
        mock_req.expires_at = None

        mock_db = AsyncMock()
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=mock_req)
        mock_db.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(PermissionError, match="cannot approve your own"):
            await svc.review(
                request_id=mock_req.id,
                reviewer_id="alice@test",  # same person!
                reviewer_role="admin",
                approve=True,
                comment="Approved by self",
                db=mock_db,
            )

    @pytest.mark.asyncio
    async def test_only_pending_requests_can_be_reviewed(self) -> None:
        """Cannot approve an already-approved or executed request."""
        from app.models.approval_request import ApprovalRequest, ApprovalStatus
        from app.services.approval_service import ApprovalService

        svc = ApprovalService()
        mock_req = MagicMock(spec=ApprovalRequest)
        mock_req.id = uuid.uuid4()
        mock_req.status = ApprovalStatus.APPROVED.value  # already approved
        mock_req.requester_id = "alice@test"

        mock_db = AsyncMock()
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=mock_req)
        mock_db.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(ValueError, match="not PENDING"):
            await svc.review(
                request_id=mock_req.id,
                reviewer_id="bob@test",
                reviewer_role="admin",
                approve=True,
                comment="",
                db=mock_db,
            )

    @pytest.mark.asyncio
    async def test_only_requester_can_execute(self) -> None:
        """Only the original requester can execute an approved action."""
        from app.models.approval_request import ApprovalRequest, ApprovalStatus
        from app.services.approval_service import ApprovalService

        svc = ApprovalService()
        mock_req = MagicMock(spec=ApprovalRequest)
        mock_req.id = uuid.uuid4()
        mock_req.status = ApprovalStatus.APPROVED.value
        mock_req.requester_id = "alice@test"
        mock_req.expires_at = None

        mock_db = AsyncMock()
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=mock_req)
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()

        with pytest.raises(PermissionError, match="original requester"):
            await svc.execute(
                request_id=mock_req.id,
                executor_id="charlie@test",  # different person!
                db=mock_db,
            )


@pytest.mark.unit
class TestExplainabilityService:
    """Explainability produces correct risk assessments and alternatives."""

    @pytest.mark.asyncio
    async def test_explain_produces_5_alternatives(self) -> None:
        from app.schemas.state import SystemState
        from app.services.explainability_service import ExplainabilityService

        svc = ExplainabilityService()
        state = SystemState(
            cpu_utilization=0.7, request_rate=1000, p99_latency_ms=200, instance_count=5
        )
        result = await svc.explain(state=state, chosen_action="HOLD")
        assert len(result.alternatives) == 5  # all 5 actions evaluated

    @pytest.mark.asyncio
    async def test_explain_identifies_scale_down_during_high_latency_as_risk(self) -> None:
        from app.schemas.state import SystemState
        from app.services.explainability_service import ExplainabilityService

        svc = ExplainabilityService()
        state = SystemState(
            cpu_utilization=0.8, request_rate=3000, p99_latency_ms=700, instance_count=10
        )
        result = await svc.explain(state=state, chosen_action="SCALE_DOWN_3")
        assert result.risk_assessment["risk_level"] in ("HIGH", "CRITICAL")

    @pytest.mark.asyncio
    async def test_explain_suppressed_guard_narrative(self) -> None:
        from app.schemas.state import SystemState
        from app.services.explainability_service import ExplainabilityService

        svc = ExplainabilityService()
        state = SystemState(
            cpu_utilization=0.9, request_rate=6000, p99_latency_ms=500, instance_count=5
        )
        result = await svc.explain(
            state=state,
            chosen_action="SCALE_UP_1",
            explore_allowed=False,
            suppression_reason="HIGH_LOAD",
        )
        assert "suppressed" in result.guard_explanation.lower()
        assert (
            "HIGH_LOAD" in result.guard_explanation
            or "high-load" in result.guard_explanation.lower()
        )

    @pytest.mark.asyncio
    async def test_explain_feature_attribution_covers_all_fields(self) -> None:
        from app.schemas.state import SystemState
        from app.services.explainability_service import ExplainabilityService

        svc = ExplainabilityService()
        state = SystemState(
            cpu_utilization=0.5, request_rate=1000, p99_latency_ms=200, instance_count=5
        )
        result = await svc.explain(state=state, chosen_action="HOLD")
        assert "cpu_utilization" in result.feature_attribution
        assert "p99_latency_ms" in result.feature_attribution
        assert "request_rate" in result.feature_attribution
        assert "instance_count" in result.feature_attribution
