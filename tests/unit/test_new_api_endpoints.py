"""
API endpoint tests for Phases 10-17.

Tests validation-only paths that don't need DB or Redis.
Integration tests for DB-dependent paths are in tests/integration/.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.unit
class TestAuthEndpoints:
    """Verify auth API endpoints."""

    @pytest.mark.asyncio
    async def test_login_dev_mode(self, client: AsyncClient) -> None:
        """In dev mode, any credentials return a token."""
        response = await client.post(
            "/api/v1/auth/token",
            json={"username": "test@test.com", "password": "test", "role": "operator"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["role"] == "operator"

    @pytest.mark.asyncio
    async def test_login_returns_bearer_token(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/auth/token",
            json={"username": "user@test.com", "password": "pw", "role": "viewer"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["token_type"] == "bearer"
        assert data["expires_in_minutes"] > 0

    @pytest.mark.asyncio
    async def test_me_endpoint_dev_mode(self, client: AsyncClient) -> None:
        """In dev mode, /auth/me returns ADMIN role."""
        response = await client.get("/api/v1/auth/me")
        assert response.status_code == 200
        data = response.json()
        assert data["role"] == "admin"  # dev mode returns ADMIN

    @pytest.mark.asyncio
    async def test_invalid_role_defaults_to_viewer(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/auth/token",
            json={"username": "x", "password": "y", "role": "invalid_role"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["role"] == "viewer"  # falls back to VIEWER for invalid roles


@pytest.mark.unit
class TestOperatorEndpoints:
    """Verify operator API validation (dev mode passes auth)."""

    @pytest.mark.asyncio
    async def test_kill_switch_requires_reason(self, client: AsyncClient) -> None:
        """POST without reason body should return 422."""
        response = await client.post(
            "/api/v1/operator/kill-switch/activate",
            json={},  # missing reason
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_force_baseline_requires_reason(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/operator/override/force-baseline",
            json={},
        )
        assert response.status_code == 422


@pytest.mark.unit
class TestCanaryEndpoints:
    """Verify canary API validation."""

    @pytest.mark.asyncio
    async def test_start_canary_requires_version_id(self, client: AsyncClient) -> None:
        """Starting canary without version_id should return 422."""
        response = await client.post(
            "/api/v1/canary/RL/start",
            json={},  # missing version_id
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_policy_type_returns_422(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/canary/INVALID/status")
        assert response.status_code == 422


@pytest.mark.unit
class TestBacktestEndpoints:
    """Verify backtest API validation."""

    @pytest.mark.asyncio
    async def test_counterfactual_invalid_delta(self, client: AsyncClient) -> None:
        """Delta outside [-3, 3] should return 422."""
        import uuid

        response = await client.post(
            f"/api/v1/backtest/counterfactual/{uuid.uuid4()}",
            json={"counterfactual_delta": 10},  # outside [-3, 3]
        )
        assert response.status_code == 422


@pytest.mark.unit
class TestTrustEndpoints:
    """Verify trust score API."""

    @pytest.mark.asyncio
    async def test_invalid_policy_type_returns_422(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/trust/INVALID_POLICY")
        assert response.status_code == 422


@pytest.mark.unit
class TestCostAndInfraEndpoints:
    """Verify cost and infrastructure endpoints."""


@pytest.mark.unit
class TestAuditEndpoints:
    """Verify audit API validation."""

    @pytest.mark.asyncio
    async def test_replay_invalid_uuid(self, client: AsyncClient) -> None:
        """Non-UUID ID should return 422."""
        response = await client.get("/api/v1/audit/decisions/not-a-uuid/replay")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_timeline_window_hours_validation(self, client: AsyncClient) -> None:
        """window_hours > 48 should return 422."""
        response = await client.get("/api/v1/audit/incidents/timeline?window_hours=100")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_timeline_min_window(self, client: AsyncClient) -> None:
        """window_hours=0 should return 422."""
        response = await client.get("/api/v1/audit/incidents/timeline?window_hours=0")
        assert response.status_code == 422
