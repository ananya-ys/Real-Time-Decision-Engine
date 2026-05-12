"""
Decision API endpoint tests — Phase 2 gate.

Tests that hit /api/v1/decision with valid state need DB → marked integration.
Tests that validate input rejection (422) work without DB.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.unit
class TestDecisionAPIValidation:
    """Verify API validates input at boundary — no DB needed."""

    @pytest.mark.asyncio
    async def test_invalid_cpu_returns_422(self, client: AsyncClient) -> None:
        """cpu > 1.0 must be rejected at API boundary."""
        response = await client.post(
            "/api/v1/decision",
            json={
                "state": {
                    "cpu_utilization": 2.0,
                    "request_rate": 100.0,
                    "p99_latency_ms": 50.0,
                    "instance_count": 3,
                }
            },
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_required_field_returns_422(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/decision",
            json={"state": {"cpu_utilization": 0.5, "request_rate": 100.0, "p99_latency_ms": 50.0}},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_negative_latency_returns_422(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/decision",
            json={
                "state": {
                    "cpu_utilization": 0.5,
                    "request_rate": 100.0,
                    "p99_latency_ms": -10.0,
                    "instance_count": 3,
                }
            },
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_zero_instances_returns_422(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/decision",
            json={
                "state": {
                    "cpu_utilization": 0.5,
                    "request_rate": 100.0,
                    "p99_latency_ms": 50.0,
                    "instance_count": 0,
                }
            },
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_empty_body_returns_422(self, client: AsyncClient) -> None:
        response = await client.post("/api/v1/decision", json={})
        assert response.status_code == 422


@pytest.mark.integration
class TestDecisionAPIIntegration:
    """These tests need a real DB. Marked integration. Skipped without DB."""

    @pytest.mark.asyncio
    async def test_valid_decision_returns_200(self, client: AsyncClient) -> None:
        """Valid state → 200 with scaling decision (needs DB)."""
        response = await client.post(
            "/api/v1/decision",
            json={
                "state": {
                    "cpu_utilization": 0.85,
                    "request_rate": 2000.0,
                    "p99_latency_ms": 300.0,
                    "instance_count": 5,
                }
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "action" in data
        assert "trace_id" in data
        assert "policy_type" in data

    @pytest.mark.asyncio
    async def test_response_has_trace_header(self, client: AsyncClient) -> None:
        """Every response must include X-Trace-ID header (needs DB)."""
        response = await client.post(
            "/api/v1/decision",
            json={
                "state": {
                    "cpu_utilization": 0.5,
                    "request_rate": 100.0,
                    "p99_latency_ms": 50.0,
                    "instance_count": 3,
                }
            },
        )
        assert response.status_code == 200
        assert "x-trace-id" in response.headers
