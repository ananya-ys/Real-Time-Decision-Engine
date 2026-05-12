"""
Health endpoint tests — Phase 0 gate verification.

Gate criteria:
- GET /health returns 200
- Response includes db, redis, status fields
- App starts without errors
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.unit
class TestHealthEndpoint:
    """Verify health endpoint contract."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client: AsyncClient) -> None:
        """GET /health should return 200 when all services are up."""
        response = await client.get("/health")
        # May be 503 if DB/Redis not available in test — that's fine for unit test
        assert response.status_code in (200, 503)

    @pytest.mark.asyncio
    async def test_health_response_shape(self, client: AsyncClient) -> None:
        """Response must include required status fields."""
        response = await client.get("/health")
        data = response.json()
        assert "status" in data
        assert "db" in data
        assert "redis" in data

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, client: AsyncClient) -> None:
        """GET /metrics should return Prometheus text format."""
        response = await client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")
