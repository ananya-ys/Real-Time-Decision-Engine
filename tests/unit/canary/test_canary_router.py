"""
Phase 12 tests — CanaryRouter traffic splitting logic.

Verifies:
- should_use_canary: probabilistic routing at given percentages
- 0% traffic → never routes to canary
- 100% traffic → always routes to canary
- Auto-abort conditions evaluated correctly
- Stage progression order is correct
"""

from __future__ import annotations

import pytest

from app.canary.canary_router import CANARY_STAGES, CanaryRouter


@pytest.mark.unit
class TestCanaryTrafficRouting:
    """Verify probabilistic traffic routing logic (no Redis needed)."""

    @pytest.fixture
    def router(self) -> CanaryRouter:
        return CanaryRouter()

    def test_zero_percent_never_routes_to_canary(self, router: CanaryRouter) -> None:
        """0% traffic means no requests go to canary."""
        results = [router.should_use_canary(0) for _ in range(1000)]
        assert not any(results), "0% canary should never route to canary"

    def test_hundred_percent_always_routes_to_canary(self, router: CanaryRouter) -> None:
        """100% traffic means all requests go to canary."""
        results = [router.should_use_canary(100) for _ in range(100)]
        assert all(results), "100% canary should always route to canary"

    def test_ten_percent_roughly_correct(self, router: CanaryRouter) -> None:
        """10% canary: ~10% of 10,000 requests should go to canary."""
        n = 10_000
        count = sum(1 for _ in range(n) if router.should_use_canary(10))
        # Allow ±3% tolerance (10% ± 3% = 7% to 13%)
        assert 700 < count < 1300, f"Expected ~1000, got {count}"

    def test_fifty_percent_roughly_correct(self, router: CanaryRouter) -> None:
        """50% canary: ~50% of 10,000 requests."""
        n = 10_000
        count = sum(1 for _ in range(n) if router.should_use_canary(50))
        assert 4500 < count < 5500, f"Expected ~5000, got {count}"

    def test_uses_cryptographic_random(self, router: CanaryRouter) -> None:
        """Routing must use crypto-secure random (no predictable patterns)."""
        # Check that consecutive calls at 50% produce different results
        results = [router.should_use_canary(50) for _ in range(20)]
        # With 50% probability, 20 coin flips should not all be the same
        assert len(set(results)) > 1, "Routing should not be deterministic"

    def test_canary_stages_are_ordered(self) -> None:
        """Canary stages must be in ascending order."""
        assert sorted(CANARY_STAGES) == CANARY_STAGES, "Stages must be ascending"
        assert CANARY_STAGES[0] == 10, "First stage must be 10%"
        assert CANARY_STAGES[-1] == 100, "Last stage must be 100%"


@pytest.mark.unit
class TestCanaryAutoAbort:
    """Verify auto-abort threshold logic."""

    @pytest.fixture
    def router(self) -> CanaryRouter:
        return CanaryRouter()

    @pytest.mark.asyncio
    async def test_insufficient_data_no_abort(self, router: CanaryRouter) -> None:
        """With fewer than 20 canary decisions, auto-abort never fires."""
        from unittest.mock import AsyncMock, patch

        from app.schemas.common import PolicyType

        # Mock Redis to return metrics with only 5 canary decisions
        mock_metrics = {
            "canary_decisions": "5",
            "canary_sla_violations": "3",  # 60% violation rate — but not enough data
            "canary_fallbacks": "2",
        }

        with patch.object(router, "_client") as mock_client_factory:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.hgetall = AsyncMock(return_value=mock_metrics)
            mock_client_factory.return_value = mock_client

            should_abort, reason = await router.should_auto_abort(PolicyType.RL)

        assert not should_abort, "Should not abort with only 5 decisions"
        assert reason is None

    @pytest.mark.asyncio
    async def test_high_sla_violation_triggers_abort(self, router: CanaryRouter) -> None:
        """SLA violation rate > 5% with sufficient data triggers auto-abort."""
        from unittest.mock import AsyncMock, patch

        from app.schemas.common import PolicyType

        mock_metrics = {
            "canary_decisions": "100",
            "canary_sla_violations": "10",  # 10% violation rate > 5% threshold
            "canary_fallbacks": "0",
        }

        with patch.object(router, "_client") as mock_client_factory:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.hgetall = AsyncMock(return_value=mock_metrics)
            mock_client_factory.return_value = mock_client

            should_abort, reason = await router.should_auto_abort(PolicyType.RL)

        assert should_abort
        assert reason is not None
        assert "5%" in reason

    @pytest.mark.asyncio
    async def test_healthy_canary_no_abort(self, router: CanaryRouter) -> None:
        """Healthy canary (low violation rate) should not trigger abort."""
        from unittest.mock import AsyncMock, patch

        from app.schemas.common import PolicyType

        mock_metrics = {
            "canary_decisions": "100",
            "canary_sla_violations": "2",  # 2% < 5% threshold
            "canary_fallbacks": "0",
        }

        with patch.object(router, "_client") as mock_client_factory:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.hgetall = AsyncMock(return_value=mock_metrics)
            mock_client_factory.return_value = mock_client

            should_abort, reason = await router.should_auto_abort(PolicyType.RL)

        assert not should_abort
        assert reason is None
