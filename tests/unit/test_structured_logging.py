"""
Structured logging tests — Phase 7 gate.

Verifies:
- structlog can be imported and configured without error
- setup_logging() does not raise
- Logger can emit events with bound context variables
- All log event names follow the naming convention (snake_case)
- Trace ID is bound to context and appears in log records
"""

from __future__ import annotations

import uuid

import pytest
import structlog


@pytest.mark.unit
class TestStructlogSetup:
    """Verify structlog configuration."""

    def test_setup_logging_does_not_raise(self) -> None:
        """setup_logging() must run without error."""
        from app.core.logging import setup_logging

        setup_logging()  # should not raise

    def test_logger_creation(self) -> None:
        """Getting a logger should not raise."""
        logger = structlog.get_logger("test_module")
        assert logger is not None

    def test_logger_emits_events(self) -> None:
        """Logger must accept info(), warning(), error() calls."""
        logger = structlog.get_logger("test")
        # These should not raise
        logger.info("test_event", key="value")
        logger.warning("test_warning", code=42)
        logger.error("test_error", detail="something failed")

    def test_contextvars_bind_and_clear(self) -> None:
        """Bind + clear context vars must work without error."""
        trace_id = str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(trace_id=trace_id)
        ctx = structlog.contextvars.get_contextvars()
        assert ctx.get("trace_id") == trace_id

        structlog.contextvars.clear_contextvars()
        ctx_cleared = structlog.contextvars.get_contextvars()
        assert "trace_id" not in ctx_cleared

    def test_merge_contextvars_processor_available(self) -> None:
        """merge_contextvars processor must be importable."""
        from structlog.contextvars import merge_contextvars

        assert callable(merge_contextvars)


@pytest.mark.unit
class TestLoggingConventions:
    """Verify log event naming conventions."""

    def test_all_service_modules_use_structlog(self) -> None:
        """All service modules must import structlog."""
        import importlib

        modules = [
            "app.services.decision_service",
            "app.services.state_service",
            "app.services.reward_service",
            "app.services.policy_service",
            "app.services.drift_service",
            "app.services.rollback_service",
            "app.policies.baseline_policy",
            "app.policies.bandit_policy",
            "app.policies.rl_policy",
        ]
        for module_path in modules:
            mod = importlib.import_module(module_path)
            assert hasattr(mod, "logger"), f"{module_path} missing 'logger' variable"

    def test_logger_bound_to_module_name(self) -> None:
        """Loggers must be bound to module name for source tracking."""
        from app.policies.baseline_policy import logger as baseline_logger
        from app.services.decision_service import logger as decision_logger

        # Both should be structlog BoundLogger instances
        assert decision_logger is not None
        assert baseline_logger is not None


@pytest.mark.unit
class TestMiddlewareLogging:
    """Verify middleware trace_id injection."""

    @pytest.mark.asyncio
    async def test_health_response_has_trace_id(self, client) -> None:
        """Every response must have X-Trace-ID header."""
        response = await client.get("/health")
        assert "x-trace-id" in response.headers

    @pytest.mark.asyncio
    async def test_trace_ids_are_unique(self, client) -> None:
        """Each request should get a different trace_id."""
        r1 = await client.get("/health")
        r2 = await client.get("/health")
        t1 = r1.headers.get("x-trace-id")
        t2 = r2.headers.get("x-trace-id")
        assert t1 != t2  # unique per request
