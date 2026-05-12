"""
Exception hierarchy tests.

Verifies:
- All custom exceptions inherit from RTDEBaseError
- Exception messages are preserved
"""

from __future__ import annotations

import pytest

from app.core.exceptions import (
    CheckpointError,
    DriftDetectedError,
    LockUnavailableError,
    PolicyError,
    PolicyNotFoundError,
    RTDEBaseError,
    StateValidationError,
    StateVersionConflictError,
)


@pytest.mark.unit
class TestExceptions:
    """Verify exception hierarchy and message propagation."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            StateValidationError,
            LockUnavailableError,
            StateVersionConflictError,
            PolicyError,
            PolicyNotFoundError,
            DriftDetectedError,
            CheckpointError,
        ],
    )
    def test_inherits_from_base(self, exc_class: type[RTDEBaseError]) -> None:
        """All RTDE exceptions must inherit from RTDEBaseError."""
        exc = exc_class("test message")
        assert isinstance(exc, RTDEBaseError)

    def test_message_preserved(self) -> None:
        """Exception message must be accessible."""
        exc = StateValidationError("cpu_pct out of range")
        assert exc.message == "cpu_pct out of range"
        assert str(exc) == "cpu_pct out of range"

    def test_default_message(self) -> None:
        """Base error should have a default message."""
        exc = RTDEBaseError()
        assert exc.message == "An internal error occurred"
