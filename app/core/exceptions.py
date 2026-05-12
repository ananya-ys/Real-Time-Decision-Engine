"""
Custom exception hierarchy for RTDE.

WHY THIS EXISTS:
- Named exceptions map to specific HTTP status codes. Generic exceptions = unhelpful 500s.
- Service layer raises domain exceptions. Error handlers translate to HTTP responses.
- Keeps error handling centralized and consistent.

WHAT BREAKS IF WRONG:
- Generic Exception → 500 with no useful info for the caller.
- Exception handling scattered across routers → inconsistent error shapes.
"""

from __future__ import annotations


class RTDEBaseError(Exception):
    """Base exception for all RTDE domain errors."""

    def __init__(self, message: str = "An internal error occurred") -> None:
        self.message = message
        super().__init__(self.message)


class StateValidationError(RTDEBaseError):
    """Raised when incoming state fails schema validation."""

    pass


class LockUnavailableError(RTDEBaseError):
    """Raised when SELECT FOR UPDATE NOWAIT cannot acquire the row lock."""

    pass


class StateVersionConflictError(RTDEBaseError):
    """Raised when optimistic concurrency check fails (version mismatch)."""

    pass


class PolicyError(RTDEBaseError):
    """Raised when a policy fails to produce a valid action."""

    pass


class PolicyNotFoundError(RTDEBaseError):
    """Raised when no active policy is found in the registry."""

    pass


class DriftDetectedError(RTDEBaseError):
    """Raised when drift detection identifies policy degradation."""

    pass


class CheckpointError(RTDEBaseError):
    """Raised when checkpoint save/load fails."""

    pass


class ExplorationSuppressedError(RTDEBaseError):
    """Raised when ExplorationGuard blocks exploration (informational, not fatal)."""

    pass
