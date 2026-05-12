"""
Global exception handlers — registered on the FastAPI app.

WHY THIS EXISTS:
- Consistent error response shape across all endpoints.
- Maps domain exceptions to HTTP status codes.
- No raw 500s — every error returns structured JSON.
"""

from __future__ import annotations

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

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

logger = structlog.get_logger(__name__)


def _error_response(status_code: int, error_type: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": error_type, "message": message},
    )


async def state_validation_handler(request: Request, exc: StateValidationError) -> JSONResponse:
    logger.warning("state_validation_error", path=str(request.url), detail=exc.message)
    return _error_response(422, "state_validation_error", exc.message)


async def lock_unavailable_handler(request: Request, exc: LockUnavailableError) -> JSONResponse:
    logger.warning("lock_unavailable", path=str(request.url), detail=exc.message)
    return _error_response(409, "lock_unavailable", exc.message)


async def version_conflict_handler(
    request: Request, exc: StateVersionConflictError
) -> JSONResponse:
    logger.warning("version_conflict", path=str(request.url), detail=exc.message)
    return _error_response(409, "version_conflict", exc.message)


async def policy_error_handler(request: Request, exc: PolicyError) -> JSONResponse:
    logger.error("policy_error", path=str(request.url), detail=exc.message)
    return _error_response(500, "policy_error", exc.message)


async def policy_not_found_handler(request: Request, exc: PolicyNotFoundError) -> JSONResponse:
    logger.error("policy_not_found", path=str(request.url), detail=exc.message)
    return _error_response(404, "policy_not_found", exc.message)


async def drift_detected_handler(request: Request, exc: DriftDetectedError) -> JSONResponse:
    logger.critical("drift_detected", path=str(request.url), detail=exc.message)
    return _error_response(503, "drift_detected", exc.message)


async def checkpoint_error_handler(request: Request, exc: CheckpointError) -> JSONResponse:
    logger.error("checkpoint_error", path=str(request.url), detail=exc.message)
    return _error_response(500, "checkpoint_error", exc.message)


async def generic_rtde_handler(request: Request, exc: RTDEBaseError) -> JSONResponse:
    logger.error("rtde_error", path=str(request.url), detail=exc.message)
    return _error_response(500, "internal_error", exc.message)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_exception", path=str(request.url), error=str(exc))
    return _error_response(500, "internal_server_error", "An unexpected error occurred")


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers on the FastAPI app."""
    app.add_exception_handler(StateValidationError, state_validation_handler)  # type: ignore[arg-type]
    app.add_exception_handler(LockUnavailableError, lock_unavailable_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StateVersionConflictError, version_conflict_handler)  # type: ignore[arg-type]
    app.add_exception_handler(PolicyError, policy_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(PolicyNotFoundError, policy_not_found_handler)  # type: ignore[arg-type]
    app.add_exception_handler(DriftDetectedError, drift_detected_handler)  # type: ignore[arg-type]
    app.add_exception_handler(CheckpointError, checkpoint_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RTDEBaseError, generic_rtde_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)  # type: ignore[arg-type]
