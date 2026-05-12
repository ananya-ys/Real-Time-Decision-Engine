"""
Request logging middleware — trace_id injection, latency, and Prometheus HTTP metrics.

WHY THIS EXISTS:
- Every request gets a UUID trace_id threaded through all log events.
- Latency measured and logged for SLO tracking.
- Prometheus counters + histograms enable P99 dashboarding.
- One grep on trace_id reconstructs the full request lifecycle.

WHAT BREAKS IF WRONG:
- No trace_id = no way to correlate logs across service layers.
- No Prometheus instrumentation = no alerts on degraded API performance.
"""

from __future__ import annotations

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core import metrics as prom

logger = structlog.get_logger(__name__)

_EXCLUDED_PATHS = {"/metrics", "/health", "/docs", "/redoc", "/openapi.json"}


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with trace_id, method, path, status, and latency.
    Emit Prometheus HTTP metrics for all non-infra endpoints.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        trace_id = str(uuid.uuid4())
        start_time = time.perf_counter()
        path = str(request.url.path)

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            trace_id=trace_id,
            method=request.method,
            path=path,
        )

        request.state.trace_id = trace_id

        response = await call_next(request)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        response.headers["X-Trace-ID"] = trace_id

        log_fn = logger.warning if response.status_code >= 400 else logger.info
        log_fn(
            "request_completed",
            status_code=response.status_code,
            latency_ms=latency_ms,
        )

        if path not in _EXCLUDED_PATHS:
            prom.http_requests_total.labels(
                method=request.method,
                path=path,
                status_code=str(response.status_code),
            ).inc()
            prom.http_request_duration_ms.labels(
                method=request.method,
                path=path,
            ).observe(latency_ms)

        return response
