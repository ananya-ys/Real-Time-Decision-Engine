"""
Gunicorn configuration for production RTDE deployment.

WHY THIS FILE:
- Environment-based worker count.
- UvicornWorker bridges Gunicorn (process management) with Uvicorn (async I/O).
- Proper timeouts prevent zombie workers under load.

USAGE:
  gunicorn app.main:app -c gunicorn.conf.py

ALTERNATIVE (simpler, for Railway/Render with single dyno):
  uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1
"""

import multiprocessing
import os

# ── Workers ─────────────────────────────────────────────────────
# Formula: 2 × CPU cores + 1
# Overridable via WEB_CONCURRENCY env var (Railway/Heroku standard)
workers = int(os.environ.get("WEB_CONCURRENCY", (2 * multiprocessing.cpu_count()) + 1))

# ── Worker class ────────────────────────────────────────────────
worker_class = "uvicorn.workers.UvicornWorker"

# ── Timeouts ────────────────────────────────────────────────────
# Keep-alive for load balancers (Railway, Render use ALBs)
keepalive = 120

# Worker timeout: kill and restart worker if silent for this long
timeout = 120

# Graceful shutdown: wait this long for requests to finish
graceful_timeout = 30

# ── Binding ─────────────────────────────────────────────────────
host = os.environ.get("HOST", "0.0.0.0")
port = int(os.environ.get("PORT", 8000))
bind = f"{host}:{port}"

# ── Logging ─────────────────────────────────────────────────────
# Access log to stdout (captured by platform log aggregators)
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info").lower()

# ── Process naming ───────────────────────────────────────────────
proc_name = "rtde-api"

# ── Worker lifecycle ─────────────────────────────────────────────
# Restart workers after processing this many requests
# Prevents memory leaks from slowly accumulating over time
max_requests = 1000
max_requests_jitter = 50  # random jitter prevents thundering herd on restart
