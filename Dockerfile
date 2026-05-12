# ══════════════════════════════════════════════════════════════════
# RTDE — Multi-stage Dockerfile
# Stage 1: builder  — installs all Python deps
# Stage 2: runtime  — lean image, non-root user
# ══════════════════════════════════════════════════════════════════

FROM python:3.12-slim AS builder

WORKDIR /build
ENV PYTHONPATH=/app
# Build deps for asyncpg + psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ──────────────────────────────────────────────────────────────────

FROM python:3.12-slim AS runtime

# Runtime deps: libpq for asyncpg, curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl && \
    rm -rf /var/lib/apt/lists/*

# Non-root user — security best practice
RUN groupadd -r rtde && useradd -r -g rtde -u 1000 -m rtde

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy app code
COPY --chown=rtde:rtde . .

USER rtde

EXPOSE 8000

# Default: run API server
# Override via docker-compose command: for migrate, celery, etc.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
