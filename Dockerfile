
# ══════════════════════════════════════════════════════════════════
# RTDE — Production Dockerfile
# ══════════════════════════════════════════════════════════════════

FROM python:3.12-slim AS builder

WORKDIR /build

# Needed for asyncpg / psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ──────────────────────────────────────────────────────────────────

FROM python:3.12-slim AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy source code
COPY . /app

# Create non-root user
RUN groupadd -r rtde && useradd -r -g rtde -u 1000 -m rtde

# Give permissions
RUN chown -R rtde:rtde /app

EXPOSE 8000

# Run migrations first, then start server
CMD alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000

# Switch to non-root user AFTER setup
USER rtde
