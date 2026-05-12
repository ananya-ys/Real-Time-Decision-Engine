# RTDE Deployment Guide

**Production-Grade ML Decision Engine** — deploy in 15 minutes.

---

## Prerequisites

- Python 3.12 (local dev only)
- Docker (local dev + CI builds)
- Git + GitHub account
- One of: Railway account, Render account, or VPS

---

## Step 1: GitHub — Push Code

```bash
cd rtde_backend/

# Initialize git repository
git init
git add .
git commit -m "feat: RTDE Phase 0-8 complete

- Three-tier policy: Baseline → Bandit → RL (DQN)
- Two-signal drift detection (PSI + Mann-Whitney)
- 5-step atomic rollback with Celery retraining signal
- Shadow mode evaluation gate (≥5 seeds)
- 86% test coverage, 342 tests
- Full Prometheus observability"

# Connect to GitHub
git remote add origin https://github.com/YOUR_USERNAME/rtde-backend.git

# Push to main (CI pipeline triggers automatically)
git branch -M main
git push -u origin main
```

**CI pipeline** (`.github/workflows/ci.yml`) runs:
1. `ruff check` — lint
2. `ruff format --check` — format
3. `pytest tests/unit/ tests/concurrency/ tests/chaos/` — unit tests
4. `pytest tests/integration/` — integration tests (with PostgreSQL service)
5. `docker build` — verify image builds

---

## Step 2: Supabase — PostgreSQL Database

1. Go to [supabase.com](https://supabase.com) → New project
2. Note your project password and project reference ID
3. Go to **Settings → Database → Connection string (URI)**
4. Copy the connection string — it looks like:
   ```
   postgresql://postgres:[password]@db.[ref].supabase.co:5432/postgres
   ```
5. **Modify it** to use asyncpg:
   ```
   postgresql+asyncpg://postgres:[password]@db.[ref].supabase.co:5432/postgres?ssl=require
   ```
6. This becomes your `DATABASE_URL`

**Run migrations** (first time only):
```bash
DATABASE_URL="postgresql+asyncpg://..." alembic upgrade head
```

---

## Step 3A: Railway (Recommended)

### Add Services

1. Go to [railway.app](https://railway.app) → New Project
2. **Deploy from GitHub repo** → select `rtde-backend`
3. Railway auto-detects `railway.toml` — uses Dockerfile

### Add Redis

1. In your Railway project → **New** → Redis
2. Note the connection string from **Variables** tab

### Set Environment Variables

In Railway dashboard → select your service → **Variables** tab:

| Variable | Value |
|----------|-------|
| `APP_ENV` | `production` |
| `SECRET_KEY` | `$(openssl rand -hex 32)` |
| `DATABASE_URL` | From Supabase Step 2 |
| `REDIS_URL` | From Railway Redis |
| `CELERY_BROKER_URL` | Same as REDIS_URL (append `/1`) |
| `CELERY_RESULT_BACKEND` | Same as REDIS_URL (append `/2`) |
| `LOG_LEVEL` | `INFO` |
| `POSTGRES_PASSWORD` | Same password as DATABASE_URL |

### Add Celery Worker (optional but recommended)

1. Railway project → **New Service** → Same GitHub repo
2. Override start command: `celery -A app.worker.celery_app worker --loglevel=info`
3. Add same environment variables

### Verify Deployment

```bash
curl https://your-app.up.railway.app/health
```

Expected:
```json
{"status": "ok", "db": "ok", "redis": "ok"}
```

---

## Step 3B: Render

1. Go to [render.com](https://render.com) → New → **Blueprint**
2. Connect GitHub repo — Render detects `render.yaml`
3. Render creates: API service + Celery worker + Celery beat + PostgreSQL
4. Set secret variables in each service:
   - `SECRET_KEY` → generate in Render dashboard
   - `DATABASE_URL` → from Supabase (or use Render's PostgreSQL)

### Render-specific note:

`render.yaml` references `fromDatabase` for DATABASE_URL. If using Supabase instead:
- Remove the `fromDatabase` reference
- Set `DATABASE_URL` manually as an environment variable

---

## Step 4: Run Database Migrations

Railway and Render run migrations automatically via start command:
```
alembic upgrade head && uvicorn app.main:app ...
```

For manual migration (e.g., after schema change):
```bash
# Railway CLI
railway run alembic upgrade head

# Render shell
# Open Render dashboard → Service → Shell tab
alembic upgrade head
```

---

## Step 5: Verify Production Deployment

```bash
BASE_URL="https://your-app.railway.app"

# Health check
curl $BASE_URL/health | jq .

# Prometheus metrics (raw)
curl $BASE_URL/metrics | head -30

# Dashboard (requires DB)
curl $BASE_URL/api/v1/monitoring/dashboard | jq .system_status

# SLO health
curl $BASE_URL/api/v1/monitoring/health/slo | jq .overall_passing

# Make a test decision
curl -X POST $BASE_URL/api/v1/decision \
  -H "Content-Type: application/json" \
  -d '{
    "state": {
      "cpu_utilization": 0.85,
      "request_rate": 2000.0,
      "p99_latency_ms": 300.0,
      "instance_count": 5
    }
  }' | jq .
```

---

## Step 6: Configure Prometheus + Grafana (optional)

Point Prometheus scrape config at `/metrics`:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: rtde
    static_configs:
      - targets: ["your-app.railway.app:80"]
    metrics_path: /metrics
    scheme: https
```

**Key metrics to alert on:**
```promql
# P99 decision latency (SLO: < 300ms)
histogram_quantile(0.99, rate(rtde_decision_latency_ms_bucket[5m]))

# Rollback rate
rate(rtde_rollbacks_total[1h])

# Fallback rate (policy exceptions)
rate(rtde_fallback_total[5m]) / rate(rtde_decisions_total[5m])

# SLA violation rate
rate(rtde_sla_violations_total[5m]) / rate(rtde_decisions_total[5m])
```

---

## Environment Variables Reference

All variables documented in `.env.production.example`.

**Minimum required for deployment:**
```
SECRET_KEY          # JWT signing key
DATABASE_URL        # PostgreSQL connection string (asyncpg)
REDIS_URL           # Redis connection string
CELERY_BROKER_URL   # Redis for task queue (different DB number)
CELERY_RESULT_BACKEND # Redis for task results
APP_ENV=production
```

**Optional performance tuning:**
```
WEB_CONCURRENCY     # Number of uvicorn workers (default: 2*CPU+1)
DB_POOL_SIZE        # PostgreSQL connection pool size (default: 10)
RL_WARM_START_MIN_DECISIONS  # Bandit decisions before RL activates (default: 1000)
DRIFT_HYSTERESIS_K  # Consecutive windows before rollback (default: 3)
```

---

## Troubleshooting

### App starts but returns 503

Check DB connectivity:
```bash
curl https://your-app.railway.app/health
# Look at "db" and "redis" fields
```

### Celery worker not processing tasks

Check broker URL:
```bash
# Via Railway CLI
railway run celery -A app.worker.celery_app inspect active
```

### Database migration fails

Run migration manually:
```bash
DATABASE_URL="your-url" alembic upgrade head --sql  # preview SQL
DATABASE_URL="your-url" alembic upgrade head         # run
```

### RL policy not activating

RL requires `RL_WARM_START_MIN_DECISIONS` (default 1000) Bandit decisions first.
Check via:
```bash
curl https://your-app.railway.app/api/v1/policies/active
```

---

## Local Development

```bash
# Start all services
docker compose up -d

# Wait for health
sleep 5 && curl http://localhost:8000/health

# Run migrations
make migrate

# Run tests
make test

# Lint + format
make lint && make format
```
