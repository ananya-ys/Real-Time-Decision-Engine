# RTDE — Xenon Control System

**Real-Time Decision Engine** — Production ML system for dynamic cloud resource allocation.
Bandit + DQN policies, 58 API routes, WebSocket live streams, full operator control suite.

[![CI](https://github.com/YOUR_USERNAME/rtde-xenon-control/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/rtde-xenon-control/actions)

---

## 🔴 Live Demo

| Service | URL |
|---------|-----|
| **Frontend (Vercel)** | https://rtde-xenon-control.vercel.app |
| **Backend API (Render)** | https://rtde-api.onrender.com |
| **API Docs (Swagger)** | https://rtde-api.onrender.com/docs |

> ⚠️ Render free tier cold-starts in ~30s after 15 min idle. Normal for demo.

---

## 🚀 Run Locally (WSL + Docker)

**Prerequisites:** WSL2 + Docker Desktop for Windows (with WSL2 backend enabled)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/rtde-xenon-control.git
cd rtde-xenon-control

# 2. Start everything — PostgreSQL + Redis + API + Celery worker
docker compose up --build

# That's it. Services start in this order:
#   postgres  → healthy
#   redis     → healthy
#   migrate   → alembic upgrade head (exits 0)
#   app       → uvicorn on :8000
#   celery-worker → background ML tasks
```

**Open in browser:**
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health
- Frontend: open `index.html` directly in browser (zero build step)

**Useful commands:**
```bash
make logs       # tail API logs
make ps         # show container status
make health     # curl /health and pretty-print
make decision   # fire a test ML decision
make psql       # open postgres shell
make down       # stop everything
make down-clean # stop + wipe database
```

---

## 🎛️ The 8 Panels

| Panel | Features |
|-------|---------|
| **Dashboard** | Live decision stream (WebSocket), policy status, SLO metrics, auto-refresh 4s |
| **Decision Lab** | CPU/RPS/latency/instances sliders → real ML inference → Q-value heatmap |
| **Policies** | Trust score gauges (6-component weighted), policy version registry |
| **Operator** | Kill switch, maintenance mode, force baseline, exploration freeze + audit reason |
| **Canary** | Start/advance/abort canary rollouts, live SLA violation rate, auto-abort |
| **Backtest** | Baseline vs Bandit replay, counterfactual simulator ("what if cpu=0.9?") |
| **Cost & Infra** | Hourly budget gauge, SLO pass/fail, circuit breaker states (CLOSED/OPEN/HALF) |
| **Audit** | Decision time-travel replay, incident timeline builder |

---

## 📐 Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Frontend (Vercel)              Backend (Render / Docker)         │
│  index.html                     FastAPI 0.115 + asyncpg           │
│  React 18 (CDN, no build)  ──► 58 REST endpoints                │
│  8 control panels               2 WebSocket streams               │
│  Dark/light mode                Bandit + DQN policies             │
│  Live API URL switcher          Redis circuit breakers            │
│                                 PostgreSQL 16 (11 tables)         │
│                                 Celery workers (ML training)      │
└──────────────────────────────────────────────────────────────────┘
```

## 🧠 ML Stack

- **ε-greedy Contextual Bandit** — UCB exploration, decaying epsilon (1.0→0.05)
- **DQN (Deep Q-Network)** — pure NumPy implementation, no PyTorch dependency
- **Baseline Policy** — threshold-based rule engine (CPU/latency/RPS)
- **Drift Detection** — PSI + KS-test, hysteresis filter
- **Reward Signal** — latency + cost + SLA violation + instability composite

## 🔑 Auth (Dev Mode)

With `APP_ENV=development` (default), **all requests get ADMIN role automatically** — no token needed. Perfect for local demo and interviewer testing.

To get a JWT anyway:
```bash
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"demo","role":"admin"}'
```

## 🔥 Test the API

```bash
# Health check
curl http://localhost:8000/health

# Make an ML decision
curl -X POST http://localhost:8000/api/v1/decision \
  -H "Content-Type: application/json" \
  -d '{
    "state": {
      "cpu_utilization": 0.82,
      "request_rate": 4200,
      "p99_latency_ms": 450,
      "instance_count": 6,
      "hour_of_day": 14,
      "day_of_week": 2,
      "traffic_regime": "BURST",
      "error_rate": 0.02,
      "memory_utilization": 0.71,
      "active_instances": 6
    }
  }'

# Activate kill switch
curl -X POST http://localhost:8000/api/v1/operator/kill-switch/activate \
  -H "Content-Type: application/json" \
  -d '{"reason": "demo test", "actor": "interviewer"}'

# List policy versions
curl http://localhost:8000/api/v1/policies/versions

# Run a backtest
curl -X POST http://localhost:8000/api/v1/backtesting/quick \
  -H "Content-Type: application/json" \
  -d '{"episodes": 50, "policy_a": "BASELINE", "policy_b": "BANDIT"}'
```

---

## 🌐 Deploy (GitHub → Render + Vercel)

See [`DEPLOY_STEPS.md`](DEPLOY_STEPS.md) for the exact 10-minute deploy walkthrough.

**Quick version:**
1. Push this repo to GitHub
2. Render → New Web Service → connect repo → reads `render.yaml` → free Postgres + API
3. Vercel → Import repo → Framework: Other → Deploy → free frontend CDN
4. Update `window.RTDE_API_URL` in `index.html` with your Render URL → push → done

---

## 🗂️ Project Structure

```
rtde-xenon-control/
├── app/
│   ├── api/v1/          # 15 router files, 58 endpoints
│   ├── core/            # config, database, auth, RBAC, metrics, middleware
│   ├── models/          # 11 SQLAlchemy models
│   ├── schemas/         # Pydantic request/response schemas
│   ├── services/        # business logic layer
│   ├── policies/        # baseline, bandit, RL policy implementations
│   ├── backtesting/     # backtest engine + counterfactual simulator
│   ├── canary/          # canary router + state machine
│   ├── circuit_breaker/ # Redis-backed circuit breakers
│   ├── ml/              # state normalizer, replay buffer
│   ├── operator/        # kill switch, maintenance mode
│   └── worker/          # Celery app + async tasks
├── alembic/             # 3 versioned migrations
├── tests/               # 400+ tests
├── index.html           # frontend (zero-dependency React)
├── docker-compose.yml   # local dev stack
├── Dockerfile           # multi-stage build
├── render.yaml          # Render platform config (free tier)
├── vercel.json          # Vercel static frontend config
└── .github/workflows/   # CI pipeline
```
