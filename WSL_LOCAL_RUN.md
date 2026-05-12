# RTDE — Running Locally on Windows WSL + Docker

## Prerequisites

1. **WSL2** — Windows Subsystem for Linux v2
   ```powershell
   # In PowerShell (admin):
   wsl --install
   wsl --set-default-version 2
   ```

2. **Docker Desktop for Windows**
   - Download: https://www.docker.com/products/docker-desktop
   - Install → Settings → **Resources → WSL Integration** → enable for your distro
   - Confirm: `docker --version` inside WSL terminal

3. **Git** (already in WSL by default)
   ```bash
   git --version   # should show git 2.x
   ```

---

## Quick Start (3 commands)

Open your WSL terminal and run:

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/rtde-xenon-control.git
cd rtde-xenon-control

# 2. Start full stack (builds Docker image, starts postgres + redis + api + worker)
docker compose up --build

# 3. Open frontend — just open index.html in Windows browser
#    (right-click index.html → Open with → Chrome/Edge)
#    OR serve it:
python3 -m http.server 3000
# Then visit: http://localhost:3000
```

**Startup sequence (takes ~2-3 min first time, ~20s after that):**
```
[+] Building ... (Docker builds Python image with all deps)
rtde_postgres  | database system is ready to accept connections
rtde_redis     | Ready to accept connections
rtde_migrate   | INFO  [alembic] Running upgrade -> 001_initial_schema
rtde_migrate   | INFO  [alembic] Running upgrade -> 002_operator_and_canary
rtde_migrate   | INFO  [alembic] Running upgrade -> 003_approvals_and_incidents
rtde_migrate   | Done
rtde_api       | INFO: Application startup complete.
rtde_api       | INFO: Uvicorn running on http://0.0.0.0:8000
```

---

## URLs (once running)

| Service | URL |
|---------|-----|
| Backend API | http://localhost:8000 |
| Swagger Docs | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |
| Prometheus metrics | http://localhost:8000/metrics |
| Frontend | Open `index.html` in browser |

---

## Common Commands

```bash
# View live logs
docker compose logs -f app

# Check all container status
docker compose ps

# Restart only the API (fast — no rebuild)
docker compose restart app

# Open PostgreSQL shell
docker exec -it rtde_postgres psql -U rtde_user -d rtde_db

# Open bash in API container
docker exec -it rtde_api /bin/bash

# Stop everything
docker compose down

# Stop + delete database (fresh start)
docker compose down -v

# Rebuild after code changes
docker compose up --build
```

Or use `make` shortcuts:
```bash
make up         # start
make down       # stop
make logs       # tail logs
make health     # test /health
make decision   # fire a test decision
make psql       # postgres shell
```

---

## Test It's Working

```bash
# Health check — should return {"status":"ok","db":"ok","redis":"ok"}
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

# Expected response:
# {
#   "decision_id": "...",
#   "action": "SCALE_UP_3",
#   "policy_type": "BANDIT",
#   "q_values": {...},
#   "latency_ms": 12.4,
#   ...
# }
```

---

## Troubleshooting

### `docker: command not found`
Docker Desktop isn't running or WSL integration isn't enabled.
→ Start Docker Desktop → Settings → Resources → WSL Integration → enable your distro

### `port 5432 already in use`
Another Postgres is running locally.
→ `sudo service postgresql stop`  OR change `"5432:5432"` to `"5433:5432"` in docker-compose.yml

### `port 8000 already in use`
Something else on port 8000.
→ Change `"8000:8000"` to `"8001:8000"` in docker-compose.yml, then use http://localhost:8001

### Migration fails with `relation already exists`
Old database volume has partial data.
→ `docker compose down -v && docker compose up --build`

### `rtde_migrate exited with code 1`
Usually a DB connection issue — postgres wasn't ready yet.
→ `docker compose logs migrate` to see the exact error
→ `docker compose up --build` again (migrate retries cleanly)

### Frontend shows "Connecting..." but never connects
API URL mismatch. In the frontend sidebar, change API URL to `http://localhost:8000`

### WSL can't reach localhost from Windows browser
→ Use `http://localhost:8000` (Docker Desktop maps ports to Windows localhost automatically)
→ If that fails: `ip addr show eth0` in WSL → use that IP in browser

---

## Architecture (local)

```
Windows Browser
      │  http://localhost:3000 (index.html)
      │  http://localhost:8000 (API calls)
      ▼
   WSL2 Network
      │
      ├── rtde_api (container)        :8000
      │      ├── FastAPI app
      │      ├── asyncpg → postgres
      │      └── aioredis → redis
      │
      ├── rtde_postgres (container)   :5432
      │      └── 11 tables, all migrated
      │
      ├── rtde_redis (container)      :6379
      │      └── kill switch, canary, circuit breakers
      │
      └── rtde_worker (container)
             └── Celery: ML training tasks
```
