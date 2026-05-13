# RTDE — Xenon Control System

Real-time decision engine for adaptive cloud resource management.

This project simulates a production-style ML control system that:
- receives live infrastructure state
- makes scaling decisions
- logs rewards asynchronously
- supports explainability and audit replay
- includes operator controls, canary rollout, approvals, and incident handling

---

## Tech Stack

### Backend
- FastAPI
- PostgreSQL
- SQLAlchemy
- Alembic
- Redis
- Docker

### Frontend
- HTML
- CSS
- JavaScript

### Infrastructure
- Docker Compose
- Render
- Vercel

---

## Core Features

- Real-time scaling decisions
- Reward feedback pipeline
- Audit replay system
- Explainability APIs
- Operator controls and kill switches
- Canary rollout workflows
- Incident management APIs
- Approval and promotion workflows
- Swagger API documentation
- WebSocket-based live updates

---

## Local Development

### Prerequisites

- Docker Desktop
- WSL2 (Windows)
- Git
- VS Code (recommended)

---

### Run Locally

Clone the repository:

```bash
git clone https://github.com/ananya-ys/Real-Time-Decision-Engine.git
cd Real-Time-Decision-Engine
```

Start the application stack:

```bash
docker compose up --build
```

---

## Local URLs

Backend API:
```text
http://localhost:8000
```

Swagger Docs:
```text
http://localhost:8000/docs
```

Health Check:
```text
http://localhost:8000/health
```

Frontend:
Open `frontend/index.html` using VS Code Live Server.

---

## Useful Commands

```bash
docker compose up --build
docker compose down
docker ps
docker logs <container_id>
```

---

## Example API Request

```bash
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
      "traffic_regime": "BURST"
    }
  }'
```

---

## Project Structure

```text
app/
frontend/
tests/
alembic/
docs/
runbooks/
```

---

## Main System Components

### Decision Engine
Handles scaling decisions based on incoming infrastructure state.

### Reward Pipeline
Processes asynchronous reward feedback for previous decisions.

### Explainability Layer
Provides reasoning and Q-value analysis for policy actions.

### Audit System
Supports replay and investigation of historical decisions.

### Operator Controls
Includes maintenance mode, kill switches, approvals, and rollout management.

---

## Deployment

Backend deployment:
- Render

Frontend deployment:
- Vercel

Deployment configuration files:
```text
render.yaml
vercel.json
docker-compose.yml
```

---

## Status

Current state:
- Backend APIs operational
- Frontend operational
- Docker environment configured
- Local development workflow tested
- Deployment configuration prepared

---

## License

This project is intended for educational and portfolio purposes.