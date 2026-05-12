# RTDE — Developer shortcuts
# Usage: make up | make down | make logs | make test | make shell

.PHONY: up down restart logs ps build clean migrate shell test lint

## Start everything (build if needed)
up:
	docker compose up --build -d
	@echo ""
	@echo "✅  RTDE is starting..."
	@echo "   Backend:  http://localhost:8000"
	@echo "   API Docs: http://localhost:8000/docs"
	@echo "   Health:   http://localhost:8000/health"
	@echo ""
	@echo "   Frontend: open index.html in your browser"
	@echo "   Or:       python -m http.server 3000  (then visit http://localhost:3000)"

## Start with logs visible
up-logs:
	docker compose up --build

## Stop all containers
down:
	docker compose down

## Stop and remove volumes (wipe database)
down-clean:
	docker compose down -v
	@echo "🗑️  Database wiped"

## Restart the API only (fast — no rebuild)
restart:
	docker compose restart app

## Watch logs
logs:
	docker compose logs -f app

## Watch all logs
logs-all:
	docker compose logs -f

## Show running containers
ps:
	docker compose ps

## Build images only
build:
	docker compose build

## Run migrations manually
migrate:
	docker compose run --rm migrate

## Open shell in API container
shell:
	docker exec -it rtde_api /bin/bash

## Open psql shell
psql:
	docker exec -it rtde_postgres psql -U rtde_user -d rtde_db

## Run tests (inside container)
test:
	docker compose run --rm app python -m pytest tests/ -x -v --no-header

## Lint (ruff)
lint:
	docker compose run --rm app python -m ruff check app/ tests/

## Quick health check
health:
	@curl -s http://localhost:8000/health | python3 -m json.tool

## Make a test decision
decision:
	@curl -s -X POST http://localhost:8000/api/v1/decision \
	  -H "Content-Type: application/json" \
	  -d '{"state":{"cpu_utilization":0.75,"request_rate":2500,"p99_latency_ms":320,"instance_count":4,"hour_of_day":14,"day_of_week":2,"traffic_regime":"STEADY","error_rate":0.01,"memory_utilization":0.6,"active_instances":4}}' \
	  | python3 -m json.tool
