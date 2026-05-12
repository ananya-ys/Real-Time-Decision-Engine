# RTDE v2 — Updated System Architecture

**Review verdict:** Architecture ambition 9/10 | Production realism 6.5/10 → target 9.5/10

---

## What the review found

### Previously strong
- Safety gating, shadow mode, model versioning
- Two-signal drift detection (PSI + Mann-Whitney)
- Atomic rollback, row-level locking, ExplorationGuard
- Reproducible benchmarking, explicit SLOs

### Critical gaps (now being addressed)

| Gap | Severity | Phase |
|-----|----------|-------|
| No kill switch or manual override | P0 | 10 |
| No RBAC (all endpoints unprotected) | P0 | 10 |
| No canary rollout (shadow ≠ canary) | P1 | 12 |
| No audit timeline / decision replay | P1 | 11 |
| No backtesting engine | P1 | 13 |
| No cost-per-decision tracking | P2 | 14 |
| No circuit breakers | P2 | 15 |
| No human approval workflow | P2 | 16 |
| No counterfactual simulator | WOW | 17 |
| No policy trust score | WOW | 17 |
| No auto-generated postmortem | WOW | 17 |

---

## Updated Architecture (v2)

```
┌──────────────────────────────────────────────────────────────────────┐
│                        API Gateway Layer                              │
│  Auth middleware → RBAC guard → Rate limiter → Request logging       │
├──────────────────────────────────────────────────────────────────────┤
│                      Operator Control Plane                           │
│  Kill switch │ Manual override │ Freeze exploration │ Canary router  │
├──────────────────────────────────────────────────────────────────────┤
│                        Decision Service                               │
│  Circuit breaker → ExplorationGuard → Policy.decide() → Cost gate   │
│                           │                                           │
│          ┌────────────────┼────────────────┐                         │
│     BaselinePolicy    BanditPolicy      RLPolicy                     │
│       (always on)   (online, ε/UCB)  (DQN, async)                  │
├──────────────────────────────────────────────────────────────────────┤
│                     Audit & Replay Layer                              │
│  Immutable event log │ Decision replay │ Counterfactual simulator   │
├──────────────────────────────────────────────────────────────────────┤
│                    Policy Lifecycle Layer                             │
│  Canary router │ Shadow gate │ Promotion workflow │ Trust score      │
├──────────────────────────────────────────────────────────────────────┤
│                        Safety Layer                                   │
│  DriftService │ RollbackService │ PolicyPromoter │ CostGuard         │
├──────────────────────────────────────────────────────────────────────┤
│                        ML Pipeline                                    │
│  StateNormalizer │ ReplayBuffer │ Backtesting engine │ Lineage        │
├──────────────────────────────────────────────────────────────────────┤
│                     Infrastructure Layer                              │
│  PostgreSQL 16 │ Redis 7 + Circuit breaker │ Celery 5 + DLQ        │
│  Prometheus │ Structlog │ Docker │ Railway/Render/Supabase          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## New Phases (10-17)

### Phase 10 — Operator Controls + RBAC
**The #1 missing piece. Every production ML system has a kill switch.**

Files:
- `app/operator/kill_switch.py` — global + per-policy disable
- `app/operator/manual_override.py` — force baseline, freeze promotion
- `app/operator/freeze_manager.py` — "stop all exploration NOW"
- `app/core/rbac.py` — roles: VIEWER, OPERATOR, ADMIN, SERVICE
- `app/models/operator_event.py` — immutable operator action log
- `app/models/rbac.py` — User, Role, ApiKey models
- `app/api/v1/operator.py` — operator control endpoints
- `alembic/versions/002_rbac_and_operator.py`

Gate: Kill switch kills exploration in < 100ms. RBAC 401/403 correct. All operator actions logged.

### Phase 11 — Decision Replay + Audit Timeline
**Makes the system debuggable. Without this, incidents take hours to diagnose.**

Files:
- `app/audit/replay_engine.py` — reconstruct exact state at any past decision
- `app/audit/timeline_builder.py` — incident timeline from audit events
- `app/audit/event_chain.py` — hash-chained immutable event log
- `app/repositories/decision_repository.py` — efficient query patterns
- `app/api/v1/audit.py` — GET /decisions/{id}/replay, GET /incidents/{id}/timeline

Gate: Any decision replayed in < 500ms. Hash chain validates. Timeline auto-generated after rollback.

### Phase 12 — Canary Rollout + Traffic Splitting
**Shadow mode ≠ canary. Shadow logs but never commits. Canary commits to real traffic.**

Files:
- `app/canary/canary_router.py` — traffic percentage router (10%→25%→50%→100%)
- `app/canary/canary_config.py` — per-policy traffic weight
- `app/canary/auto_abort.py` — auto-disable on SLA breach during canary
- `app/models/canary_config.py` — DB-backed canary state
- `app/api/v1/canary.py` — POST /canary/start, GET /canary/status, POST /canary/abort

Gate: Traffic split verified at P99. Auto-abort fires within 60s of SLA breach.

### Phase 13 — Backtesting Engine
**The biggest credibility gap. Production ML systems replay history before promoting.**

Files:
- `app/backtesting/engine.py` — replay historical decisions through any policy
- `app/backtesting/scenario_library.py` — BURST, STEADY, DRIFT, OVERLOAD scenarios
- `app/backtesting/counterfactual.py` — "what if we chose SCALE_UP_3?"
- `app/backtesting/report.py` — structured backtest report
- `app/api/v1/backtesting.py` — POST /backtest/run, GET /backtest/{id}

Gate: Full 24h trace replayed through all 3 policies in < 30s. Counterfactual within 10s.

### Phase 14 — Cost Controls + Budget Guardrails
**A system that optimizes latency while ignoring cost is not production-grade.**

Files:
- `app/cost/cost_tracker.py` — per-decision, per-policy cost tracking
- `app/cost/budget_guard.py` — ceiling enforcement, auto-fallback
- `app/cost/cost_reporter.py` — cost/SLA tradeoff metrics
- `app/models/cost_log.py` — cost log table

Gate: Cost tracked per decision. Auto-fallback on budget breach.

### Phase 15 — Failure Containment
**Circuit breakers, DLQ, graceful degradation.**

Files:
- `app/circuit_breaker/db_breaker.py` — DB circuit breaker (closed/open/half-open)
- `app/circuit_breaker/redis_breaker.py` — Redis circuit breaker
- `app/circuit_breaker/worker_breaker.py` — Celery circuit breaker
- `app/worker/dead_letter.py` — DLQ for failed training jobs

Gate: DB failure → circuit opens → baseline serves → circuit recovers.

### Phase 16 — Human Workflow + Postmortem
**Production ML needs human decision points, not just automation.**

Files:
- `app/workflow/promotion_approval.py` — approve/reject policy promotion
- `app/workflow/postmortem.py` — auto-generated incident postmortem
- `app/workflow/runbook_tracker.py` — runbook execution state machine
- `app/models/approval.py` — promotion approval requests
- `app/api/v1/workflow.py` — approval endpoints

Gate: Postmortem auto-generated within 5min of rollback. Approval gate blocks unapproved promotions.

### Phase 17 — Wow Features
**These are what make the project feel elite.**

Files:
- `app/trust/policy_trust_score.py` — composite trust: drift + reward + confidence + violations
- `app/backtesting/counterfactual_api.py` — "what would SCALE_UP_3 have done?"
- `app/observability/shadow_comparison.py` — shadow vs active diff dashboard
- `app/operator/seasonality_guard.py` — hour/day aware exploration thresholds
- `app/workflow/postmortem_generator.py` — NL postmortem generation

Gate: Trust score updates every 60s. Counterfactual responds in < 1s.

---

## Layer Boundaries (enforced)

```
Router     → service call only, zero business logic
Service    → orchestrates operator → policy → cost → audit
Operator   → reads kill_switch before ANY policy.decide()
Policy     → no imports from services or operator
Cost       → reads after decision, blocks if over budget
Audit      → writes after every decision in same transaction
Repository → DB access only, called from services
```

## RBAC Matrix

| Role | Read decisions | Make decisions | Operator controls | Promote policy | Admin |
|------|---------------|----------------|-------------------|----------------|-------|
| VIEWER | ✅ | ❌ | ❌ | ❌ | ❌ |
| OPERATOR | ✅ | ✅ | ✅ | ❌ | ❌ |
| ADMIN | ✅ | ✅ | ✅ | ✅ | ✅ |
| SERVICE | ❌ | ✅ | ❌ | ❌ | ❌ |

## New SLOs (v2)

| SLO | Target | Phase |
|-----|--------|-------|
| Kill switch activation | < 100ms | 10 |
| Decision replay | < 500ms | 11 |
| Canary auto-abort | < 60s after SLA breach | 12 |
| Backtest 24h trace | < 30s | 13 |
| Postmortem generation | < 5min after rollback | 16 |
| Trust score refresh | < 60s | 17 |

---

## Priority Order (from review)

1. ✅ Kill switch + manual override (Phase 10)
2. ✅ RBAC + auth (Phase 10)
3. ✅ Audit/replay timeline (Phase 11)
4. ✅ Canary rollout (Phase 12)
5. ✅ Backtesting engine (Phase 13)
6. ✅ Cost controls (Phase 14)
7. ✅ Circuit breakers (Phase 15)
8. ✅ Human workflow (Phase 16)
9. ✅ Wow features (Phase 17)
