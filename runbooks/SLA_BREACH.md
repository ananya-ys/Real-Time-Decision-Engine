# Runbook: SLA_BREACH

**Alert Severity:** P1  
**MTTR Target:** < 10 minutes  
**On-Call:** Infrastructure + ML Engineering  

---

## What Just Happened

More than 5% of scaling decisions in the last hour have P99 latency > 300ms.
This is the RTDE inference latency SLO — not the application P99 latency.

---

## Immediate Triage

```bash
curl https://your-app.railway.app/api/v1/monitoring/health/slo | jq .
```

Look at:
- `decision_latency_slo.breach_rate` — what fraction of decisions are slow?
- `decision_latency_slo.total` — how many decisions in the window?
- `fallback_rate_slo.fallback_rate` — is the active policy throwing exceptions?

---

## Common Causes

### Cause 1: DB connection pool exhaustion
**Symptom:** `fallback_rate_slo.fallback_rate` > 1% + DB errors in logs  
**Fix:** Scale `DB_POOL_SIZE` env var, or investigate long-running transactions

### Cause 2: RL policy training on inference path
**Symptom:** All slow decisions are `policy_type: RL`  
**Fix:** Verify `train_step()` is only called from Celery — never in `decide()`  
**Check:** `structlog key: training_step` should only appear in Celery worker logs

### Cause 3: ExplorationGuard configuration too loose
**Symptom:** High `exploration_suppressed_total` counter + slow decisions during bursts  
**Fix:** Review `EXPLORATION_LATENCY_WARNING_MS` threshold in env vars

### Cause 4: Celery worker using same DB pool as API
**Symptom:** Slow decisions correlate with training task runs  
**Fix:** Separate DB connection pools for API and Celery workers

---

## Resolution

1. If fallback rate > 5%: force policy downgrade to BASELINE:
   ```bash
   # Check active policy
   curl https://your-app.railway.app/api/v1/policies/active

   # If RL/Bandit is active and failing, trigger manual rollback
   curl -X POST https://your-app.railway.app/api/v1/admin/rollback-to-baseline
   ```

2. Verify SLO recovers within 5 minutes:
   ```bash
   curl https://your-app.railway.app/api/v1/monitoring/health/slo | jq .overall_passing
   ```

3. Root cause analysis via structured logs:
   ```bash
   # Find slow decisions
   # structlog query: latency_ms > 300 AND event = "decision_made"
   ```
