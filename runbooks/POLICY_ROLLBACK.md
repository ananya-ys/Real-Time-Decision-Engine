# Runbook: POLICY_ROLLBACK

**Alert Severity:** P2  
**MTTR Target:** < 20 minutes  
**On-Call:** ML Engineering  

---

## When to Use This Runbook

1. Automatic rollback fired (DRIFT_DETECTED alert) — use this for recovery
2. You want to **manually** trigger a rollback before drift fires
3. Shadow promotion failed unexpectedly

---

## Manual Rollback Procedure

### Option A: Force Baseline (Immediate Safety)

```bash
# Confirm current active policy
curl https://your-app.railway.app/api/v1/policies/active | jq .

# Check dashboard for system status
curl https://your-app.railway.app/api/v1/monitoring/dashboard | jq .system_status
```

If you need to force baseline immediately, retire the active ML policy:
```bash
# Via API (when implemented in Phase 8)
curl -X POST https://your-app.railway.app/api/v1/admin/force-baseline

# Or via DB (emergency)
# UPDATE policy_versions SET status='RETIRED', demoted_at=NOW()
# WHERE status='ACTIVE' AND policy_type IN ('RL', 'BANDIT');
```

---

### Option B: Roll Back to Previous Version

If you want to roll back to a specific previous version (not baseline):

1. List available RETIRED versions:
   ```bash
   curl https://your-app.railway.app/api/v1/policies/versions | jq '.versions[] | select(.status=="RETIRED")'
   ```

2. Promote the previous version back to SHADOW:
   ```bash
   # Update status via API (Phase 8)
   curl -X PATCH https://your-app.railway.app/api/v1/policies/{old_version_id}/status \
     -d '{"status": "SHADOW"}'
   ```

3. Run evaluation (need ≥ 5 seeds):
   ```bash
   curl -X POST https://your-app.railway.app/api/v1/training/evaluate \
     -d '{"policy_version_id": "{old_version_id}", "n_seeds": 5}'
   ```

4. Promote when eval complete:
   ```bash
   curl -X POST https://your-app.railway.app/api/v1/policies/{old_version_id}/promote
   ```

---

## Verifying Rollback Success

```bash
# 1. Active policy should be BASELINE (or your target version)
curl https://your-app.railway.app/api/v1/policies/active | jq .

# 2. SLO should be passing
curl https://your-app.railway.app/api/v1/monitoring/health/slo | jq .overall_passing

# 3. No more drift events being generated
curl https://your-app.railway.app/api/v1/monitoring/dashboard | jq .drift_status.drift_events_24h
```

---

## Preventing Recurrence

After manual rollback:

1. **Root cause:** Check DriftEvent table for `psi_score` and `reward_delta`
2. **Retraining:** Ensure `train_rl_policy` Celery task is running with updated config
3. **Thresholds:** Review `DRIFT_PSI_THRESHOLD` and `DRIFT_SIGNIFICANCE_ALPHA` env vars
4. **Monitor:** Shadow new policy for ≥ 24h before promoting to active

---

## Key Prometheus Queries

```promql
# Rollback rate
rate(rtde_rollbacks_total[1h])

# Decision latency P99
histogram_quantile(0.99, rate(rtde_decision_latency_ms_bucket[5m]))

# SLA violation rate
rate(rtde_sla_violations_total[5m]) / rate(rtde_decisions_total[5m])

# Exploration suppression rate
rate(rtde_exploration_suppressed_total[5m])
```
