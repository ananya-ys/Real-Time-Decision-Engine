# Runbook: DRIFT_DETECTED

**Alert Severity:** P1  
**MTTR Target:** < 15 minutes  
**On-Call:** ML Engineering  

---

## What Just Happened

The RTDE drift detector identified that the active ML policy's performance has degraded below 
the rollback threshold for **3 consecutive 60-second evaluation windows**.

The system has **already automatically rolled back** to the BaselinePolicy (deterministic threshold rules).
The BaselinePolicy is always safe — it uses configurable CPU and latency thresholds and makes no ML decisions.

Your job: **verify the rollback succeeded**, identify the drift cause, and prepare for policy recovery.

---

## Step 1: Verify Rollback Completed (< 2 min)

```bash
curl https://your-app.railway.app/api/v1/monitoring/dashboard | jq .active_policy
```

**Expected:**
```json
{"policy_type": "BASELINE", "version": null, "algorithm": "threshold_v1"}
```

**If ML policy is still active:** The rollback failed. Escalate immediately.

---

## Step 2: Identify Drift Signal (< 3 min)

```bash
curl https://your-app.railway.app/api/v1/monitoring/dashboard | jq .drift_status
```

**Possible signals:**

| drift_signal | Meaning | Action |
|--------------|---------|--------|
| `REWARD_DEGRADATION` | Policy performance dropped, same traffic | Retrain from scratch |
| `INPUT_DRIFT` | Traffic pattern shifted from training distribution | Collect new data, retrain |
| `BOTH` | Both signals fired | Treat as INPUT_DRIFT + retrain |

**Check PSI score:**
- `psi_score < 0.1`: No input distribution shift → pure reward degradation
- `psi_score > 0.2`: Significant input shift → model never saw this traffic pattern

---

## Step 3: Check SLO Recovery (< 2 min)

```bash
curl https://your-app.railway.app/api/v1/monitoring/health/slo | jq .
```

**Expected after rollback:**
```json
{
  "overall_passing": true,
  "decision_latency_slo": {"passing": true},
  "fallback_rate_slo": {"passing": true}
}
```

If SLOs are not recovering, check: Is BaselinePolicy receiving traffic? Are environment states valid?

---

## Step 4: Verify Retraining Task (< 2 min)

```bash
# Check Celery task queue
celery -A app.worker.celery_app inspect active

# Or via logs:
# structlog key: "training_task_started"
```

If no retraining task is running: manually enqueue one:
```bash
curl -X POST https://your-app.railway.app/api/v1/training/enqueue \
  -H "Content-Type: application/json" \
  -d '{"policy_type": "RL", "n_steps": 2000}'
```

---

## Step 5: Policy Recovery (after retraining completes)

1. Wait for `train_rl_policy` task to complete
2. Verify eval_seeds ≥ 5:
   ```bash
   curl https://your-app.railway.app/api/v1/policies/versions | jq '.versions[] | select(.status=="SHADOW")'
   ```
3. If eval metrics are acceptable, promote shadow to active:
   ```bash
   curl -X POST https://your-app.railway.app/api/v1/policies/{version_id}/promote
   ```

---

## Escalation

If not resolved in 15 minutes:
1. Keep BaselinePolicy active (it is safe and stable)
2. Page on-call ML engineer
3. Share dashboard output and drift event details

---

## Post-Incident

1. File incident report referencing `drift_event_id` from logs
2. Check `DriftEvent` table for psi_score and reward_delta
3. Determine if drift was:
   - Traffic pattern change → update normalizer training distribution
   - Model overfit → increase replay buffer size or regularization
   - Reward function mismatch → review reward weight configuration
