# RTDE — Deploy Steps (GitHub → Render + Vercel)

## What you get
| Service | URL | Cost |
|---------|-----|------|
| Frontend | `https://rtde-xenon-control.vercel.app` | Free forever |
| Backend API | `https://rtde-api.onrender.com` | Free (sleeps at idle) |
| PostgreSQL | Managed by Render | Free (90 days) |
| API Docs | `https://rtde-api.onrender.com/docs` | — |

---

## STEP 1 — Push to GitHub

```bash
# Install GitHub CLI: https://cli.github.com
# Mac:   brew install gh
# WSL:   sudo apt install gh  OR  download from releases page

gh auth login

# Inside the project directory:
cd rtde-xenon-control
gh repo create rtde-xenon-control --public --push --source=.
```

Copy your GitHub repo URL — you'll need it in the next step.

---

## STEP 2 — Deploy Backend on Render (free)

**Option A — One-click (fastest):**

Click: [![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/YOUR_USERNAME/rtde-xenon-control)

**Option B — Manual (3 minutes):**

1. Go to **https://dashboard.render.com** → sign up free
2. Click **New +** → **Blueprint** (reads `render.yaml` automatically)
3. Connect your GitHub account → select `rtde-xenon-control`
4. Render creates:
   - `rtde-postgres` — free PostgreSQL database
   - `rtde-api` — free web service (Python runtime)
5. First deploy takes ~5 min (pip install scipy etc.)
6. ✅ Backend URL: `https://rtde-api.onrender.com`

**Verify backend is alive:**
```bash
curl https://rtde-api.onrender.com/health
# Expected: {"status":"ok","db":"ok","redis":"error",...}
# redis shows "error" on free tier (no Redis add-on) — that's fine, app still works
```

---

## STEP 3 — Wire the live backend URL into the frontend

Edit `index.html` — find line 8 and update:

```html
<!-- BEFORE -->
window.RTDE_API_URL = window.RTDE_API_URL || "https://rtde-api.onrender.com";

<!-- AFTER — paste your actual Render URL -->
window.RTDE_API_URL = window.RTDE_API_URL || "https://rtde-api-XXXX.onrender.com";
```

> 💡 **Or skip this** — users can type the URL live in the sidebar API input field.

Commit and push:
```bash
git add index.html
git commit -m "config: set live Render backend URL"
git push
```

---

## STEP 4 — Deploy Frontend on Vercel (free)

**Option A — CLI (30 seconds):**
```bash
npm install -g vercel   # or: npx vercel
vercel --prod
# Framework: Other
# Root directory: ./
# Accept defaults for everything else
```

**Option B — Web UI:**
1. Go to **https://vercel.com** → sign up free
2. **Add New Project** → Import from GitHub → select `rtde-xenon-control`
3. **Framework Preset**: Other
4. **Root Directory**: `/` (default)
5. No build command needed (it's a static HTML file)
6. Click **Deploy**
7. ✅ Frontend URL: `https://rtde-xenon-control.vercel.app`

Vercel auto-redeploys on every `git push`.

---

## STEP 5 — Verify everything works end-to-end

```bash
# 1. Health check
curl https://rtde-api.onrender.com/health

# 2. Make an ML decision
curl -X POST https://rtde-api.onrender.com/api/v1/decision \
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

# 3. Open frontend
open https://rtde-xenon-control.vercel.app

# 4. Open Swagger docs
open https://rtde-api.onrender.com/docs
```

---

## Render Free Tier Notes

| Behaviour | Explanation |
|-----------|-------------|
| Cold start ~30s | Free services sleep after 15 min idle. First request wakes them up. |
| No Redis | Free tier has no Redis add-on. Kill switch + canary router degrade gracefully. |
| PostgreSQL 90-day limit | Free DB expires after 90 days. Enough for job hunting. |
| 512MB RAM | Sufficient for uvicorn + numpy DQN inference. |

**To eliminate cold starts:** Upgrade to Render "Starter" ($7/mo) — always-on.

---

## Resume Links

```
Frontend:   https://rtde-xenon-control.vercel.app
API Docs:   https://rtde-api.onrender.com/docs
GitHub:     https://github.com/YOUR_USERNAME/rtde-xenon-control
```
