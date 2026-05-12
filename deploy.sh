#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# RTDE — One-Shot Deploy Script
# Pushes to GitHub → triggers Render (backend) + Vercel (frontend)
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO_NAME="rtde-xenon-control"
RENDER_SERVICE_NAME="rtde-api"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║       RTDE — Xenon Control Deploy Script             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── 1. Check prerequisites ────────────────────────────────────────────
check_cmd() {
  if ! command -v "$1" &>/dev/null; then
    echo "❌  '$1' not found. Install it and re-run."
    echo "    $2"
    exit 1
  fi
}

check_cmd git  "brew install git  / apt install git"
check_cmd gh   "brew install gh   / https://cli.github.com"

echo "✅  Prerequisites satisfied"
echo ""

# ── 2. GitHub auth check ──────────────────────────────────────────────
if ! gh auth status &>/dev/null; then
  echo "🔐  GitHub CLI not authenticated. Launching auth..."
  gh auth login
fi
echo "✅  GitHub authenticated"

# ── 3. Git init if needed ─────────────────────────────────────────────
if [ ! -d ".git" ]; then
  git init
  git add .
  git commit -m "feat: RTDE Xenon Control — initial production deploy"
fi

# ── 4. Create GitHub repo + push ─────────────────────────────────────
if gh repo view "$REPO_NAME" &>/dev/null 2>&1; then
  echo "📦  Repo $REPO_NAME already exists — pushing updates..."
  git add -A
  git diff --staged --quiet || git commit -m "chore: update $(date '+%Y-%m-%d %H:%M')"
  git push
else
  echo "📦  Creating GitHub repo: $REPO_NAME..."
  gh repo create "$REPO_NAME" \
    --public \
    --description "RTDE Xenon Control — Real-Time ML Decision Engine" \
    --push \
    --source=.
fi

REPO_URL=$(gh repo view "$REPO_NAME" --json url -q .url)
echo ""
echo "✅  GitHub repo: $REPO_URL"
echo ""

# ── 5. Render instructions ────────────────────────────────────────────
echo "════════════════════════════════════════════════════════"
echo "  STEP 2 — Deploy Backend on Render (free)"
echo "════════════════════════════════════════════════════════"
echo ""
echo "  1. Go to: https://render.com/deploy?repo=$REPO_URL"
echo "     (or: dashboard.render.com → New → Web Service)"
echo ""
echo "  2. Connect repo: $REPO_NAME"
echo ""
echo "  3. Render auto-reads render.yaml — it will:"
echo "     • Create a FREE PostgreSQL database (rtde-postgres)"
echo "     • Create a FREE web service (rtde-api)"
echo "     • Run migrations then start uvicorn"
echo ""
echo "  4. After ~3 minutes your backend is live at:"
echo "     https://rtde-api.onrender.com"
echo ""
echo "  NOTE: Free Render services sleep after 15 min inactivity."
echo "  First request after sleep takes ~30s (cold start)."
echo ""

# ── 6. Vercel instructions ────────────────────────────────────────────
echo "════════════════════════════════════════════════════════"
echo "  STEP 3 — Deploy Frontend on Vercel (free)"
echo "════════════════════════════════════════════════════════"
echo ""

if command -v npx &>/dev/null; then
  echo "  Deploying frontend with Vercel CLI..."
  echo ""
  VERCEL_URL=$(npx vercel --prod --yes 2>&1 | grep "https://" | tail -1)
  echo ""
  echo "✅  Frontend live at: $VERCEL_URL"
else
  echo "  Option A (CLI):"
  echo "    npm i -g vercel && vercel --prod"
  echo ""
  echo "  Option B (UI):"
  echo "    1. Go to: https://vercel.com/new"
  echo "    2. Import repo: $REPO_NAME"
  echo "    3. Framework: Other"
  echo "    4. Root Directory: /"
  echo "    5. Click Deploy"
  echo ""
  echo "  Frontend URL will be: https://$REPO_NAME.vercel.app"
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "  STEP 4 — Wire Backend URL into Frontend"
echo "════════════════════════════════════════════════════════"
echo ""
echo "  After both deploy, update index.html line 8:"
echo "    window.RTDE_API_URL = \"https://YOUR-RENDER-URL.onrender.com\";"
echo ""
echo "  Then: git add index.html && git commit -m 'config: set live backend url' && git push"
echo "  Vercel auto-redeploys on push."
echo ""
echo "════════════════════════════════════════════════════════"
echo "  📋  RESUME LINKS"
echo "════════════════════════════════════════════════════════"
echo ""
echo "  Frontend (Vercel):  https://$REPO_NAME.vercel.app"
echo "  Backend API:        https://rtde-api.onrender.com"
echo "  API Docs:           https://rtde-api.onrender.com/docs"
echo "  GitHub:             $REPO_URL"
echo ""
