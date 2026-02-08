#!/bin/bash
# ============================================================================
# Browser Agent Server - One-Click Setup
# Installs all dependencies and deploys the application
# ============================================================================
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$SKILL_DIR/scripts"
DEST="${1:-./outputs/browser-agent}"
VENV="${BROWSER_AGENT_VENV:-/home/node/browser-agent-venv}"

echo "================================================"
echo "  Browser Agent Server - Setup"
echo "================================================"
echo "  Skill:  $SKILL_DIR"
echo "  Deploy: $DEST"
echo "  Venv:   $VENV"
echo ""

# ── System dependencies ─────────────────────────────────────────────────────
echo "[1/5] Installing system dependencies..."
if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq xvfb x11vnc x11-apps imagemagick novnc x11-xserver-utils 2>/dev/null || true
    echo "  OK"
elif command -v apk &>/dev/null; then
    apk add --no-cache xvfb x11vnc imagemagick novnc 2>/dev/null || true
    echo "  OK (Alpine)"
else
    echo "  WARNING: Unknown package manager. Install manually: xvfb x11vnc imagemagick"
fi

# ── Python venv ──────────────────────────────────────────────────────────────
echo "[2/5] Setting up Python virtual environment..."
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    echo "  Created venv at $VENV"
else
    echo "  Venv already exists at $VENV"
fi

echo "[3/5] Installing Python packages..."
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet \
    'browser-use==0.11.9' \
    'fastapi>=0.100.0' \
    'uvicorn[standard]>=0.20.0' \
    'websockets>=11.0'
echo "  OK"

echo "  Installing Chromium via Playwright..."
"$VENV/bin/python3" -m playwright install chromium 2>/dev/null
echo "  OK"

echo "  Installing Chromium system dependencies (libatk, libasound, fonts, etc.)..."
"$VENV/bin/python3" -m playwright install-deps chromium 2>/dev/null || \
    echo "  WARNING: playwright install-deps failed. If browser times out, run manually:"
    echo "           $VENV/bin/python3 -m playwright install-deps chromium"
echo "  OK"

# ── Deploy files ─────────────────────────────────────────────────────────────
echo "[4/5] Deploying application files..."
mkdir -p "$DEST"
cp "$SCRIPTS/agent_server.py" "$DEST/"
cp "$SCRIPTS/dashboard.html" "$DEST/"
cp "$SCRIPTS/start.sh" "$DEST/"
chmod +x "$DEST/start.sh"
echo "  Deployed to $DEST"

# ── Verify ───────────────────────────────────────────────────────────────────
echo "[5/5] Verifying installation..."
CHECKS=0
TOTAL=6
command -v Xvfb &>/dev/null && echo "  OK: Xvfb" && CHECKS=$((CHECKS+1))
command -v x11vnc &>/dev/null && echo "  OK: x11vnc" && CHECKS=$((CHECKS+1))
[ -f "$VENV/bin/python3" ] && echo "  OK: Python venv" && CHECKS=$((CHECKS+1))
"$VENV/bin/python3" -c "import browser_use" 2>/dev/null && echo "  OK: browser-use" && CHECKS=$((CHECKS+1))
"$VENV/bin/python3" -c "import fastapi" 2>/dev/null && echo "  OK: FastAPI" && CHECKS=$((CHECKS+1))

# Verify Chromium binary can actually load (shared libs present)
CHROME_BIN=$(find /home/node/.cache/ms-playwright -name "chrome" -type f 2>/dev/null | head -1)
if [ -n "$CHROME_BIN" ] && "$CHROME_BIN" --version &>/dev/null; then
    echo "  OK: Chromium ($($CHROME_BIN --version 2>/dev/null))"
    CHECKS=$((CHECKS+1))
else
    echo "  FAIL: Chromium binary cannot load - shared libraries missing!"
    echo "         Fix: $VENV/bin/python3 -m playwright install-deps chromium"
fi

echo ""
if [ $CHECKS -ge $TOTAL ]; then
    echo "Setup complete! All $TOTAL checks passed."
    echo ""
    echo "Start with:"
    echo "  cd $DEST"
    echo "  DISPLAY=:99 $VENV/bin/python3 agent_server.py"
    echo ""
    echo "Or use the startup script:"
    echo "  $DEST/start.sh"
else
    echo "WARNING: $CHECKS/$TOTAL checks passed. Review output above."
fi
