#!/bin/bash
# ============================================================================
# Browser Agent - Enterprise Startup Script
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="/home/node/browser-agent-venv"
PYTHON="$VENV/bin/python3"
LOG_DIR="$SCRIPT_DIR/logs"

mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Browser Agent - Enterprise Control Panel"
echo "============================================"
echo ""

# ─── Check prerequisites ─────────────────────────────────────────────────────
echo "[1/4] Checking prerequisites..."

if ! command -v Xvfb &>/dev/null; then
    echo "ERROR: Xvfb not installed. Run: sudo apt-get install xvfb"
    exit 1
fi

if ! command -v x11vnc &>/dev/null; then
    echo "ERROR: x11vnc not installed. Run: sudo apt-get install x11vnc"
    exit 1
fi

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Python venv not found at $VENV"
    exit 1
fi

# Check for API key
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -z "${AI_GATEWAY_API_KEY:-}" ]; then
    echo "WARNING: Neither ANTHROPIC_API_KEY nor AI_GATEWAY_API_KEY is set."
    echo "         The agent will not be able to call the LLM."
fi

echo "  OK: All prerequisites met"

# ─── Check for xwd / ImageMagick ─────────────────────────────────────────────
echo "[2/4] Checking screenshot tools..."

HAS_SCREENSHOT=true
if ! command -v xwd &>/dev/null; then
    echo "  WARNING: xwd not found - installing x11-apps..."
    sudo apt-get install -y -qq x11-apps 2>/dev/null || HAS_SCREENSHOT=false
fi

if ! command -v convert &>/dev/null; then
    echo "  WARNING: ImageMagick not found - installing..."
    sudo apt-get install -y -qq imagemagick 2>/dev/null || HAS_SCREENSHOT=false
fi

if ! command -v xsetroot &>/dev/null; then
    echo "  WARNING: xsetroot not found - installing x11-xserver-utils..."
    sudo apt-get install -y -qq x11-xserver-utils 2>/dev/null || true
fi

if [ "$HAS_SCREENSHOT" = true ]; then
    echo "  OK: Screenshot tools available"
else
    echo "  WARNING: Screenshot tools missing - screenshot mode will be unavailable"
    echo "           VNC stream will still work"
fi

# ─── Kill existing processes ──────────────────────────────────────────────────
echo "[3/4] Cleaning up existing processes..."

pkill -f "Xvfb :99" 2>/dev/null || true
pkill -f "x11vnc.*5999" 2>/dev/null || true
pkill -f "websockify.*6080" 2>/dev/null || true
pkill -f "agent_server.py" 2>/dev/null || true
sleep 0.5
echo "  OK: Clean slate"

# ─── Launch ───────────────────────────────────────────────────────────────────
echo "[4/4] Starting services..."
echo ""

# Set DISPLAY for the Python process
export DISPLAY=:99

# Start the server (it manages Xvfb, VNC, noVNC internally)
echo "  Starting Browser Agent server on port ${AGENT_PORT:-8888}..."
cd "$SCRIPT_DIR"

exec "$PYTHON" agent_server.py 2>&1 | tee "$LOG_DIR/server.log"
