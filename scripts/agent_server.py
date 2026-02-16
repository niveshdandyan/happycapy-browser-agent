#!/usr/bin/env python3
"""
Browser Agent Server - Enterprise-grade browser automation with live screen viewing.

Architecture:
  [User Browser] <--WebSocket--> [FastAPI Server] <--controls--> [browser-use Agent]
                                                                       |
  [User Browser] <--noVNC WebSocket--> [x11vnc] <-- Xvfb display <----+
"""

import asyncio
import base64
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal as TypingLiteral, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# ─── Configuration ──────────────────────────────────────────────────────────────

DISPLAY_NUM = int(os.environ.get("DISPLAY_NUM", "98"))
DISPLAY = f":{DISPLAY_NUM}"
SCREEN_WIDTH = int(os.environ.get("SCREEN_WIDTH", "1280"))
SCREEN_HEIGHT = int(os.environ.get("SCREEN_HEIGHT", "1024"))
SCREEN_DEPTH = 24
VNC_PORT = 5999
NOVNC_PORT = 6080
API_PORT = int(os.environ.get("AGENT_PORT", "8888"))
SCREENSHOT_INTERVAL = 0.5  # seconds between screenshot pushes

# ─── Available Models ────────────────────────────────────────────────────────────

AVAILABLE_MODELS = [
    {"id": "anthropic/claude-sonnet-4.5", "name": "Claude Sonnet 4.5", "tier": "fast", "vision": True},
    {"id": "anthropic/claude-opus-4.5", "name": "Claude Opus 4.5", "tier": "reasoning", "vision": True},
    {"id": "openai/gpt-4o", "name": "GPT-4o", "tier": "fast", "vision": True},
    {"id": "moonshotai/kimi-k2.5", "name": "Kimi K2.5", "tier": "fast", "vision": True},
    {"id": "google/gemini-2.5-flash", "name": "Gemini 2.5 Flash", "tier": "fast", "vision": True},
    {"id": "google/gemini-2.5-pro", "name": "Gemini 2.5 Pro", "tier": "reasoning", "vision": True},
]
DEFAULT_MODEL_ID = "openai/gpt-4o"

STRATEGY_DESCRIPTIONS = {
    "single": "Single model handles all steps",
    "fallback_chain": "Primary model runs; auto-switches to secondary on error/rate-limit",
    "planner_executor": "Strong model plans; fast model executes browser steps",
    "consensus": "Primary model acts; judge model validates every step in real-time + final verdict",
    "council": "Single model runs; on repeated failure, stuck-in-loop, or stalled step (60s+), all models convene as council to diagnose, advise, and replan",
}

# Council configuration
COUNCIL_FAILURE_THRESHOLD = 2  # consecutive failures before council convenes
COUNCIL_LOOP_THRESHOLD = 3    # repeated identical/similar actions before council convenes
COUNCIL_LOOP_COOLDOWN = 3     # steps to wait after a loop-triggered council before re-triggering
COUNCIL_STALL_TIMEOUT = 60    # seconds a single step can run before council is convened for stall

# ─── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agent-server")


# ─── Splash Screen ─────────────────────────────────────────────────────────────

_SPLASH_PROCESS: Optional[subprocess.Popen] = None

SPLASH_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{
  background:#F9F6F1;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
  display:flex;align-items:center;justify-content:center;height:100vh;
  overflow:hidden;color:#3d3d3d;
}
[data-status="failed"] body{background:#FDF2F2}
.card{
  text-align:center;padding:3rem 4rem;
  background:#fff;border-radius:1rem;
  box-shadow:0 8px 24px rgba(0,0,0,0.06);
  max-width:560px;width:90%;
}
.capy{width:160px;height:160px;margin:0 auto 1.5rem}
.title{
  font-family:'Instrument Serif',Georgia,serif;font-size:2rem;font-weight:400;
  color:#2d2d2d;margin-bottom:0.5rem;line-height:1.25;
}
.subtitle{font-size:0.95rem;color:#7a7a7a;line-height:1.5;margin-bottom:1.5rem}
.badge{
  display:inline-flex;align-items:center;gap:6px;
  padding:6px 16px;border-radius:9999px;font-size:0.8rem;font-weight:600;
}
.badge.success{background:rgba(125,155,118,0.12);color:#5a7a52;border:1px solid rgba(125,155,118,0.25)}
.badge.failed{background:rgba(198,122,107,0.12);color:#a85a4a;border:1px solid rgba(198,122,107,0.25)}
.badge.stopped{background:rgba(212,167,106,0.12);color:#8a6d3a;border:1px solid rgba(212,167,106,0.25)}
.dot{width:8px;height:8px;border-radius:50%}
.dot.success{background:#7D9B76}
.dot.failed{background:#C67A6B}
.dot.stopped{background:#D4A76A}
.stats{
  display:flex;justify-content:center;gap:2rem;margin-top:1.25rem;
  font-size:0.8rem;color:#9B8B7A;
}
.stats span{font-weight:600;color:#5a5a5a}
.hint{font-size:0.75rem;color:#b0a898;margin-top:1.5rem}
</style></head>
<body>
<div class="card">
  <div class="capy">
    <svg viewBox="0 0 160 160" xmlns="http://www.w3.org/2000/svg">
      <!-- water ripples -->
      <ellipse cx="80" cy="145" rx="65" ry="8" fill="#7BA3A8" opacity="0.15"/>
      <ellipse cx="80" cy="145" rx="45" ry="5" fill="#7BA3A8" opacity="0.1"/>
      <!-- body -->
      <ellipse cx="80" cy="105" rx="42" ry="32" fill="#C4A882"/>
      <!-- head -->
      <ellipse cx="80" cy="68" rx="30" ry="26" fill="#D4B896"/>
      <!-- ears -->
      <ellipse cx="57" cy="48" rx="8" ry="6" fill="#C4A882" transform="rotate(-15 57 48)"/>
      <ellipse cx="103" cy="48" rx="8" ry="6" fill="#C4A882" transform="rotate(15 103 48)"/>
      <ellipse cx="57" cy="48" rx="5" ry="3.5" fill="#E8D5BC" transform="rotate(-15 57 48)"/>
      <ellipse cx="103" cy="48" rx="5" ry="3.5" fill="#E8D5BC" transform="rotate(15 103 48)"/>
      <!-- eyes -->
      <circle cx="70" cy="63" r="4" fill="#3d3d3d"/>
      <circle cx="90" cy="63" r="4" fill="#3d3d3d"/>
      <circle cx="71.2" cy="61.8" r="1.2" fill="#fff"/>
      <circle cx="91.2" cy="61.8" r="1.2" fill="#fff"/>
      <!-- nose -->
      <ellipse cx="80" cy="74" rx="6" ry="4" fill="#A08060"/>
      <circle cx="77.5" cy="73.5" r="1.2" fill="#8a6a4a"/>
      <circle cx="82.5" cy="73.5" r="1.2" fill="#8a6a4a"/>
      <!-- mouth -->
      <path d="M75 78 Q80 82 85 78" stroke="#A08060" stroke-width="1.2" fill="none" stroke-linecap="round"/>
      <!-- blush -->
      <ellipse cx="63" cy="72" rx="5" ry="3" fill="#E8A090" opacity="0.35"/>
      <ellipse cx="97" cy="72" rx="5" ry="3" fill="#E8A090" opacity="0.35"/>
      <!-- whiskers -->
      <line x1="55" y1="73" x2="40" y2="70" stroke="#C4A882" stroke-width="0.8" opacity="0.5"/>
      <line x1="55" y1="76" x2="40" y2="77" stroke="#C4A882" stroke-width="0.8" opacity="0.5"/>
      <line x1="105" y1="73" x2="120" y2="70" stroke="#C4A882" stroke-width="0.8" opacity="0.5"/>
      <line x1="105" y1="76" x2="120" y2="77" stroke="#C4A882" stroke-width="0.8" opacity="0.5"/>
      <!-- feet in water -->
      <ellipse cx="65" cy="132" rx="10" ry="5" fill="#B89B78"/>
      <ellipse cx="95" cy="132" rx="10" ry="5" fill="#B89B78"/>
      <!-- water line -->
      <path d="M15 138 Q40 132 80 138 Q120 144 145 138" stroke="#7BA3A8" stroke-width="1.5" fill="none" opacity="0.3"/>
      <path d="M25 143 Q50 137 80 143 Q110 149 135 143" stroke="#7BA3A8" stroke-width="1" fill="none" opacity="0.2"/>
    </svg>
  </div>
  <div class="title" id="title">Task Completed</div>
  <div class="subtitle" id="subtitle">The browser agent has finished its work.</div>
  <div class="badge success" id="badge"><div class="dot success"></div>Completed</div>
  <div class="stats">
    <div>Steps: <span id="steps">0</span></div>
    <div>Actions: <span id="actions">0</span></div>
  </div>
  <div class="hint">Start a new task from the dashboard to continue</div>
</div>
</body></html>"""


def _show_splash(status: str = "completed", steps: int = 0, actions: int = 0, result_text: str = ""):
    """Show a styled splash screen on the Xvfb display after task finishes."""
    global _SPLASH_PROCESS
    _kill_splash()

    # Customize HTML for the status
    html = SPLASH_HTML
    if status == "idle":
        html = html.replace('Task Completed', 'Browser Agent Ready')
        html = html.replace('has finished its work.', 'Waiting for a task. Open the dashboard to get started.')
        html = html.replace('badge success', 'badge stopped').replace('dot success', 'dot stopped')
        html = html.replace('>Completed<', '>Idle<')
    elif status == "failed":
        html = html.replace('<body>', '<body data-status="failed">')
        html = html.replace('Task Completed', 'Task Failed')
        html = html.replace('has finished its work.', 'encountered an error.')
        html = html.replace('badge success', 'badge failed').replace('dot success', 'dot failed')
        html = html.replace('>Completed<', '>Failed<')
    elif status == "stopped":
        html = html.replace('Task Completed', 'Task Stopped')
        html = html.replace('has finished its work.', 'was stopped by the user.')
        html = html.replace('badge success', 'badge stopped').replace('dot success', 'dot stopped')
        html = html.replace('>Completed<', '>Stopped<')

    # Inject stats via simple JS
    stats_js = f"""<script>
document.getElementById('steps').textContent='{steps}';
document.getElementById('actions').textContent='{actions}';
</script></body>"""
    html = html.replace('</body>', stats_js)

    splash_path = Path("/tmp/browser-agent-splash.html")
    splash_path.write_text(html)

    # Find Chromium binary
    chrome_bin = None
    pw_dir = Path.home() / ".cache" / "ms-playwright"
    if pw_dir.exists():
        for p in sorted(pw_dir.glob("chromium-*/chrome-linux64/chrome"), reverse=True):
            if p.is_file():
                chrome_bin = str(p)
                break

    if not chrome_bin:
        logger.warning("Cannot show splash: Chromium binary not found")
        return

    # Set Xvfb root window background to match splash body color so any gaps are invisible
    splash_bg = "#FDF2F2" if status == "failed" else "#F9F6F1"
    try:
        subprocess.run(
            ["xsetroot", "-solid", splash_bg],
            env={**os.environ, "DISPLAY": DISPLAY},
            capture_output=True, timeout=3,
        )
    except Exception:
        pass

    try:
        _SPLASH_PROCESS = subprocess.Popen(
            [
                chrome_bin,
                "--no-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-translate",
                "--disable-sync",
                "--disable-extensions",
                "--disable-infobars",
                f"--user-data-dir=/tmp/splash-chrome-profile",
                "--kiosk",
                "--window-position=0,0",
                f"--window-size={SCREEN_WIDTH},{SCREEN_HEIGHT}",
                f"file://{splash_path}",
            ],
            env={**os.environ, "DISPLAY": DISPLAY},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info(f"Splash screen shown (status={status}, PID={_SPLASH_PROCESS.pid})")
    except Exception as e:
        logger.warning(f"Failed to show splash screen: {e}")


def _kill_splash():
    """Kill the splash screen Chromium process if running."""
    global _SPLASH_PROCESS
    if _SPLASH_PROCESS and _SPLASH_PROCESS.poll() is None:
        try:
            _SPLASH_PROCESS.terminate()
            _SPLASH_PROCESS.wait(timeout=3)
        except Exception:
            try:
                _SPLASH_PROCESS.kill()
            except Exception:
                pass
    _SPLASH_PROCESS = None


# ─── FastAPI App ────────────────────────────────────────────────────────────────

app = FastAPI(title="Browser Agent Server", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to trusted domains in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Security Headers Middleware ─────────────────────────────────────────────

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        # Permissive CSP for internal dev tool served via iframe proxy.
        # Only frame-ancestors is restricted to prevent third-party framing.
        response.headers["Content-Security-Policy"] = (
            "frame-ancestors 'self' https://*.happycapy.ai;"
        )
        return response


app.add_middleware(SecurityHeadersMiddleware)

# ─── Global State ───────────────────────────────────────────────────────────────

class AgentState:
    def __init__(self):
        self.agent = None
        self.agent_task: Optional[asyncio.Task] = None
        self.is_running = False
        self.current_task_text = ""
        self.action_log: list[dict] = []
        self.step_count = 0
        self.status = "idle"  # idle, running, completed, failed
        self.result = None
        self.ws_clients: set[WebSocket] = set()
        self.screenshot_task: Optional[asyncio.Task] = None
        self.xvfb_proc = None
        self.vnc_proc = None
        self.novnc_proc = None
        self.active_strategy = "single"
        self.active_primary_model = ""
        self.active_secondary_model = ""
        self.active_council_members: list[str] = []
        self.generated_plan: str = ""  # plan text from planner_executor strategy
        self.plan_progress: list = []  # live plan items with status (pending/current/done/skipped)
        self.last_step_start_time: float = 0.0  # monotonic timestamp of when current step began
        self.last_step_number: int = 0           # step number currently in progress
        self.stall_council_fired_for_step: int = -1  # step number we already fired stall council for
        self.stall_monitor_task: Optional[asyncio.Task] = None
        self.browser_session = None  # Persistent BrowserSession across tasks
        # Previous task context (for session continuity)
        self.previous_task: str = ""
        self.previous_result: str = ""
        self.previous_model: str = ""
        # Human-in-the-loop: allow user to send questions/messages to the running agent
        self.pending_user_question: Optional[str] = None  # question text waiting for agent to pick up
        self.user_answer: Optional[str] = None            # answer from user waiting to be injected
        self.user_answer_event: Optional[asyncio.Event] = None  # signal that answer arrived
        self.awaiting_user_input: bool = False             # True while agent is paused waiting for user

state = AgentState()

# ─── Display Management ────────────────────────────────────────────────────────

def start_virtual_display():
    """Start Xvfb virtual display, or reuse an existing one."""
    logger.info(f"Starting Xvfb on display {DISPLAY} ({SCREEN_WIDTH}x{SCREEN_HEIGHT}x{SCREEN_DEPTH})")

    # Check if Xvfb is already running on this display
    result = subprocess.run(["pgrep", "-af", f"Xvfb {DISPLAY}"], capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        logger.info(f"Xvfb already running on {DISPLAY}, reusing: {result.stdout.strip().splitlines()[0]}")
        os.environ["DISPLAY"] = DISPLAY
        state.xvfb_proc = None
        return

    # Kill any existing Xvfb on this display
    subprocess.run(["pkill", "-f", f"Xvfb {DISPLAY}"], capture_output=True)
    time.sleep(0.5)

    state.xvfb_proc = subprocess.Popen(
        [
            "Xvfb", DISPLAY,
            "-screen", "0", f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}x{SCREEN_DEPTH}",
            "-ac",  # disable access control
            "+extension", "RANDR",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(1)

    if state.xvfb_proc.poll() is not None:
        stderr = state.xvfb_proc.stderr.read().decode()
        raise RuntimeError(f"Xvfb failed to start: {stderr}")

    os.environ["DISPLAY"] = DISPLAY
    logger.info(f"Xvfb started (PID: {state.xvfb_proc.pid})")


def start_vnc_server():
    """Start x11vnc server attached to the virtual display."""
    logger.info(f"Starting x11vnc on display {DISPLAY}, VNC port {VNC_PORT}")

    subprocess.run(["pkill", "-f", "x11vnc"], capture_output=True)
    time.sleep(0.3)

    state.vnc_proc = subprocess.Popen(
        [
            "x11vnc",
            "-display", DISPLAY,
            "-rfbport", str(VNC_PORT),
            "-nopw",           # no password (internal only)
            "-shared",         # allow multiple viewers
            "-forever",        # don't exit after first client disconnects
            "-noxdamage",
            "-cursor", "most",
            "-ncache", "10",
            "-xkb",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(1)

    if state.vnc_proc.poll() is not None:
        stderr = state.vnc_proc.stderr.read().decode()
        logger.warning(f"x11vnc stderr: {stderr}")

    logger.info(f"x11vnc started (PID: {state.vnc_proc.pid})")


def start_novnc():
    """Start noVNC web proxy for browser-based VNC viewing."""
    logger.info(f"Starting noVNC on port {NOVNC_PORT}")

    subprocess.run(["pkill", "-f", "websockify.*6080"], capture_output=True)
    time.sleep(0.3)

    # Find noVNC web directory
    novnc_dirs = [
        "/usr/share/novnc",
        "/usr/share/websockify/rebind.so",
    ]
    novnc_web = "/usr/share/novnc"

    # Find websockify binary: prefer venv, fallback to system
    websockify_bin = "websockify"
    venv_websockify = os.path.join(os.path.dirname(sys.executable), "websockify")
    if os.path.isfile(venv_websockify):
        websockify_bin = venv_websockify

    state.novnc_proc = subprocess.Popen(
        [
            websockify_bin,
            "--web", novnc_web,
            str(NOVNC_PORT),
            f"localhost:{VNC_PORT}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(1)
    logger.info(f"noVNC/websockify started (PID: {state.novnc_proc.pid})")


# ─── Screenshot Service ────────────────────────────────────────────────────────

async def take_screenshot() -> Optional[str]:
    """Capture screenshot from Xvfb display, return as base64 PNG."""
    try:
        # Use ImageMagick 'import' to capture the root window directly as PNG
        proc = await asyncio.create_subprocess_exec(
            "import", "-display", DISPLAY, "-window", "root", "png:-",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        png_data, stderr = await proc.communicate()

        if proc.returncode != 0 or not png_data:
            return None

        return base64.b64encode(png_data).decode("ascii")
    except Exception as e:
        logger.error(f"Screenshot error: {e}")
        return None


async def screenshot_broadcast_loop():
    """Continuously capture and broadcast screenshots to WebSocket clients."""
    while True:
        try:
            if state.ws_clients:
                screenshot = await take_screenshot()
                if screenshot:
                    msg = json.dumps({
                        "type": "screenshot",
                        "data": screenshot,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    disconnected = set()
                    for ws in state.ws_clients:
                        try:
                            await ws.send_text(msg)
                        except Exception:
                            disconnected.add(ws)
                    state.ws_clients -= disconnected

            await asyncio.sleep(SCREENSHOT_INTERVAL)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Screenshot loop error: {e}")
            await asyncio.sleep(1)


# ─── Action Summary Helper ────────────────────────────────────────────────────

def _summarize_action(act) -> str:
    """Convert a browser-use ActionModel into a short human-readable description."""
    try:
        raw = str(act)
        # Navigate
        if "NavigateAction" in raw:
            url = getattr(act, "navigate", None)
            if url and hasattr(url, "url"):
                return f"Navigate to {url.url}"
            # fallback: parse from string
            import re
            m = re.search(r"url='([^']*)'", raw)
            if m:
                return f"Navigate to {m.group(1)}"
            return "Navigate to page"

        # Click
        if "ClickAction" in raw or "ClickElement" in raw:
            idx_match = None
            import re
            idx_match = re.search(r"index=(\d+)", raw)
            idx = idx_match.group(1) if idx_match else "?"
            return f"Click element [{idx}]"

        # Input / type text
        if "InputTextAction" in raw or "InputAction" in raw:
            import re
            idx_match = re.search(r"index=(\d+)", raw)
            text_match = re.search(r"text='([^']*)'", raw)
            idx = idx_match.group(1) if idx_match else "?"
            text = text_match.group(1)[:40] if text_match else "..."
            return f"Type \"{text}\" into element [{idx}]"

        # Scroll
        if "ScrollAction" in raw:
            direction = "down" if "down=True" in raw else "up"
            return f"Scroll {direction}"

        # Extract / read content
        if "ExtractAction" in raw:
            return "Extract page content"

        # Done
        if "DoneAction" in raw:
            import re
            text_match = re.search(r"text=['\"]([^'\"]{0,60})", raw)
            if text_match:
                return f"Done: {text_match.group(1)}..."
            return "Task completed"

        # GoBack
        if "GoBack" in raw:
            return "Go back"

        # SendKeys
        if "SendKeys" in raw or "KeyAction" in raw:
            import re
            key_match = re.search(r"keys?='([^']*)'", raw)
            key = key_match.group(1) if key_match else "keys"
            return f"Press {key}"

        # Tab actions
        if "SwitchTab" in raw or "NewTab" in raw:
            return "Switch browser tab"

        # Fallback: truncate raw string
        return raw[:80]
    except Exception:
        return str(act)[:80]


# ─── Plan Progress Auto-Update ────────────────────────────────────────────────

_plan_highest_done_idx = -1  # Track highest completed step index across plan resets


def _auto_update_plan_progress(agent, log_entry: dict):
    """Proactively mark plan steps as done/current based on what actually happened.

    browser-use only updates plan step statuses when the LLM explicitly outputs
    `current_plan_item`. Many models skip this field, leaving steps stuck on
    'pending' even when the work is done. This function matches the agent's
    thought/actions/URL against plan step text to infer progress.

    Also handles browser-use plan_update resets: if the LLM replaces the plan,
    we re-apply progress up to the highest previously completed step.
    """
    global _plan_highest_done_idx
    plan = agent.state.plan
    if not plan:
        return

    # Check if browser-use reset our progress (plan_update replaced the plan)
    # If all steps are pending/current but we previously had done steps, re-apply
    done_count = sum(1 for item in plan if item.status == "done")
    if done_count == 0 and _plan_highest_done_idx >= 0:
        # Plan was reset by browser-use. Re-mark previously completed steps.
        for i in range(min(_plan_highest_done_idx + 1, len(plan))):
            plan[i].status = "done"
        next_idx = min(_plan_highest_done_idx + 1, len(plan) - 1)
        if plan[next_idx].status != "done":
            plan[next_idx].status = "current"
        logger.info(
            f"Plan auto-update: restored progress after plan_update reset "
            f"({_plan_highest_done_idx + 1} steps re-marked done)"
        )

    # Gather context from the step
    thought = (log_entry.get("thought") or "").lower()
    actions_text = " ".join(str(a) for a in log_entry.get("action_summaries", [])).lower()
    url = (log_entry.get("url") or "").lower()
    evaluation = (log_entry.get("evaluation") or "").lower()
    memory = (log_entry.get("memory") or "").lower()
    context = f"{thought} {actions_text} {url} {evaluation} {memory}"

    # Find the first 'current' step
    current_idx = None
    for i, item in enumerate(plan):
        if item.status == "current":
            current_idx = i
            break

    if current_idx is None:
        # No current step -- find first pending and make it current
        for i, item in enumerate(plan):
            if item.status == "pending":
                item.status = "current"
                current_idx = i
                break
        if current_idx is None:
            return  # All done or skipped

    current_step = plan[current_idx]
    step_text = current_step.text.lower()

    # Extract key action words from the plan step for matching
    step_keywords = [w for w in re.split(r'[\s,;.!?()\[\]]+', step_text) if len(w) > 3]

    # Check if the current step's intent appears to be satisfied
    if step_keywords:
        matches = sum(1 for kw in step_keywords if kw in context)
        match_ratio = matches / len(step_keywords)
    else:
        match_ratio = 0

    # Also check for explicit success signals in the evaluation
    success_signals = ["success" in evaluation, "completed" in evaluation,
                       "done" in evaluation, "verdict: success" in evaluation]
    has_success = any(success_signals)

    # Mark current step as done if we have good evidence
    if match_ratio >= 0.4 or has_success:
        plan[current_idx].status = "done"
        # Track highest done index
        if current_idx > _plan_highest_done_idx:
            _plan_highest_done_idx = current_idx
        # Advance to next pending step
        for i in range(current_idx + 1, len(plan)):
            if plan[i].status == "pending":
                plan[i].status = "current"
                break
        logger.debug(
            f"Plan auto-update: step {current_idx} '{current_step.text[:50]}' -> done "
            f"(match={match_ratio:.0%}, success={has_success})"
        )


# ─── Agent Step Callback ───────────────────────────────────────────────────────

async def on_agent_step(browser_state, agent_output, step_number):
    """Called after each agent step -- broadcasts to all connected clients."""
    state.step_count = step_number
    state.last_step_start_time = time.monotonic()
    state.last_step_number = step_number

    # Extract action info -- both raw and human-readable summaries
    actions = []
    action_summaries = []
    if agent_output and hasattr(agent_output, "action") and agent_output.action:
        action_list = agent_output.action if isinstance(agent_output.action, list) else [agent_output.action]
        for act in action_list:
            actions.append(str(act))
            action_summaries.append(_summarize_action(act))

    # Extract reasoning fields from AgentOutput
    thought = ""
    evaluation = ""
    next_goal = ""
    memory = ""
    if agent_output:
        # Direct fields on AgentOutput (browser-use 0.11.9)
        if hasattr(agent_output, "thinking") and agent_output.thinking:
            thought = agent_output.thinking
        elif hasattr(agent_output, "current_state"):
            cs = agent_output.current_state
            if hasattr(cs, "thinking") and cs.thinking:
                thought = cs.thinking
            elif hasattr(cs, "thought") and cs.thought:
                thought = cs.thought

        if hasattr(agent_output, "evaluation_previous_goal") and agent_output.evaluation_previous_goal:
            evaluation = agent_output.evaluation_previous_goal
        elif hasattr(agent_output, "current_state"):
            cs = agent_output.current_state
            if hasattr(cs, "evaluation_previous_goal") and cs.evaluation_previous_goal:
                evaluation = cs.evaluation_previous_goal

        if hasattr(agent_output, "next_goal") and agent_output.next_goal:
            next_goal = agent_output.next_goal
        elif hasattr(agent_output, "current_state"):
            cs = agent_output.current_state
            if hasattr(cs, "next_goal") and cs.next_goal:
                next_goal = cs.next_goal

        if hasattr(agent_output, "memory") and agent_output.memory:
            memory = agent_output.memory
        elif hasattr(agent_output, "current_state"):
            cs = agent_output.current_state
            if hasattr(cs, "memory") and cs.memory:
                memory = cs.memory

    # Determine current model (may change mid-run for fallback_chain)
    current_model = state.active_primary_model
    if state.agent and hasattr(state.agent, "llm") and hasattr(state.agent.llm, "model"):
        current_model = state.agent.llm.model

    log_entry = {
        "step": step_number,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "thought": thought,
        "evaluation": evaluation,
        "next_goal": next_goal,
        "memory": memory,
        "actions": actions,
        "action_summaries": action_summaries,
        "url": getattr(browser_state, "url", "") if browser_state else "",
        "model": current_model,
        "strategy": state.active_strategy,
    }
    state.action_log.append(log_entry)

    # Extract plan progress from browser-use agent state
    # Also proactively update plan step statuses based on what actually happened
    plan_items = None
    if state.agent and hasattr(state.agent, "state") and hasattr(state.agent.state, "plan"):
        bu_plan = state.agent.state.plan
        if bu_plan:
            _auto_update_plan_progress(state.agent, log_entry)
            plan_items = [
                {"text": item.text, "status": item.status}
                for item in bu_plan
            ]
            # Store latest plan snapshot on our state for status API
            state.plan_progress = plan_items

    # Broadcast to WebSocket clients
    msg = json.dumps({"type": "step", "data": log_entry})
    disconnected = set()
    for ws in state.ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            disconnected.add(ws)
    state.ws_clients -= disconnected

    # Broadcast plan progress if available
    if plan_items is not None:
        plan_msg = json.dumps({"type": "plan_progress", "data": {"items": plan_items, "step": step_number}})
        for ws in state.ws_clients:
            try:
                await ws.send_text(plan_msg)
            except Exception:
                pass

    logger.info(f"Step {step_number}: {thought[:100]}..." if thought else f"Step {step_number}")

    # ── Grounding verification: detect common failures and inject corrective feedback ──
    await _verify_grounding(state.agent, log_entry, step_number)

    # ── Human-in-the-loop: check if user has a pending message for the agent ──
    user_msg = await _check_and_handle_user_message()
    if user_msg and state.agent:
        _inject_user_message_to_agent(state.agent, user_msg)


async def _verify_grounding(agent, log_entry: dict, step_number: int):
    """Detect grounding failures after each step and inject corrective hints."""
    if not agent:
        return

    history = agent.history
    if not history or not history.history:
        return

    hints = []
    last_step = history.history[-1]

    # Check 1: Action errors (element not found, stale index, etc.)
    for r in last_step.result:
        if r.error:
            err = str(r.error).lower()
            if "not available" in err or "not found" in err or "no element" in err:
                hints.append(
                    "GROUNDING: An element index was stale or unavailable. "
                    "The page likely changed since last observation. "
                    "Look at the CURRENT element list carefully before choosing an index."
                )
            elif "timeout" in err:
                hints.append(
                    "GROUNDING: Action timed out. The element may be obscured by an overlay, "
                    "modal, or cookie banner. Check if something is blocking the target element."
                )

    # Check 2: Repeated same action (grounding loop - clicking same thing)
    if len(history.history) >= 3:
        recent_actions = []
        for h in history.history[-3:]:
            if h.model_output and hasattr(h.model_output, "action") and h.model_output.action:
                recent_actions.append(str(h.model_output.action[0]) if h.model_output.action else "")
        if len(recent_actions) == 3 and recent_actions[0] == recent_actions[1] == recent_actions[2]:
            hints.append(
                "GROUNDING: You have repeated the exact same action 3 times in a row. "
                "This means the action is not having the expected effect. "
                "Try a completely different approach: use a different element, scroll, "
                "or use extract to understand the page structure better."
            )

    # Check 3: URL unchanged after navigation-like action
    if len(history.history) >= 2:
        prev_url = getattr(history.history[-2].state, "url", "") if history.history[-2].state else ""
        curr_url = getattr(last_step.state, "url", "") if last_step.state else ""
        # Check if the action was a click or navigate but URL didn't change
        actions_desc = ""
        if last_step.model_output and hasattr(last_step.model_output, "action"):
            actions_desc = str(last_step.model_output.action).lower()
        if ("navigate" in actions_desc or "go_to" in actions_desc) and prev_url == curr_url and curr_url:
            hints.append(
                "GROUNDING: You attempted navigation but the URL did not change. "
                "The navigation may have failed or been blocked. "
                "Verify the URL is correct and try again, or check for redirects."
            )

    # Inject hints into agent's long_term_memory if any issues detected
    if hints:
        combined = "\n".join(hints)
        if agent.state.last_result:
            existing = agent.state.last_result[-1].long_term_memory or ""
            agent.state.last_result[-1].long_term_memory = (
                (existing + "\n" + combined) if existing else combined
            )
        else:
            from browser_use.agent.views import ActionResult as AR
            agent.state.last_result = [AR(long_term_memory=combined)]
        logger.info(f"Grounding hints injected at step {step_number}: {combined[:150]}")


# ─── Human-in-the-Loop ──────────────────────────────────────────────────────────

async def _broadcast_question_to_user(question: str, step: int, source: str = "agent"):
    """Send a question to all connected dashboard clients."""
    msg = json.dumps({
        "type": "ask_user",
        "data": {
            "question": question,
            "step": step,
            "source": source,  # "agent" = agent is paused, "user" = user initiated
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    })
    disconnected = set()
    for ws in state.ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            disconnected.add(ws)
    state.ws_clients -= disconnected


async def _check_and_handle_user_message():
    """Check if a user sent a message/question to the agent. If so, pause and wait for answer.
    Called from on_agent_step after each step completes.
    Returns the user's message (or None if no pending question)."""
    if not state.pending_user_question:
        return None

    question = state.pending_user_question
    step = state.last_step_number
    state.pending_user_question = None

    logger.info(f"Human-in-the-loop: pausing agent at step {step} for user question: {question[:100]}")

    # Set up the wait mechanism
    state.user_answer_event = asyncio.Event()
    state.awaiting_user_input = True
    state.user_answer = None

    # Broadcast question to dashboard
    await _broadcast_question_to_user(question, step, source="user")

    # Broadcast status update so dashboard knows agent is paused
    await broadcast_status()

    # Wait for user answer (with timeout)
    try:
        await asyncio.wait_for(state.user_answer_event.wait(), timeout=300.0)  # 5 min timeout
    except asyncio.TimeoutError:
        logger.warning("Human-in-the-loop: timeout waiting for user answer (300s)")
        state.awaiting_user_input = False
        state.user_answer_event = None
        # Broadcast resumption
        await broadcast_status()
        return None

    answer = state.user_answer
    state.awaiting_user_input = False
    state.user_answer = None
    state.user_answer_event = None

    logger.info(f"Human-in-the-loop: received answer: {str(answer)[:200]}")

    # Broadcast that we've resumed
    await broadcast_status()

    return answer


def _inject_user_message_to_agent(agent, message: str):
    """Inject a user message into the agent's long_term_memory so it reads it on the next step."""
    feedback = f"[HUMAN MESSAGE] The user sent the following message to you: {message}"
    if agent.state.last_result:
        existing = agent.state.last_result[-1].long_term_memory or ""
        agent.state.last_result[-1].long_term_memory = (
            (existing + "\n" + feedback) if existing else feedback
        )
    else:
        from browser_use.agent.views import ActionResult as AR
        agent.state.last_result = [AR(long_term_memory=feedback)]
    logger.info(f"Injected user message into agent long_term_memory: {message[:150]}")


# ─── Action Format Fixer (monkey-patch) ──────────────────────────────────────────

_ACTION_FIELD_FIXES = {
    # Common model mistakes: wrong field names -> correct field names
    "element_index": "index",
    "element_id": "index",
    "elem_index": "index",
    "elementIndex": "index",
    "element": "index",
    "target_index": "index",
    "input_text": "text",     # inside input action
    "value": "text",           # inside input action
}

def _fix_action_json(raw_json: str) -> str:
    """Fix common field-name mistakes in LLM-generated action JSON before Pydantic validates."""
    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return raw_json

    changed = False
    if "action" in data and isinstance(data["action"], list):
        for action_obj in data["action"]:
            if not isinstance(action_obj, dict):
                continue
            for action_name, action_params in list(action_obj.items()):
                if not isinstance(action_params, dict):
                    continue
                for wrong_key, correct_key in _ACTION_FIELD_FIXES.items():
                    if wrong_key in action_params and correct_key not in action_params:
                        action_params[correct_key] = action_params.pop(wrong_key)
                        changed = True
                        logger.debug(f"Action fix: {action_name}.{wrong_key} -> {correct_key}")

    if changed:
        logger.info(f"Fixed malformed action JSON field names")
        return json.dumps(data)
    return raw_json


def _install_action_format_fixer():
    """Monkey-patch browser-use's ChatOpenAI.ainvoke to fix malformed action JSON before Pydantic validation."""
    try:
        from browser_use import ChatOpenAI
        _original_ainvoke = ChatOpenAI.ainvoke

        async def _patched_ainvoke(self, messages, output_format=None, **kwargs):
            result = await _original_ainvoke(self, messages, output_format=output_format, **kwargs)
            return result

        # Instead of patching ainvoke (which happens after validation), we need to patch
        # the AgentOutput.model_validate_json to fix JSON before validation
        from browser_use.agent.views import AgentOutput
        _original_validate = AgentOutput.model_validate_json

        @classmethod
        def _patched_validate(cls, json_data, *args, **kwargs):
            if isinstance(json_data, (str, bytes)):
                json_str = json_data if isinstance(json_data, str) else json_data.decode()
                json_str = _fix_action_json(json_str)
                return _original_validate.__func__(cls, json_str, *args, **kwargs)
            return _original_validate.__func__(cls, json_data, *args, **kwargs)

        AgentOutput.model_validate_json = _patched_validate
        logger.info("Installed action format fixer (monkey-patched AgentOutput.model_validate_json)")
    except Exception as e:
        logger.warning(f"Could not install action format fixer: {e}")


# Install the fixer at module load time
_install_action_format_fixer()


# ─── Request Models ──────────────────────────────────────────────────────────────

class ModelConfig(BaseModel):
    strategy: TypingLiteral["single", "fallback_chain", "planner_executor", "consensus", "council"] = "single"
    primary_model: str = ""
    secondary_model: str = ""
    council_members: list[str] = []  # explicit list of model IDs for the council
    use_council_planning: bool = False  # planner_executor: use council vote instead of single planner


# ─── LLM Factory ─────────────────────────────────────────────────────────────────

def create_llm(model_id: str):
    """Create a ChatOpenAI instance for a gateway model ID."""
    from browser_use import ChatOpenAI

    api_key = (os.environ.get("AI_GATEWAY_API_KEY")
               or os.environ.get("OPENAI_API_KEY")
               or os.environ.get("ANTHROPIC_API_KEY"))
    if not api_key:
        raise ValueError("No API key found. Set AI_GATEWAY_API_KEY or OPENAI_API_KEY")

    custom_headers_raw = os.environ.get("ANTHROPIC_CUSTOM_HEADERS", "")
    default_headers = {}
    if custom_headers_raw:
        for part in custom_headers_raw.split(","):
            if ":" in part:
                k, v = part.split(":", 1)
                default_headers[k.strip()] = v.strip()

    gateway_base = "https://ai-gateway.happycapy.ai/api/v1"
    return ChatOpenAI(
        model=model_id,
        api_key=api_key,
        base_url=gateway_base,
        temperature=0,
        default_headers=default_headers if default_headers else None,
    )


async def _per_step_judge(agent, judge_llm, task: str) -> None:
    """Called via on_step_end when consensus strategy is active.

    Evaluates the last step and broadcasts the verdict to the dashboard.
    On WARN or FAIL verdicts, injects feedback into the agent's long_term_memory
    so the executing agent sees the critique and can adjust its approach.
    """
    from browser_use.llm.messages import SystemMessage, UserMessage

    try:
        history = agent.history
        if not history or not history.history:
            return

        step_num = len(history.history)
        last_step = history.history[-1]

        # Build a concise summary of the last step
        thought = ""
        actions_desc = ""
        if last_step.model_output:
            if hasattr(last_step.model_output, "current_state"):
                cs = last_step.model_output.current_state
                if hasattr(cs, "thought") and cs.thought:
                    thought = cs.thought
            if hasattr(last_step.model_output, "action") and last_step.model_output.action:
                actions_desc = "; ".join(str(a) for a in last_step.model_output.action)

        # Check results for errors
        result_info = ""
        for r in last_step.result:
            if r.error:
                result_info += f" Error: {r.error}"
            if r.extracted_content:
                result_info += f" Content: {r.extracted_content[:200]}"

        url = getattr(last_step.state, "url", "") if last_step.state else ""

        # Summarise progress so far (brief)
        total_steps = len(history.history)
        prev_actions = []
        for h in history.history[:-1]:
            if h.model_output and hasattr(h.model_output, "action") and h.model_output.action:
                prev_actions.append("; ".join(str(a) for a in h.model_output.action[:1]))
        prev_summary = " -> ".join(prev_actions[-5:]) if prev_actions else "(first step)"

        system_msg = SystemMessage(content=(
            "You are a browser automation quality judge. After each step you must evaluate whether "
            "the agent is making correct progress toward the task. Respond with EXACTLY this format:\n"
            "VERDICT: PASS | WARN | FAIL\n"
            "REASON: <one sentence explanation>\n"
            "SUGGESTION: <one sentence on what the agent should do differently, or 'none' if PASS>\n\n"
            "Rules:\n"
            "- PASS: The step clearly advances toward the goal. SUGGESTION should be 'none'.\n"
            "- WARN: The step is questionable, might be off-track, or sub-optimal but not catastrophic.\n"
            "- FAIL: The step is clearly wrong, navigated to wrong page, filled wrong data, or is stuck in a loop.\n"
            "Be concise. One sentence each."
        ))

        user_msg = UserMessage(content=(
            f"TASK: {task}\n"
            f"STEP {step_num}: Thought: {thought}\n"
            f"Actions: {actions_desc}\n"
            f"URL: {url}\n"
            f"Result: {result_info.strip()}\n"
            f"Previous steps: {prev_summary}\n"
        ))

        response = await judge_llm.ainvoke([system_msg, user_msg])
        verdict_text = response.completion.strip()

        # Parse verdict
        verdict = "PASS"
        reason = verdict_text
        suggestion = ""
        for line in verdict_text.split("\n"):
            line_upper = line.strip().upper()
            if line_upper.startswith("VERDICT:"):
                v = line_upper.replace("VERDICT:", "").strip()
                if "FAIL" in v:
                    verdict = "FAIL"
                elif "WARN" in v:
                    verdict = "WARN"
                else:
                    verdict = "PASS"
            if line.strip().upper().startswith("REASON:"):
                reason = line.strip()[7:].strip()
            if line.strip().upper().startswith("SUGGESTION:"):
                suggestion = line.strip()[11:].strip()

        logger.info(f"Judge step {step_num}: {verdict} - {reason}")

        # ── Inject feedback into agent on WARN/FAIL so it adapts ──
        if verdict in ("WARN", "FAIL"):
            feedback_msg = (
                f"[JUDGE FEEDBACK - Step {step_num}: {verdict}] "
                f"{reason}"
            )
            if suggestion and suggestion.lower() != "none":
                feedback_msg += f" Suggestion: {suggestion}"

            if agent.state.last_result:
                # Append to existing long_term_memory if present
                existing = agent.state.last_result[-1].long_term_memory or ""
                agent.state.last_result[-1].long_term_memory = (
                    (existing + "\n" + feedback_msg) if existing else feedback_msg
                )
            else:
                from browser_use.agent.views import ActionResult as AR
                agent.state.last_result = [AR(long_term_memory=feedback_msg)]

            logger.info(f"Judge feedback injected into agent: {feedback_msg[:100]}")

        # Store in action log (append to the latest entry)
        if state.action_log:
            state.action_log[-1]["judge_verdict"] = verdict
            state.action_log[-1]["judge_reason"] = reason
            if suggestion:
                state.action_log[-1]["judge_suggestion"] = suggestion

        # Broadcast judge verdict to dashboard
        msg = json.dumps({
            "type": "judge_verdict",
            "data": {
                "step": step_num,
                "verdict": verdict,
                "reason": reason,
                "suggestion": suggestion,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        })
        disconnected = set()
        for ws in state.ws_clients:
            try:
                await ws.send_text(msg)
            except Exception:
                disconnected.add(ws)
        state.ws_clients -= disconnected

    except Exception as e:
        logger.warning(f"Per-step judge failed on step: {e}")


async def _council_convene(
    agent, council_llms: dict, task: str, failure_count: int,
    trigger_reason: str = "consecutive_failures",
) -> None:
    """Convene the LLM council when the agent is stuck.

    All available models are queried in parallel.  Their diagnoses are
    synthesised into:
      1. Root-cause explanation (injected as long_term_memory so the agent
         knows *why* it was failing).
      2. Concrete next-action advice.
      3. Optional revised plan (replaces agent.state.plan).

    Args:
        trigger_reason: "consecutive_failures" or "action_loop" -- adjusts
            the prompt so the council knows *why* it was convened.
    """
    from browser_use.llm.messages import SystemMessage, UserMessage

    try:
        history = agent.history
        if not history or not history.history:
            return

        # ── Build failure context ──
        recent_steps = history.history[-min(5, len(history.history)):]
        step_summaries = []
        for h in recent_steps:
            thought = ""
            actions_desc = ""
            error_info = ""
            if h.model_output:
                if hasattr(h.model_output, "current_state"):
                    cs = h.model_output.current_state
                    if hasattr(cs, "thought") and cs.thought:
                        thought = cs.thought
                if hasattr(h.model_output, "action") and h.model_output.action:
                    actions_desc = "; ".join(str(a) for a in h.model_output.action)
            for r in h.result:
                if r.error:
                    error_info += f" ERROR: {r.error}"
            url = getattr(h.state, "url", "") if h.state else ""
            step_summaries.append(
                f"  Thought: {thought}\n  Actions: {actions_desc}\n  URL: {url}\n  {error_info.strip()}"
            )

        context = "\n---\n".join(
            f"Step {i+1}:\n{s}" for i, s in enumerate(step_summaries)
        )

        if trigger_reason == "action_loop":
            situation_desc = (
                "The agent is STUCK IN A LOOP -- it keeps repeating the exact same "
                "action on the same page without making any progress. Each step "
                "'succeeds' (no error) but the agent is not advancing toward the goal. "
                "This is often caused by: the target element not existing, a click not "
                "registering, waiting for something that never happens, or the agent "
                "misidentifying which element to interact with."
            )
            trigger_label = f"REPEATED IDENTICAL ACTIONS: {failure_count} times"
        elif trigger_reason == "step_stall":
            situation_desc = (
                "The agent is STALLED on a single step -- it has been running the same "
                f"step for {failure_count} seconds without completing. The step appears "
                "to be hanging. This is often caused by: waiting for a page load that "
                "never finishes, a modal/dialog blocking interaction, a network request "
                "timing out, the browser being unresponsive, or the agent waiting for "
                "an element that will never appear."
            )
            trigger_label = f"STEP STALLED: {failure_count} seconds on current step"
        else:
            situation_desc = (
                "The agent has failed the same step multiple times with errors."
            )
            trigger_label = f"CONSECUTIVE FAILURES: {failure_count}"

        system_msg = SystemMessage(content=(
            "You are a senior browser-automation expert on an LLM council. "
            f"{situation_desc} Your job:\n"
            "1. DIAGNOSE: What is the root cause of the repeated failure/loop? (1-2 sentences)\n"
            "2. ADVICE: What concrete, DIFFERENT action should the agent take next? "
            "Do NOT suggest the same action the agent is already doing. (1-2 sentences)\n"
            "3. REPLAN: If the original approach is fundamentally broken, provide a numbered "
            "revised plan (3-8 steps). If no replan is needed, write REPLAN: NONE.\n\n"
            "Format your response EXACTLY as:\n"
            "DIAGNOSE: <root cause>\n"
            "ADVICE: <next action>\n"
            "REPLAN: NONE | <numbered steps>"
        ))

        user_msg = UserMessage(content=(
            f"TASK: {task}\n"
            f"{trigger_label}\n"
            f"RECENT STEPS (oldest first):\n{context}\n"
        ))

        # ── Query all council members in parallel ──
        async def _query_member(model_id, llm):
            try:
                resp = await llm.ainvoke([system_msg, user_msg])
                return {"model": model_id, "response": resp.completion.strip()}
            except Exception as e:
                logger.warning(f"Council member {model_id} failed: {e}")
                return {"model": model_id, "response": None}

        tasks = [
            _query_member(mid, llm)
            for mid, llm in council_llms.items()
        ]
        results = await asyncio.gather(*tasks)
        valid = [r for r in results if r["response"]]

        if not valid:
            logger.warning("Council convened but no models responded")
            return

        # ── Parse each member's response ──
        diagnoses = []
        advices = []
        replans = []

        for r in valid:
            text = r["response"]
            diag = ""
            adv = ""
            replan_text = ""
            for line in text.split("\n"):
                lu = line.strip().upper()
                if lu.startswith("DIAGNOSE:"):
                    diag = line.strip()[9:].strip()
                elif lu.startswith("ADVICE:"):
                    adv = line.strip()[7:].strip()
                elif lu.startswith("REPLAN:"):
                    rest = line.strip()[7:].strip()
                    if rest.upper() != "NONE":
                        replan_text = rest
                elif replan_text:
                    # continuation of replan (numbered lines)
                    replan_text += "\n" + line
            if diag:
                diagnoses.append(f"[{r['model'].split('/')[-1]}] {diag}")
            if adv:
                advices.append(f"[{r['model'].split('/')[-1]}] {adv}")
            if replan_text.strip():
                replans.append({"model": r["model"], "plan": replan_text.strip()})

        # ── Synthesise council verdict ──
        diagnosis_summary = " | ".join(diagnoses) if diagnoses else "No diagnosis available"
        advice_summary = " | ".join(advices) if advices else "Try a different approach"

        # Pick the replan with the most steps (heuristic for thoroughness)
        chosen_replan = None
        if replans:
            chosen_replan = max(replans, key=lambda p: len(p["plan"].split("\n")))

        if trigger_reason == "action_loop":
            trigger_desc = f"{failure_count} identical looped actions"
        elif trigger_reason == "step_stall":
            trigger_desc = f"step stalled for {failure_count}s"
        else:
            trigger_desc = f"{failure_count} failures"
        council_msg = (
            f"[LLM COUNCIL - {len(valid)} models convened after {trigger_desc}]\n"
            f"DIAGNOSIS: {diagnosis_summary}\n"
            f"ADVICE: {advice_summary}"
        )
        if chosen_replan:
            council_msg += f"\nREVISED PLAN (from {chosen_replan['model'].split('/')[-1]}):\n{chosen_replan['plan']}"

        logger.info(f"Council verdict ({len(valid)} members): {diagnosis_summary[:150]}")

        # ── Inject feedback into agent ──
        # 1. Set long_term_memory on last result so agent sees it next step
        if agent.state.last_result:
            agent.state.last_result[-1].long_term_memory = council_msg
        else:
            from browser_use.agent.views import ActionResult as AR
            agent.state.last_result = [AR(long_term_memory=council_msg)]

        # 2. Replace plan if council suggested a replan
        if chosen_replan:
            from browser_use.agent.views import PlanItem
            new_plan_lines = [
                ln.strip() for ln in chosen_replan["plan"].split("\n")
                if ln.strip() and ln.strip()[0].isdigit()
            ]
            if new_plan_lines:
                agent.state.plan = [
                    PlanItem(text=ln) for ln in new_plan_lines
                ]
                agent.state.plan[0].status = "current"
                logger.info(f"Council replaced plan with {len(new_plan_lines)} steps")

        # ── Broadcast to dashboard ──
        member_details = []
        for r in valid:
            text = r["response"]
            member_details.append({
                "model": r["model"],
                "diagnosis": next((d.split("] ", 1)[-1] for d in diagnoses if r["model"].split("/")[-1] in d), ""),
                "advice": next((a.split("] ", 1)[-1] for a in advices if r["model"].split("/")[-1] in a), ""),
                "has_replan": any(p["model"] == r["model"] for p in replans),
            })

        ws_msg = json.dumps({
            "type": "council_verdict",
            "data": {
                "step": len(history.history),
                "failure_count": failure_count,
                "trigger_reason": trigger_reason,
                "members": member_details,
                "diagnosis": diagnosis_summary,
                "advice": advice_summary,
                "has_replan": chosen_replan is not None,
                "replan_from": chosen_replan["model"].split("/")[-1] if chosen_replan else None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        })
        disconnected = set()
        for ws in state.ws_clients:
            try:
                await ws.send_text(ws_msg)
            except Exception:
                disconnected.add(ws)
        state.ws_clients -= disconnected

        # Also store in action log
        if state.action_log:
            state.action_log[-1]["council_verdict"] = {
                "members": len(valid),
                "trigger_reason": trigger_reason,
                "diagnosis": diagnosis_summary,
                "advice": advice_summary,
                "has_replan": chosen_replan is not None,
            }

    except Exception as e:
        logger.error(f"Council convene failed: {e}", exc_info=True)


def _normalize_url_for_fingerprint(url: str) -> str:
    """Normalize a URL for loop detection: strip query params and fragments."""
    from urllib.parse import urlparse
    try:
        p = urlparse(url.strip().lower())
        return f"{p.scheme}://{p.netloc}{p.path}".rstrip("/")
    except Exception:
        return url.strip().lower()


def _normalize_action_for_fingerprint(action_str: str) -> str:
    """Normalize an action string for comparison.

    Strips variable parts like coordinates, timestamps, and whitespace
    so that semantically identical actions match even with minor changes.
    """
    import re
    s = action_str.strip().lower()
    # Remove coordinate/index numbers that vary slightly (e.g., index=15 vs index=16)
    # Keep the action TYPE but normalize the index
    s = re.sub(r"index=\d+", "index=N", s)
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def _build_step_fingerprint(step) -> str:
    """Build a fingerprint string from a history step for loop detection.

    Combines the action type/descriptions and normalized URL into a string.
    Two steps with the same fingerprint are considered "the same action."
    Uses fuzzy matching: normalizes URLs (strip query params) and action
    strings (normalize indices) so that minor variations still match.
    """
    parts = []
    if step.model_output and hasattr(step.model_output, "action") and step.model_output.action:
        action_list = step.model_output.action if isinstance(step.model_output.action, list) else [step.model_output.action]
        for act in action_list:
            parts.append(_normalize_action_for_fingerprint(str(act)))
    url = getattr(step.state, "url", "") if step.state else ""
    if url:
        parts.append(_normalize_url_for_fingerprint(url))
    return "|".join(parts)


def _build_loose_fingerprint(step) -> str:
    """Build a LOOSE fingerprint that only cares about the action TYPE and URL domain.

    This catches loops where the agent keeps doing the same kind of action
    (e.g., clicking different elements on the same page) without progress.
    """
    parts = []
    if step.model_output and hasattr(step.model_output, "action") and step.model_output.action:
        action_list = step.model_output.action if isinstance(step.model_output.action, list) else [step.model_output.action]
        for act in action_list:
            # Just the action class name (e.g., ClickAction, InputAction)
            parts.append(type(act).__name__.lower() if hasattr(act, '__class__') else str(type(act.root).__name__).lower())
    url = getattr(step.state, "url", "") if step.state else ""
    if url:
        from urllib.parse import urlparse
        try:
            p = urlparse(url.strip())
            parts.append(f"{p.netloc}{p.path}".rstrip("/").lower())
        except Exception:
            parts.append(url.strip().lower())
    return "|".join(parts)


def _detect_action_loop(agent, threshold: int = COUNCIL_LOOP_THRESHOLD) -> int:
    """Check recent agent history for repeated identical or similar actions.

    Uses a tiered detection strategy:
    1. Strict match: exact same normalized action + URL (threshold steps)
    2. Loose match: same action TYPE on same page (threshold + 1 steps)
    3. Same-URL stall: agent stays on same URL doing different things with
       no extracted content/progress for threshold + 2 steps

    Returns the number of consecutive repeated steps (0 if no loop detected).
    """
    history = agent.history
    if not history or not history.history:
        return 0

    steps = history.history
    if len(steps) < threshold:
        return 0

    # ── Tier 1: Strict fingerprint match ──
    latest_fp = _build_step_fingerprint(steps[-1])
    if latest_fp:
        strict_count = 1
        for i in range(len(steps) - 2, -1, -1):
            fp = _build_step_fingerprint(steps[i])
            if fp == latest_fp:
                strict_count += 1
            else:
                break
        if strict_count >= threshold:
            return strict_count

    # ── Tier 2: Loose fingerprint match (same action type on same page) ──
    latest_loose = _build_loose_fingerprint(steps[-1])
    if latest_loose:
        loose_count = 1
        for i in range(len(steps) - 2, -1, -1):
            lf = _build_loose_fingerprint(steps[i])
            if lf == latest_loose:
                loose_count += 1
            else:
                break
        if loose_count >= threshold + 1:
            return loose_count

    # ── Tier 3: Same-URL stall (no progress despite actions) ──
    # If the agent has been on the same URL for N steps with no new
    # extracted content, it's stalled even if actions differ
    latest_url = _normalize_url_for_fingerprint(
        getattr(steps[-1].state, "url", "") if steps[-1].state else ""
    )
    if latest_url:
        url_stall_count = 1
        has_any_content = False
        for i in range(len(steps) - 2, max(len(steps) - (threshold + 3), -1), -1):
            step_url = _normalize_url_for_fingerprint(
                getattr(steps[i].state, "url", "") if steps[i].state else ""
            )
            if step_url == latest_url:
                url_stall_count += 1
                # Check if any result had extracted content
                for r in steps[i].result:
                    if r.extracted_content and len(r.extracted_content.strip()) > 10:
                        has_any_content = True
            else:
                break
        if url_stall_count >= threshold + 2 and not has_any_content:
            return url_stall_count

    return 0


# Track the last step number where a loop-triggered council was convened
_last_loop_council_step = 0


async def _council_step_hook(agent, council_llms: dict, task: str) -> None:
    """on_step_end hook for council strategy.

    Monitors two conditions and convenes the council when either is met:
    1. Consecutive failures (original behavior) -- the agent's own error tracking
    2. Action loop detection (3-tier) -- the agent repeats the same or similar action
       without making progress, even if each step "succeeds" without errors
    """
    global _last_loop_council_step

    current_step = len(agent.history.history) if agent.history and agent.history.history else 0
    failure_count = agent.state.consecutive_failures

    # Debug: log step fingerprints so we can trace loop detection
    if agent.history and agent.history.history:
        latest = agent.history.history[-1]
        fp = _build_step_fingerprint(latest)
        lfp = _build_loose_fingerprint(latest)
        url = getattr(latest.state, "url", "?") if latest.state else "?"
        logger.debug(
            f"Council hook step {current_step}: failures={failure_count} | "
            f"fp={fp[:80]} | loose={lfp[:60]} | url={url[:60]}"
        )

    # Check 1: Consecutive failures (original behavior)
    if failure_count >= COUNCIL_FAILURE_THRESHOLD:
        logger.info(
            f"Council triggered (failures): {failure_count} consecutive failures "
            f"(threshold={COUNCIL_FAILURE_THRESHOLD})"
        )
        await _council_convene(agent, council_llms, task, failure_count)
        return

    # Check 2: Action loop detection
    loop_count = _detect_action_loop(agent, COUNCIL_LOOP_THRESHOLD)
    if loop_count >= COUNCIL_LOOP_THRESHOLD:
        # Respect cooldown to avoid council-triggering its own loop
        steps_since_last = current_step - _last_loop_council_step
        if steps_since_last < COUNCIL_LOOP_COOLDOWN:
            logger.debug(
                f"Loop detected ({loop_count} repeats) but in cooldown "
                f"({steps_since_last}/{COUNCIL_LOOP_COOLDOWN} steps since last council)"
            )
            return

        _last_loop_council_step = current_step
        logger.info(
            f"Council triggered (loop): {loop_count} identical actions detected "
            f"(threshold={COUNCIL_LOOP_THRESHOLD})"
        )
        await _council_convene(
            agent, council_llms, task, loop_count, trigger_reason="action_loop"
        )


async def _stall_monitor(agent, council_llms: dict, task: str) -> None:
    """Background coroutine that checks for stalled steps.

    Runs every 10 seconds while the agent is active. If the current step
    has been running longer than COUNCIL_STALL_TIMEOUT, convenes the council
    with trigger_reason="step_stall". Only fires once per stalled step.
    """
    try:
        while True:
            await asyncio.sleep(10)

            if not state.is_running or state.last_step_start_time == 0.0:
                continue

            elapsed = time.monotonic() - state.last_step_start_time
            current_step = state.last_step_number

            if elapsed >= COUNCIL_STALL_TIMEOUT and current_step != state.stall_council_fired_for_step:
                state.stall_council_fired_for_step = current_step
                logger.info(
                    f"Council triggered (stall): step {current_step} has been running "
                    f"for {elapsed:.0f}s (threshold={COUNCIL_STALL_TIMEOUT}s)"
                )

                # Broadcast a stall warning to dashboard before council convenes
                stall_msg = json.dumps({
                    "type": "stall_detected",
                    "data": {
                        "step": current_step,
                        "elapsed_seconds": round(elapsed, 1),
                        "threshold": COUNCIL_STALL_TIMEOUT,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                })
                disconnected = set()
                for ws in state.ws_clients:
                    try:
                        await ws.send_text(stall_msg)
                    except Exception:
                        disconnected.add(ws)
                state.ws_clients -= disconnected

                await _council_convene(
                    agent, council_llms, task,
                    failure_count=int(elapsed),
                    trigger_reason="step_stall",
                )

    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Stall monitor error: {e}", exc_info=True)


async def _generate_plan(planner_llm, task: str) -> str:
    """Use a reasoning model to produce a numbered high-level plan for the browser agent."""
    from browser_use.llm.messages import SystemMessage, UserMessage

    system_msg = SystemMessage(content=(
        "You are a browser automation planning expert. Given a user task, "
        "produce a short numbered plan of high-level actions a browser agent should follow.\n"
        "Rules:\n"
        "- Each step = ONE short sentence (max 10 words)\n"
        "- Use action verbs: Navigate, Click, Type, Extract, Scroll, Verify, Wait\n"
        "- 4-8 steps total. Be specific (include URLs, button names, field names when obvious)\n"
        "- No explanations, no code, no filler words\n"
        "Example:\n"
        "1. Navigate to google.com\n"
        "2. Type 'search query' in search box\n"
        "3. Click Google Search button\n"
        "4. Extract titles of top 3 results\n"
        "5. Report the extracted titles"
    ))
    user_msg = UserMessage(content=f"Task: {task}")

    try:
        response = await planner_llm.ainvoke([system_msg, user_msg])
        plan_text = response.completion
        logger.info(f"Generated plan ({len(plan_text)} chars):\n{plan_text}")
        return plan_text
    except Exception as e:
        logger.warning(f"Plan generation failed ({e}), using generic plan")
        return (
            f"1. Navigate to the relevant website for the task.\n"
            f"2. Complete the user's request: {task}\n"
            f"3. Verify the result matches the user's expectations.\n"
            f"4. Extract and report the final result."
        )


async def _generate_plan_with_council(council_member_ids: list[str], task: str) -> tuple[str, list[dict]]:
    """Query multiple LLMs for plans in parallel, then pick the best one by vote.

    Returns (best_plan_text, member_results) where member_results is a list of
    dicts with model, plan, and vote_count for dashboard display.
    """
    from browser_use.llm.messages import SystemMessage, UserMessage

    system_msg = SystemMessage(content=(
        "You are a browser automation planning expert. Given a user task, "
        "produce a short numbered plan of high-level actions a browser agent should follow.\n"
        "Rules:\n"
        "- Each step = ONE short sentence (max 10 words)\n"
        "- Use action verbs: Navigate, Click, Type, Extract, Scroll, Verify, Wait\n"
        "- 4-8 steps total. Be specific (include URLs, button names, field names when obvious)\n"
        "- No explanations, no code, no filler words\n"
        "Example:\n"
        "1. Navigate to google.com\n"
        "2. Type 'search query' in search box\n"
        "3. Click Google Search button\n"
        "4. Extract titles of top 3 results\n"
        "5. Report the extracted titles"
    ))
    user_msg = UserMessage(content=f"Task: {task}")

    # Phase 1: Each member generates a plan
    async def _get_plan(model_id):
        try:
            llm = create_llm(model_id)
            resp = await llm.ainvoke([system_msg, user_msg])
            return {"model": model_id, "plan": resp.completion.strip()}
        except Exception as e:
            logger.warning(f"Council planner {model_id} failed: {e}")
            return {"model": model_id, "plan": None}

    results = await asyncio.gather(*[_get_plan(mid) for mid in council_member_ids])
    valid = [r for r in results if r["plan"]]

    if not valid:
        logger.warning("Council planning: no models responded, using generic plan")
        return (
            f"1. Navigate to the relevant website for the task.\n"
            f"2. Complete the user's request: {task}\n"
            f"3. Verify the result matches the user's expectations.\n"
            f"4. Extract and report the final result."
        ), []

    if len(valid) == 1:
        logger.info(f"Council planning: only 1 response from {valid[0]['model']}")
        return valid[0]["plan"], [{"model": valid[0]["model"], "plan": valid[0]["plan"], "votes": 1}]

    # Phase 2: Each member votes for the best plan (not their own)
    plans_text = "\n\n".join(
        f"--- Plan {chr(65+i)} (by {r['model'].split('/')[-1]}) ---\n{r['plan']}"
        for i, r in enumerate(valid)
    )
    vote_msg = SystemMessage(content=(
        "You are judging browser automation plans. You will see multiple numbered plans "
        "for the same task. Vote for the BEST plan (not your own if you authored one).\n"
        "Criteria: specificity, completeness, conciseness, correct ordering.\n"
        "Reply with ONLY the letter of the best plan (A, B, C, etc.) and one sentence why.\n"
        "Format: VOTE: <letter> - <reason>"
    ))
    vote_user = UserMessage(content=f"Task: {task}\n\n{plans_text}")

    async def _cast_vote(model_id):
        try:
            llm = create_llm(model_id)
            resp = await llm.ainvoke([vote_msg, vote_user])
            text = resp.completion.strip().upper()
            # Extract the letter after VOTE:
            for line in text.split("\n"):
                if "VOTE:" in line.upper():
                    after = line.upper().split("VOTE:")[1].strip()
                    if after and after[0].isalpha():
                        return {"model": model_id, "vote": after[0]}
            # Fallback: first letter in response
            for ch in text:
                if ch.isalpha() and ord(ch) - 65 < len(valid):
                    return {"model": model_id, "vote": ch}
            return {"model": model_id, "vote": None}
        except Exception as e:
            logger.warning(f"Council voter {model_id} failed: {e}")
            return {"model": model_id, "vote": None}

    votes = await asyncio.gather(*[_cast_vote(mid) for mid in council_member_ids])

    # Tally votes
    vote_counts = {}
    for v in votes:
        if v["vote"]:
            idx = ord(v["vote"]) - 65
            if 0 <= idx < len(valid):
                vote_counts[idx] = vote_counts.get(idx, 0) + 1

    # Pick winner (most votes, tie-break: first plan with most steps)
    if vote_counts:
        winner_idx = max(vote_counts, key=lambda i: (vote_counts[i], len(valid[i]["plan"].split("\n"))))
    else:
        # No valid votes -- pick plan with most steps
        winner_idx = max(range(len(valid)), key=lambda i: len(valid[i]["plan"].split("\n")))

    winner = valid[winner_idx]
    member_results = [
        {
            "model": r["model"],
            "plan": r["plan"],
            "votes": vote_counts.get(i, 0),
            "winner": i == winner_idx,
        }
        for i, r in enumerate(valid)
    ]

    total_votes = sum(v for v in vote_counts.values())
    logger.info(
        f"Council planning: {len(valid)} plans, {total_votes} votes. "
        f"Winner: {winner['model']} ({vote_counts.get(winner_idx, 0)} votes)"
    )
    return winner["plan"], member_results


# ─── Persistent Browser Session ──────────────────────────────────────────────────

def _get_or_create_browser_session():
    """Return the persistent browser session, creating one if needed."""
    from browser_use import BrowserProfile, BrowserSession
    from browser_use.browser.profile import ViewportSize

    if state.browser_session is not None:
        # Health check: if the browser process died, clear and recreate
        try:
            if (state.browser_session._cdp_client_root is not None
                    and state.browser_session._cdp_client_root._ws is not None
                    and state.browser_session._cdp_client_root._ws.closed):
                logger.warning("Browser CDP connection lost, will create new session")
                state.browser_session = None
        except Exception:
            # If we can't check, assume it's dead
            logger.warning("Browser health check failed, will create new session")
            state.browser_session = None

    if state.browser_session is not None:
        return state.browser_session

    logger.info("Creating new persistent browser session (keep_alive=True)")
    browser_profile = BrowserProfile(
        headless=False,
        keep_alive=True,
        window_size=ViewportSize(width=SCREEN_WIDTH, height=SCREEN_HEIGHT),
        disable_security=True,
        demo_mode=False,
        args=[
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
        ],
    )
    state.browser_session = BrowserSession(browser_profile=browser_profile)
    return state.browser_session


# ─── Agent Runner ───────────────────────────────────────────────────────────────

async def run_agent(task: str, max_steps: int = 50, model_cfg: Optional[ModelConfig] = None):
    """Run the browser-use agent with the given task."""
    from browser_use import Agent, BrowserProfile, BrowserSession

    global _last_loop_council_step

    # Kill splash screen from previous run
    _kill_splash()

    state.is_running = True
    state.status = "running"
    state.current_task_text = task
    state.action_log = []
    state.step_count = 0
    state.result = None
    state.generated_plan = ""
    state.plan_progress = []
    _last_loop_council_step = 0
    global _plan_highest_done_idx
    _plan_highest_done_idx = -1

    # Broadcast status change (with cleared plan/log so frontend drops stale data)
    await broadcast_status()

    try:
        # ── Resolve model configuration ──
        if model_cfg is None:
            model_cfg = ModelConfig()

        env_model = os.environ.get("BROWSER_AGENT_MODEL", DEFAULT_MODEL_ID)
        primary_id = model_cfg.primary_model or env_model
        secondary_id = model_cfg.secondary_model or ""
        strategy = model_cfg.strategy

        state.active_strategy = strategy
        state.active_primary_model = primary_id
        state.active_secondary_model = secondary_id
        state.active_council_members = []

        logger.info(f"Strategy={strategy} | Primary={primary_id} | Secondary={secondary_id}")

        # ── Create LLM instances ──
        primary_llm = create_llm(primary_id)
        secondary_llm = create_llm(secondary_id) if secondary_id else None

        # Warn if a multi-model strategy was chosen but secondary is missing
        if strategy in ("fallback_chain", "planner_executor", "consensus") and not secondary_llm:
            logger.warning(
                f"Strategy '{strategy}' requires a secondary model but none was provided. "
                f"Falling back to single-model behavior."
            )
            # Broadcast warning to dashboard
            warn_msg = json.dumps({
                "type": "warning",
                "data": {"message": f"Strategy '{strategy}' requires a secondary model. Running as single model."},
            })
            for ws in state.ws_clients:
                try:
                    await ws.send_text(warn_msg)
                except Exception:
                    pass

        # ── Strategy-specific Agent kwargs ──
        extra_kwargs = {}

        # ── Grounding instructions (all strategies) ──
        grounding_prompt = (
            "\n\n## GROUNDING RULES (follow strictly)\n"
            "1. ALWAYS verify the current page state before acting. Read the element list carefully.\n"
            "2. Use the screenshot as your GROUND TRUTH. If the screenshot shows something different "
            "from what you expect, trust the screenshot.\n"
            "3. When filling forms, verify each field AFTER entering data by checking the element's "
            "value attribute or the screenshot.\n"
            "4. If an action fails or has no effect, do NOT repeat it. Try a different approach:\n"
            "   - Use a different element index\n"
            "   - Scroll to reveal hidden elements\n"
            "   - Use extract to understand the page structure\n"
            "   - Check for overlays, modals, or cookie banners blocking your target\n"
            "5. After clicking a button that should navigate or submit, check if the URL or page "
            "content actually changed.\n"
            "6. Pay attention to element attributes (id, name, type, placeholder, aria-label) to "
            "identify the correct element. Do not guess based on index numbers alone.\n"
            "7. If you see a GROUNDING feedback message in memory, follow its guidance immediately.\n"
            "8. CRITICAL ACTION FORMAT: Always use 'index' (NOT 'element_index') for element references.\n"
            "   Correct examples:\n"
            '   - Click: {"click": {"index": 5}}\n'
            '   - Type text: {"input_text": {"index": 5, "text": "hello"}}\n'
            '   - Scroll to: {"scroll_to_text": {"text": "Sign up"}}\n'
            "   WRONG: {\"click\": {\"element_index\": 5}} -- this WILL cause validation errors!\n"
            "9. If you need information from the user (like an OTP code, a password, or a confirmation), "
            "you can pause and the user can send you messages through the dashboard.\n"
        )
        extra_kwargs["extend_system_message"] = grounding_prompt

        if strategy == "fallback_chain" and secondary_llm:
            extra_kwargs["fallback_llm"] = secondary_llm

        elif strategy == "planner_executor":
            # Council-based planning: multiple models generate plans, then vote
            use_council = model_cfg and model_cfg.use_council_planning
            council_plan_members = []

            if use_council:
                council_plan_ids = model_cfg.council_members if model_cfg.council_members else [
                    m["id"] for m in AVAILABLE_MODELS
                ]
                # Ensure at least 2 members for a meaningful vote
                if len(council_plan_ids) < 2:
                    logger.warning("Council planning requires 2+ models, falling back to single planner")
                    use_council = False

            if use_council:
                plan_text, council_plan_members = await _generate_plan_with_council(council_plan_ids, task)
                planner_label = f"Council ({len(council_plan_members)} models)"
            elif secondary_llm:
                plan_text = await _generate_plan(secondary_llm, task)
                planner_label = secondary_id
            else:
                plan_text = await _generate_plan(primary_llm, task)
                planner_label = primary_id

            state.generated_plan = plan_text
            extra_kwargs["extend_system_message"] = extra_kwargs.get("extend_system_message", "") + (
                "\n\n## HIGH-LEVEL PLAN (from planning model)\n"
                "Follow this plan step-by-step. Adapt if pages differ from expectations.\n\n"
                + plan_text
            )
            # Enable planning so browser-use tracks plan step progress (done/current/pending)
            extra_kwargs["enable_planning"] = True
            if secondary_llm:
                extra_kwargs["page_extraction_llm"] = secondary_llm

            # Broadcast the generated plan to the dashboard
            plan_data = {
                "plan": plan_text,
                "planner_model": planner_label,
                "executor_model": primary_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if council_plan_members:
                plan_data["council_plans"] = council_plan_members
            plan_msg = json.dumps({"type": "plan", "data": plan_data})
            disconnected = set()
            for ws in state.ws_clients:
                try:
                    await ws.send_text(plan_msg)
                except Exception:
                    disconnected.add(ws)
            state.ws_clients -= disconnected

        elif strategy == "consensus" and secondary_llm:
            # Keep end-of-task judge for final verdict
            extra_kwargs["judge_llm"] = secondary_llm
            extra_kwargs["use_judge"] = True

        elif strategy == "council":
            # Enable planning so the agent can receive replans from council
            extra_kwargs["enable_planning"] = True

        # ── Use persistent browser session ──
        browser_session = _get_or_create_browser_session()

        # ── Inject previous task context when continuing a session ──
        if state.previous_task and state.previous_result:
            prev_ctx = (
                "\n\n## PREVIOUS TASK CONTEXT (browser session continues from previous task)\n"
                f"Previous task: {state.previous_task}\n"
                f"Previous result: {state.previous_result[:500]}\n"
                f"Previous model: {state.previous_model}\n"
                "The browser may still be on the page from the previous task. "
                "Use the current page state to your advantage if relevant.\n"
            )
            existing = extra_kwargs.get("extend_system_message", "")
            extra_kwargs["extend_system_message"] = existing + prev_ctx

        # ── Create agent ──
        agent = Agent(
            task=task,
            llm=primary_llm,
            browser_session=browser_session,
            register_new_step_callback=on_agent_step,
            use_vision=True,
            # Grounding: keep to 1 action per step so each action sees fresh DOM state.
            # Multi-action batching causes stale indices when earlier actions change the page.
            max_actions_per_step=1,
            generate_gif=False,
            # Grounding: include key HTML attributes in element representation so the LLM
            # can distinguish similar elements (e.g., multiple <input> fields on a form).
            include_attributes=["id", "name", "type", "placeholder", "aria-label", "role", "href", "value"],
            # Grounding: include recent browser events (clicks, navigations) so the LLM
            # sees what just happened on the page.
            include_recent_events=True,
            **extra_kwargs,
        )
        state.agent = agent

        # ── Seed agent's plan from our generated plan (planner_executor) ──
        if strategy == "planner_executor" and state.generated_plan:
            from browser_use.agent.views import PlanItem
            plan_lines = [
                line.strip()
                for line in state.generated_plan.split("\n")
                if line.strip() and re.match(r"^\d+[\.\)]", line.strip())
            ]
            plan_steps_text = [re.sub(r"^\d+[\.\)]\s*", "", ln) for ln in plan_lines]
            if plan_steps_text:
                agent.state.plan = [PlanItem(text=t) for t in plan_steps_text]
                agent.state.plan[0].status = "current"
                # Broadcast initial plan_progress so Plan tab shows progress immediately
                initial_progress = [{"text": item.text, "status": item.status} for item in agent.state.plan]
                state.plan_progress = initial_progress
                pp_msg = json.dumps({"type": "plan_progress", "data": {"items": initial_progress, "step": 0}})
                for ws in state.ws_clients:
                    try:
                        await ws.send_text(pp_msg)
                    except Exception:
                        pass

        # ── Build run() kwargs (per-step hooks for consensus / council) ──
        run_kwargs = {"max_steps": max_steps}
        if strategy == "consensus" and secondary_llm:
            async def _judge_hook(agent_instance):
                await _per_step_judge(agent_instance, secondary_llm, task)
            run_kwargs["on_step_end"] = _judge_hook

        elif strategy == "council":
            # Build council from explicit member list, or fall back to all non-primary models
            council_model_ids = model_cfg.council_members if model_cfg.council_members else [
                m["id"] for m in AVAILABLE_MODELS if m["id"] != primary_id
            ]
            # Always exclude primary from council (it's the one being advised)
            council_model_ids = [mid for mid in council_model_ids if mid != primary_id]
            council_llms = {}
            for mid in council_model_ids:
                try:
                    council_llms[mid] = create_llm(mid)
                except Exception as e:
                    logger.warning(f"Could not create council member {mid}: {e}")
            state.active_council_members = list(council_llms.keys())
            logger.info(f"Council members ({len(council_llms)}): {state.active_council_members}")

            async def _council_hook(agent_instance):
                await _council_step_hook(agent_instance, council_llms, task)
            run_kwargs["on_step_end"] = _council_hook

            # Start the stall monitor as a background task
            state.stall_monitor_task = asyncio.create_task(
                _stall_monitor(agent, council_llms, task)
            )

        logger.info(f"Starting agent with task: {task}")
        result = await agent.run(**run_kwargs)

        state.result = str(result.final_result()) if result else "No result"
        state.status = "completed"
        logger.info(f"Agent completed: {state.result[:200]}")

    except asyncio.CancelledError:
        state.status = "stopped"
        state.result = "Task was stopped by user"
        logger.info("Agent task cancelled")
    except Exception as e:
        state.status = "failed"
        state.result = f"Error: {str(e)}"
        logger.error(f"Agent failed: {e}", exc_info=True)
    finally:
        if state.stall_monitor_task and not state.stall_monitor_task.done():
            state.stall_monitor_task.cancel()
            state.stall_monitor_task = None
        # Save context for next task's session continuity
        state.previous_task = state.current_task_text
        state.previous_result = state.result or ""
        state.previous_model = state.active_primary_model
        state.is_running = False
        state.agent = None
        state.last_step_start_time = 0.0
        state.stall_council_fired_for_step = -1
        await broadcast_status()
        # Browser stays alive (keep_alive=True) so user can see where it ended up.
        # No splash screen -- the Xvfb display shows the browser's last page.


async def broadcast_status():
    """Send current status to all WebSocket clients."""
    msg = json.dumps({
        "type": "status",
        "data": {
            "status": state.status,
            "task": state.current_task_text,
            "steps": state.step_count,
            "result": state.result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": state.active_strategy,
            "primary_model": state.active_primary_model,
            "secondary_model": state.active_secondary_model,
            "council_members": state.active_council_members,
            "generated_plan": state.generated_plan,
            "plan_progress": state.plan_progress,
            "awaiting_user_input": state.awaiting_user_input,
        },
    })
    disconnected = set()
    for ws in state.ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            disconnected.add(ws)
    state.ws_clients -= disconnected


# ─── API Endpoints ──────────────────────────────────────────────────────────────

class StartAgentRequest(BaseModel):
    task: str
    max_steps: int = Field(default=50, ge=1, le=500)
    model_config_data: Optional[ModelConfig] = None
    new_session: bool = False  # if True, kill browser and start fresh

    @field_validator("task")
    @classmethod
    def task_must_be_nonempty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Task must not be empty")
        return v


async def _kill_browser_session():
    """Kill the persistent browser session and clear previous task context."""
    if state.browser_session:
        try:
            # First try graceful stop, then force kill
            state.browser_session.browser_profile.keep_alive = False
            try:
                await state.browser_session.stop()
            except Exception:
                pass
            await state.browser_session.kill()
        except Exception as e:
            logger.warning(f"Error killing browser session: {e}")
            # Fallback: try to kill underlying browser process directly
            try:
                if hasattr(state.browser_session, '_browser_pid') and state.browser_session._browser_pid:
                    os.kill(state.browser_session._browser_pid, signal.SIGKILL)
            except Exception:
                pass
        state.browser_session = None
        logger.info("Browser session killed for new session")
    # Clear previous task context since we're starting fresh
    state.previous_task = ""
    state.previous_result = ""
    state.previous_model = ""


@app.post("/api/agent/start")
async def start_agent(payload: StartAgentRequest):
    """Start a new agent task."""
    if state.is_running:
        return {"error": "Agent is already running. Stop it first.", "status": state.status}

    if payload.new_session:
        await _kill_browser_session()

    task = payload.task
    max_steps = payload.max_steps
    model_cfg = payload.model_config_data

    # Launch agent in background
    state.agent_task = asyncio.create_task(run_agent(task, max_steps, model_cfg))

    return {
        "status": "started", "task": task, "max_steps": max_steps,
        "strategy": model_cfg.strategy if model_cfg else "single",
    }


@app.post("/api/agent/stop")
async def stop_agent():
    """Stop the running agent."""
    if state.agent_task and not state.agent_task.done():
        state.agent_task.cancel()
        return {"status": "stopping"}
    return {"status": "not_running"}


@app.post("/api/agent/new-session")
async def new_browser_session():
    """Kill the persistent browser and start fresh on next task."""
    if state.is_running:
        return JSONResponse(
            status_code=409,
            content={"status": "error", "message": "Cannot reset session while a task is running"},
        )
    await _kill_browser_session()
    # Clear stale task data so dashboard shows clean state
    state.generated_plan = ""
    state.plan_progress = []
    state.action_log = []
    state.status = "idle"
    state.current_task_text = ""
    state.result = None
    state.step_count = 0
    _show_splash(status="idle")
    await broadcast_status()
    return {"status": "ok", "message": "Browser session cleared"}


@app.get("/api/agent/status")
async def get_status():
    """Get current agent status."""
    return {
        "status": state.status,
        "is_running": state.is_running,
        "task": state.current_task_text,
        "steps": state.step_count,
        "result": state.result,
        "action_log": state.action_log[-20:],
        "strategy": state.active_strategy,
        "primary_model": state.active_primary_model,
        "secondary_model": state.active_secondary_model,
        "council_members": state.active_council_members,
        "generated_plan": state.generated_plan,
        "plan_progress": state.plan_progress,
        "has_browser_session": state.browser_session is not None,
        "previous_task": state.previous_task,
    }


@app.get("/api/agent/logs")
async def get_logs():
    """Get full action log."""
    return {"logs": state.action_log}


@app.get("/api/models")
async def list_models():
    """Return available models and multi-model strategies."""
    return {
        "models": AVAILABLE_MODELS,
        "strategies": [
            {"id": sid, "description": desc}
            for sid, desc in STRATEGY_DESCRIPTIONS.items()
        ],
        "default_model": os.environ.get("BROWSER_AGENT_MODEL", DEFAULT_MODEL_ID),
    }


@app.get("/api/novnc-url")
async def get_novnc_url():
    """Return the noVNC URL for the virtual display. Reads NOVNC_PUBLIC_URL env var or constructs from known patterns."""
    novnc_url = os.environ.get("NOVNC_PUBLIC_URL", "")
    if novnc_url:
        return {"url": novnc_url}

    # Try to read from the export-port output if available
    # Default: noVNC runs on port 6080
    return {"url": "", "port": NOVNC_PORT, "message": "Set NOVNC_PUBLIC_URL env var with the exported noVNC URL"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket for real-time updates (screenshots + agent steps)."""
    await ws.accept()
    state.ws_clients.add(ws)
    logger.info(f"WebSocket client connected ({len(state.ws_clients)} total)")

    # Send current status on connect
    await ws.send_text(json.dumps({
        "type": "status",
        "data": {
            "status": state.status,
            "task": state.current_task_text,
            "steps": state.step_count,
            "result": state.result,
            "strategy": state.active_strategy,
            "primary_model": state.active_primary_model,
            "secondary_model": state.active_secondary_model,
            "council_members": state.active_council_members,
            "generated_plan": state.generated_plan,
            "plan_progress": state.plan_progress,
            "awaiting_user_input": state.awaiting_user_input,
        },
    }))

    # Send existing logs
    for entry in state.action_log:
        await ws.send_text(json.dumps({"type": "step", "data": entry}))

    try:
        while True:
            # Keep connection alive, handle client messages
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
            elif msg.get("type") == "start_task":
                if not state.is_running:
                    task = msg.get("task", "")
                    max_steps = msg.get("max_steps", 50)
                    mc_raw = msg.get("model_config", {})
                    model_cfg = ModelConfig(**mc_raw) if mc_raw else None
                    if msg.get("new_session"):
                        await _kill_browser_session()
                    if task:
                        state.agent_task = asyncio.create_task(run_agent(task, max_steps, model_cfg))
                        await ws.send_text(json.dumps({"type": "ack", "message": "Task started"}))
                else:
                    await ws.send_text(json.dumps({"type": "error", "message": "Agent already running"}))
            elif msg.get("type") == "stop_task":
                if state.agent_task and not state.agent_task.done():
                    state.agent_task.cancel()
                    await ws.send_text(json.dumps({"type": "ack", "message": "Stopping agent"}))
            elif msg.get("type") == "user_message":
                # User sends a message to the running agent -- agent will see it on next step
                user_msg = msg.get("message", "").strip()
                if user_msg and state.is_running:
                    if state.awaiting_user_input and state.user_answer_event:
                        # Agent is already paused waiting -- deliver as answer
                        state.user_answer = user_msg
                        state.user_answer_event.set()
                        await ws.send_text(json.dumps({"type": "ack", "message": "Answer delivered to agent"}))
                    else:
                        # Agent is running -- queue it so on_agent_step picks it up after current step
                        state.pending_user_question = user_msg
                        await ws.send_text(json.dumps({"type": "ack", "message": "Message queued for agent"}))
                elif not state.is_running:
                    await ws.send_text(json.dumps({"type": "error", "message": "No agent is running"}))
            elif msg.get("type") == "answer_question":
                # User answers a question the agent asked (or the pause prompt)
                answer = msg.get("answer", "").strip()
                if answer and state.awaiting_user_input and state.user_answer_event:
                    state.user_answer = answer
                    state.user_answer_event.set()
                    await ws.send_text(json.dumps({"type": "ack", "message": "Answer delivered to agent"}))
                else:
                    await ws.send_text(json.dumps({"type": "error", "message": "No pending question to answer"}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        state.ws_clients.discard(ws)
        logger.info(f"WebSocket client disconnected ({len(state.ws_clients)} total)")


# ─── Lifecycle ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Initialize virtual display, VNC, noVNC, and screenshot loop."""
    start_virtual_display()
    start_vnc_server()
    start_novnc()
    state.screenshot_task = asyncio.create_task(screenshot_broadcast_loop())
    logger.info("All services started")
    # Show idle splash so VNC doesn't start with a black screen
    _show_splash(status="idle", steps=0, actions=0)


@app.on_event("shutdown")
async def shutdown():
    """Clean up processes."""
    _kill_splash()
    if state.screenshot_task:
        state.screenshot_task.cancel()
    if state.stall_monitor_task and not state.stall_monitor_task.done():
        state.stall_monitor_task.cancel()
    if state.agent_task and not state.agent_task.done():
        state.agent_task.cancel()

    for proc_name, proc in [("Xvfb", state.xvfb_proc), ("x11vnc", state.vnc_proc), ("noVNC", state.novnc_proc)]:
        if proc and proc.poll() is None:
            proc.terminate()
            logger.info(f"Terminated {proc_name}")


# ─── Serve the dashboard HTML ──────────────────────────────────────────────────

@app.get("/")
async def serve_dashboard():
    """Serve the main dashboard page."""
    dashboard_path = Path(__file__).parent / "dashboard.html"
    return HTMLResponse(content=dashboard_path.read_text(), status_code=200)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, log_level="info")
