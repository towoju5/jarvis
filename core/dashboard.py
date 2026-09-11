"""Minimal local status dashboard: http://127.0.0.1:<DASHBOARD_PORT>/

Deliberately the smallest useful version -- plain aiohttp, one HTML page,
the browser polls /api/status every 2s. No auth (binds to 127.0.0.1 only),
no websockets/SSE, no build step. Shows: current agent state, task history
from StateManager, and a live tail of recent log records via an in-memory
ring buffer attached to the root logger.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Callable

from aiohttp import web

from config.settings import Settings
from core.state_manager import StateManager

logger = logging.getLogger(__name__)


class LogBuffer(logging.Handler):
    """Keeps the last N formatted log lines in memory for the dashboard to read."""

    def __init__(self, maxlen: int = 300) -> None:
        super().__init__()
        self._lines: deque[str] = deque(maxlen=maxlen)
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._lines.append(self.format(record))
        except Exception:
            pass

    def get_lines(self) -> list[str]:
        return list(self._lines)


@dataclass
class AgentStatus:
    """Mutated in place by main.py as the agent moves through its loop."""
    state: str = "idle"  # idle | listening | generating | pending_approval | publishing
    detail: str = ""


def _status_json(settings: Settings, status: AgentStatus, state_manager: StateManager, log_buffer: LogBuffer) -> dict:
    tasks = [
        {"id": t.id, "description": t.description, "state": t.state.name, "error": t.error}
        for t in sorted(state_manager.tasks.values(), key=lambda t: t.id, reverse=True)
    ]
    return {
        "agent_name": settings.agent_name,
        "execution_mode": settings.execution_mode,
        "trigger_hotkey": settings.trigger_hotkey if settings.enable_hotkey_trigger else "disabled (wake word only)",
        "status": status.state,
        "status_detail": status.detail,
        "tasks": tasks,
        "logs": log_buffer.get_lines()[-200:],
    }


_PAGE_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Jarvis Dashboard</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0; padding: 24px;
         background: Canvas; color: CanvasText; }
  h1 { font-size: 1.25rem; margin: 0 0 4px; }
  .sub { color: GrayText; font-size: 0.85rem; margin-bottom: 20px; }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.8rem;
           font-weight: 600; text-transform: uppercase; letter-spacing: 0.02em; }
  .badge.idle { background: #4443; }
  .badge.warming_up { background: #64748b33; color: #64748b; }
  .badge.listening { background: #3b82f633; color: #3b82f6; }
  .badge.generating, .badge.pending_approval { background: #f59e0b33; color: #f59e0b; }
  .badge.publishing { background: #8b5cf633; color: #8b5cf6; }
  section { margin-bottom: 28px; }
  h2 { font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.04em; color: GrayText;
       margin-bottom: 8px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #8883; }
  th { color: GrayText; font-weight: 500; }
  .state-tag { font-size: 0.78rem; padding: 2px 8px; border-radius: 4px; }
  .state-DONE { background: #22c55e33; color: #22c55e; }
  .state-FAILED, .state-REJECTED { background: #ef444433; color: #ef4444; }
  .state-GENERATING, .state-PENDING_APPROVAL, .state-PUBLISHING { background: #f59e0b33; color: #f59e0b; }
  #logs { background: #0003; border-radius: 6px; padding: 10px 12px; height: 260px;
          overflow-y: auto; font-family: ui-monospace, monospace; font-size: 0.78rem;
          white-space: pre-wrap; }
  .empty { color: GrayText; font-style: italic; font-size: 0.85rem; }
</style>
</head>
<body>
  <h1 id="agent-name">Jarvis</h1>
  <div class="sub">
    <span class="badge idle" id="status-badge">idle</span>
    <span id="status-detail"></span>
    &middot; mode: <span id="exec-mode">-</span>
    &middot; hotkey: <span id="hotkey">-</span>
  </div>

  <section>
    <h2>Tasks</h2>
    <table id="tasks-table">
      <thead><tr><th>ID</th><th>Description</th><th>State</th><th>Error</th></tr></thead>
      <tbody id="tasks-body"></tbody>
    </table>
    <div class="empty" id="tasks-empty" style="display:none">No tasks yet.</div>
  </section>

  <section>
    <h2>Live logs</h2>
    <div id="logs"></div>
  </section>

<script>
async function refresh() {
  let res;
  try { res = await fetch('/api/status'); } catch (e) { return; }
  if (!res.ok) return;
  const data = await res.json();

  document.getElementById('agent-name').textContent = data.agent_name;
  document.getElementById('exec-mode').textContent = data.execution_mode;
  document.getElementById('hotkey').textContent = data.trigger_hotkey;

  const badge = document.getElementById('status-badge');
  badge.textContent = data.status;
  badge.className = 'badge ' + data.status;
  document.getElementById('status-detail').textContent = data.status_detail || '';

  const body = document.getElementById('tasks-body');
  body.innerHTML = '';
  document.getElementById('tasks-empty').style.display = data.tasks.length ? 'none' : 'block';
  for (const t of data.tasks) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${t.id}</td><td>${escapeHtml(t.description)}</td>` +
      `<td><span class="state-tag state-${t.state}">${t.state}</span></td>` +
      `<td>${escapeHtml(t.error || '')}</td>`;
    body.appendChild(tr);
  }

  const logsEl = document.getElementById('logs');
  const wasAtBottom = logsEl.scrollTop + logsEl.clientHeight >= logsEl.scrollHeight - 10;
  logsEl.textContent = data.logs.join('\\n');
  if (wasAtBottom) logsEl.scrollTop = logsEl.scrollHeight;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""


def create_app(
    get_settings_fn: Callable[[], Settings],
    status: AgentStatus,
    state_manager: StateManager,
    log_buffer: LogBuffer,
) -> web.Application:
    async def index(_request: web.Request) -> web.Response:
        return web.Response(text=_PAGE_HTML, content_type="text/html")

    async def api_status(_request: web.Request) -> web.Response:
        return web.json_response(_status_json(get_settings_fn(), status, state_manager, log_buffer))

    app = web.Application()
    app.add_routes([web.get("/", index), web.get("/api/status", api_status)])
    return app


async def start_dashboard(app: web.Application, host: str, port: int) -> web.AppRunner:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info("dashboard listening on http://%s:%d/", host, port)
    return runner
