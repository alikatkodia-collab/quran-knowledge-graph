"""
Autoresearch Dashboard — Monitor and control the optimization loop.

Provides a real-time web UI to:
  - View trial results as they come in
  - See score chart, best config, parameter distributions
  - Start / Pause / Resume / Stop the autoresearch process

Usage:
    py autoresearch_dashboard.py                     # default port 8082
    py autoresearch_dashboard.py --port 8083
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

PROJECT_ROOT = Path(__file__).parent
LOG_PATH = PROJECT_ROOT / "autoresearch_local_log.jsonl"
BEST_CONFIG_PATH = PROJECT_ROOT / "best_config_local.yaml"
OUTPUT_PATH = PROJECT_ROOT / "autoresearch_local_output.txt"

app = FastAPI()

# ── Process management ──────────────────────────────────────────────────────

_process = None
_process_lock = threading.Lock()
_status = {"state": "stopped", "pid": None, "trials_target": 100, "model": "qwen2.5:14b-instruct-q6_K"}


def _find_running_process():
    """Check if an autoresearch process is already running."""
    global _process, _status
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'cmdline']):
            cmdline = proc.info.get('cmdline') or []
            if any('autoresearch_local.py' in str(c) for c in cmdline):
                _status["state"] = "running"
                _status["pid"] = proc.info['pid']
                return
    except ImportError:
        # psutil not available, check our own subprocess
        pass

    # Also check if log file is being written to (recent modification)
    if LOG_PATH.exists():
        mtime = LOG_PATH.stat().st_mtime
        if time.time() - mtime < 120:  # modified in last 2 minutes
            _status["state"] = "running"
            _status["pid"] = None  # unknown PID


_find_running_process()


# ── API endpoints ───────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    """Return current process status and trial summary."""
    # Dynamic detection: check if autoresearch process is running
    _detect_running_process()
    trials = _load_trials()
    best_score = max((t.get("score", 0) for t in trials), default=0)
    return {
        **_status,
        "trials_completed": len(trials),
        "best_score": round(best_score, 4),
    }

def _detect_running_process():
    """Dynamically detect if autoresearch_local.py is running."""
    global _status
    # If we started the process ourselves, check if it's still alive
    if _process and _process.poll() is None:
        _status["state"] = "running"
        _status["pid"] = _process.pid
        return
    if _process and _process.poll() is not None:
        _status["state"] = "stopped"
        _status["pid"] = None
        return
    # Check for externally started process
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'cmdline']):
            cmdline = proc.info.get('cmdline') or []
            if any('autoresearch_local.py' in str(c) for c in cmdline):
                _status["state"] = "running"
                _status["pid"] = proc.info['pid']
                return
    except ImportError:
        pass
    # Fallback: check if log file was recently modified
    if LOG_PATH.exists():
        mtime = LOG_PATH.stat().st_mtime
        if time.time() - mtime < 120:
            _status["state"] = "running"
            return
    # Also check output file
    if OUTPUT_PATH.exists():
        mtime = OUTPUT_PATH.stat().st_mtime
        if time.time() - mtime < 120:
            _status["state"] = "running"
            return
    _status["state"] = "stopped"
    _status["pid"] = None


@app.get("/api/trials")
def get_trials():
    """Return all trial data."""
    return _load_trials()


@app.get("/api/best_config")
def get_best_config():
    """Return the current best config."""
    if BEST_CONFIG_PATH.exists():
        import yaml
        with open(BEST_CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {"error": "No best config found yet"}


@app.post("/api/start")
def start_process(trials: int = 100, model: str = "qwen2.5:14b-instruct-q6_K"):
    """Start the autoresearch process."""
    global _process, _status
    with _process_lock:
        if _status["state"] == "running":
            return {"error": "Already running"}

        cmd = [
            sys.executable, "-u", "autoresearch_local.py",
            "--trials", str(trials),
            "--model", model,
        ]
        _process = subprocess.Popen(
            cmd,
            stdout=open(OUTPUT_PATH, "w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )
        _status["state"] = "running"
        _status["pid"] = _process.pid
        _status["trials_target"] = trials
        _status["model"] = model

    return {"status": "started", "pid": _process.pid}


@app.post("/api/stop")
def stop_process():
    """Stop the autoresearch process."""
    global _process, _status
    with _process_lock:
        if _status["state"] != "running":
            return {"error": "Not running"}

        pid = _status.get("pid")
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass

        if _process:
            try:
                _process.terminate()
                _process.wait(timeout=10)
            except Exception:
                try:
                    _process.kill()
                except Exception:
                    pass
            _process = None

        _status["state"] = "stopped"
        _status["pid"] = None

    return {"status": "stopped"}


@app.get("/api/stream")
def stream_trials():
    """SSE stream of new trial results."""
    def event_stream():
        seen = 0
        while True:
            trials = _load_trials()
            if len(trials) > seen:
                for t in trials[seen:]:
                    yield f"data: {json.dumps(t)}\n\n"
                seen = len(trials)
            time.sleep(5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/output")
def get_output():
    """Return recent console output."""
    if OUTPUT_PATH.exists():
        text = OUTPUT_PATH.read_text(encoding="utf-8", errors="replace")
        # Return last 5000 chars
        return {"output": text[-5000:]}
    return {"output": "No output yet"}


# ── helpers ─────────────────────────────────────────────────────────────────

def _load_trials():
    """Load all trials from the JSONL log."""
    trials = []
    if LOG_PATH.exists():
        with open(LOG_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        trials.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return trials


# ── HTML dashboard ──────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Autoresearch Dashboard</title>
<style>
  :root {
    --bg-primary: #060a14;
    --bg-secondary: #0a0f1a;
    --bg-card: #0d1321;
    --bg-hover: #111827;
    --border: #1e293b;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent: #10b981;
    --accent-dim: #065f46;
    --warning: #f59e0b;
    --danger: #ef4444;
    --info: #6366f1;
  }

  *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace;
    background: var(--bg-primary);
    color: var(--text-primary);
    min-height: 100vh;
    overflow-x: hidden;
    max-width: 100vw;
  }

  .header {
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
  }

  .header h1 {
    font-size: 18px;
    font-weight: 600;
    color: var(--accent);
  }

  .header .status-badge {
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
  }

  .status-running { background: var(--accent-dim); color: var(--accent); }
  .status-stopped { background: #1e1e1e; color: var(--text-muted); }

  .controls {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .controls button {
    padding: 6px 16px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--bg-card);
    color: var(--text-primary);
    font-family: inherit;
    font-size: 13px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .controls button:hover { background: var(--bg-hover); border-color: var(--accent); }
  .controls button.primary { background: var(--accent-dim); border-color: var(--accent); color: var(--accent); }
  .controls button.danger { border-color: var(--danger); color: var(--danger); }
  .controls button:disabled { opacity: 0.4; cursor: not-allowed; }

  .controls input, .controls select {
    padding: 6px 10px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--bg-card);
    color: var(--text-primary);
    font-family: inherit;
    font-size: 13px;
    width: 80px;
  }

  .grid {
    display: grid;
    grid-template-columns: 3fr 2fr;
    gap: 16px;
    padding: 16px 24px;
    max-width: 100%;
  }

  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    min-width: 0;
    overflow: auto;
  }

  .card h2 {
    font-size: 13px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 12px;
  }

  .stats-row {
    display: flex;
    gap: 16px;
    padding: 16px 24px;
    flex-wrap: wrap;
  }

  .stat {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px;
    flex: 1;
    text-align: center;
  }

  .stat .value {
    font-size: 28px;
    font-weight: 700;
    color: var(--accent);
  }

  .stat .label {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-top: 4px;
  }

  .chart-container {
    width: 100%;
    max-width: 100%;
    height: 280px;
    position: relative;
  }

  canvas { max-width: 100% !important; height: 100% !important; }

  .trial-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }

  .trial-table th {
    text-align: left;
    padding: 8px 10px;
    border-bottom: 1px solid var(--border);
    color: var(--text-muted);
    font-weight: 600;
    position: sticky;
    top: 0;
    background: var(--bg-card);
  }

  .trial-table td {
    padding: 6px 10px;
    border-bottom: 1px solid #0f172a;
    color: var(--text-secondary);
  }

  .trial-table tr:hover td { background: var(--bg-hover); }
  .trial-table .best td { color: var(--accent); font-weight: 600; }

  .table-wrap {
    max-height: 350px;
    overflow-y: auto;
    overflow-x: auto;
  }

  .param-grid {
    display: flex;
    flex-direction: column;
    gap: 6px;
    font-size: 13px;
  }

  .param-item {
    padding: 5px 10px;
    background: var(--bg-secondary);
    border-radius: 4px;
    line-height: 1.5;
  }

  .param-item .key { color: var(--text-muted); }
  .param-item .val { color: var(--accent); font-weight: 700; }

  .console {
    background: #000;
    border-radius: 6px;
    padding: 12px;
    font-size: 11px;
    color: #9ca3af;
    max-height: 200px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
  }

  .full-width { grid-column: 1 / -1; }

  .metrics-bar {
    display: flex;
    gap: 4px;
    align-items: center;
    margin-top: 4px;
  }

  .metrics-bar .bar {
    height: 6px;
    border-radius: 3px;
    flex: 1;
  }

  @media (max-width: 900px) {
    .grid { grid-template-columns: 1fr; }
    .stats-row { flex-wrap: wrap; }
  }
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</head>
<body>

<div class="header">
  <h1>AUTORESEARCH DASHBOARD</h1>
  <div class="controls">
    <label style="font-size:12px;color:var(--text-muted)">Trials:</label>
    <input type="number" id="trialsInput" value="100" min="1" max="500">
    <button id="startBtn" class="primary" onclick="startProcess()">Start</button>
    <button id="stopBtn" class="danger" onclick="stopProcess()" disabled>Stop</button>
    <span id="statusBadge" class="status-badge status-stopped">STOPPED</span>
  </div>
</div>

<div class="stats-row">
  <div class="stat">
    <div class="value" id="trialsCompleted">0</div>
    <div class="label">Trials Completed</div>
  </div>
  <div class="stat">
    <div class="value" id="bestScore">--</div>
    <div class="label">Best QIS Score</div>
  </div>
  <div class="stat">
    <div class="value" id="avgTime">--</div>
    <div class="label">Avg Trial Time</div>
  </div>
  <div class="stat">
    <div class="value" id="eta">--</div>
    <div class="label">ETA Remaining</div>
  </div>
</div>

<div class="grid">
  <div class="card">
    <h2>Score Progression</h2>
    <div class="chart-container">
      <canvas id="scoreChart"></canvas>
    </div>
  </div>

  <div class="card">
    <h2>Best Configuration</h2>
    <div id="bestConfig" class="param-grid">
      <div class="param-item"><span class="key">No data yet</span><span class="val">--</span></div>
    </div>
  </div>

  <div class="card">
    <h2>Metric Breakdown (Latest Best)</h2>
    <div class="chart-container" style="height:220px">
      <canvas id="metricsChart"></canvas>
    </div>
  </div>

  <div class="card">
    <h2>Parameter Impact</h2>
    <div class="chart-container" style="height:220px">
      <canvas id="paramChart"></canvas>
    </div>
  </div>

  <div class="card full-width">
    <h2>Trial History</h2>
    <div class="table-wrap">
      <table class="trial-table">
        <thead>
          <tr>
            <th>#</th>
            <th>QIS Score</th>
            <th>Recall</th>
            <th>Precision</th>
            <th>Grounding</th>
            <th>Relevance</th>
            <th>Time</th>
            <th>top_k</th>
            <th>seed</th>
            <th>hop1</th>
            <th>min_score</th>
          </tr>
        </thead>
        <tbody id="trialRows"></tbody>
      </table>
    </div>
  </div>

  <div class="card full-width">
    <h2>Console Output</h2>
    <div class="console" id="consoleOutput">Waiting for output...</div>
  </div>
</div>

<script>
const API = '';
let trials = [];
let bestTrialIdx = -1;
let scoreChart, metricsChart, paramChart;

// ── Charts setup ──────────────────────────────────────────────────────────

function initCharts() {
  const chartDefaults = {
    color: '#94a3b8',
    borderColor: '#1e293b',
  };
  Chart.defaults.color = '#94a3b8';
  Chart.defaults.borderColor = '#1e293b';

  scoreChart = new Chart(document.getElementById('scoreChart'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Trial Score',
          data: [],
          borderColor: '#6366f1',
          backgroundColor: 'rgba(99,102,241,0.1)',
          fill: true,
          tension: 0.3,
          pointRadius: 3,
        },
        {
          label: 'Best So Far',
          data: [],
          borderColor: '#10b981',
          borderDash: [5, 5],
          pointRadius: 0,
          tension: 0.1,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { min: 0, max: 1, grid: { color: '#1e293b' } },
        x: { grid: { display: false } }
      },
      plugins: { legend: { labels: { boxWidth: 12 } } }
    }
  });

  metricsChart = new Chart(document.getElementById('metricsChart'), {
    type: 'bar',
    data: {
      labels: ['Recall', 'Precision', 'Grounding', 'Relevance'],
      datasets: [{
        data: [0, 0, 0, 0],
        backgroundColor: ['#10b981', '#6366f1', '#f59e0b', '#ec4899'],
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { min: 0, max: 1, grid: { color: '#1e293b' } },
        x: { grid: { display: false } }
      },
      plugins: { legend: { display: false } }
    }
  });

  paramChart = new Chart(document.getElementById('paramChart'), {
    type: 'scatter',
    data: { datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { title: { display: true, text: 'QIS Score' }, min: 0, max: 1, grid: { color: '#1e293b' } },
        x: { title: { display: true, text: 'top_k' }, grid: { color: '#1e293b' } }
      },
      plugins: { legend: { display: false } }
    }
  });
}

// ── Data fetching ─────────────────────────────────────────────────────────

async function fetchTrials() {
  try {
    const res = await fetch(API + '/api/trials');
    trials = await res.json();
    updateUI();
  } catch (e) {}
}

async function fetchStatus() {
  try {
    const res = await fetch(API + '/api/status');
    const s = await res.json();
    const badge = document.getElementById('statusBadge');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');

    if (s.state === 'running') {
      badge.textContent = 'RUNNING';
      badge.className = 'status-badge status-running';
      startBtn.disabled = true;
      stopBtn.disabled = false;
    } else {
      badge.textContent = 'STOPPED';
      badge.className = 'status-badge status-stopped';
      startBtn.disabled = false;
      stopBtn.disabled = true;
    }
  } catch (e) {}
}

async function fetchConsole() {
  try {
    const res = await fetch(API + '/api/output');
    const d = await res.json();
    const el = document.getElementById('consoleOutput');
    el.textContent = d.output || 'No output yet';
    el.scrollTop = el.scrollHeight;
  } catch (e) {}
}

// ── UI updates ────────────────────────────────────────────────────────────

function updateUI() {
  if (!trials.length) return;

  // Stats
  let best = 0, bestIdx = 0;
  let totalTime = 0;
  trials.forEach((t, i) => {
    if ((t.score || 0) > best) { best = t.score; bestIdx = i; }
    totalTime += t.elapsed_s || 0;
  });
  bestTrialIdx = bestIdx;

  document.getElementById('trialsCompleted').textContent = trials.length;
  document.getElementById('bestScore').textContent = best.toFixed(4);

  const avgT = totalTime / trials.length;
  document.getElementById('avgTime').textContent = formatTime(avgT);

  const trialsTarget = parseInt(document.getElementById('trialsInput').value) || 100;
  const remaining = Math.max(0, trialsTarget - trials.length);
  document.getElementById('eta').textContent = remaining > 0 ? formatTime(remaining * avgT) : 'Done';

  // Score chart
  scoreChart.data.labels = trials.map((_, i) => i);
  scoreChart.data.datasets[0].data = trials.map(t => t.score || 0);
  let runBest = 0;
  scoreChart.data.datasets[1].data = trials.map(t => {
    runBest = Math.max(runBest, t.score || 0);
    return runBest;
  });
  scoreChart.update('none');

  // Metrics chart (best trial)
  const bestTrial = trials[bestIdx];
  if (bestTrial && bestTrial.aggregate) {
    const a = bestTrial.aggregate;
    metricsChart.data.datasets[0].data = [
      a.citation_recall || 0,
      a.citation_precision || 0,
      a.grounding_rate || 0,
      a.answer_relevance || 0,
    ];
    metricsChart.update('none');
  }

  // Param scatter (top_k vs score)
  paramChart.data.datasets = [{
    data: trials.map(t => ({
      x: t.params?.top_k || 0,
      y: t.score || 0,
    })),
    backgroundColor: trials.map((t, i) => i === bestIdx ? '#10b981' : '#6366f1'),
    pointRadius: trials.map((t, i) => i === bestIdx ? 8 : 4),
  }];
  paramChart.update('none');

  // Best config
  if (bestTrial && bestTrial.params) {
    const el = document.getElementById('bestConfig');
    el.innerHTML = Object.entries(bestTrial.params)
      .map(([k, v]) => `<div class="param-item"><span class="key">${k}:</span> <span class="val">${v}</span></div>`)
      .join('');
  }

  // Trial table
  const tbody = document.getElementById('trialRows');
  tbody.innerHTML = trials.slice().reverse().map((t, ri) => {
    const idx = trials.length - 1 - ri;
    const isBest = idx === bestIdx;
    const a = t.aggregate || {};
    return `<tr class="${isBest ? 'best' : ''}">
      <td>${t.trial}</td>
      <td>${(t.score || 0).toFixed(4)}</td>
      <td>${(a.citation_recall || 0).toFixed(3)}</td>
      <td>${(a.citation_precision || 0).toFixed(3)}</td>
      <td>${(a.grounding_rate || 0).toFixed(3)}</td>
      <td>${(a.answer_relevance || 0).toFixed(3)}</td>
      <td>${formatTime(t.elapsed_s || 0)}</td>
      <td>${t.params?.top_k ?? '--'}</td>
      <td>${t.params?.seed_limit ?? '--'}</td>
      <td>${t.params?.hop1_limit ?? '--'}</td>
      <td>${t.params?.min_score ?? '--'}</td>
    </tr>`;
  }).join('');
}

function formatTime(s) {
  if (s < 60) return Math.round(s) + 's';
  if (s < 3600) return Math.round(s / 60) + 'm';
  const h = Math.floor(s / 3600);
  const m = Math.round((s % 3600) / 60);
  return h + 'h ' + m + 'm';
}

// ── Controls ──────────────────────────────────────────────────────────────

async function startProcess() {
  const trials = document.getElementById('trialsInput').value || 100;
  try {
    const res = await fetch(API + '/api/start?trials=' + trials, { method: 'POST' });
    const d = await res.json();
    if (d.error) alert(d.error);
    else fetchStatus();
  } catch (e) { alert('Failed to start: ' + e); }
}

async function stopProcess() {
  if (!confirm('Stop the autoresearch process? Best config will be preserved.')) return;
  try {
    const res = await fetch(API + '/api/stop', { method: 'POST' });
    const d = await res.json();
    fetchStatus();
  } catch (e) { alert('Failed to stop: ' + e); }
}

// ── Init ──────────────────────────────────────────────────────────────────

initCharts();
fetchTrials();
fetchStatus();
fetchConsole();

// Poll every 10 seconds
setInterval(() => {
  fetchTrials();
  fetchStatus();
  fetchConsole();
}, 10000);
</script>
</body>
</html>
"""

# ── main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoresearch Dashboard")
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()

    print(f"Autoresearch Dashboard: http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
