#!/usr/bin/env python3
"""
Master Continuous Runner — Orchestrates all infinite loops.

Manages three concurrent processes:
  1. Infinite Deduction Loop (generates raw deductions)
  2. Infinite Analysis Loop (categorizes, scores, builds meta-graph)
  3. Meta-Graph Optimization Loop (optimizes analysis parameters)

Usage:
    python autoresearch/continuous_runner.py [--hours 0]

    --hours 0 means run forever (until killed)
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTORESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(AUTORESEARCH_DIR, "logs")


class ProcessManager:
    """Manages multiple background processes with monitoring."""

    def __init__(self):
        self.processes = {}
        self.start_time = time.time()

    def start(self, name, cmd, log_file=None):
        """Start a background process."""
        os.makedirs(LOG_DIR, exist_ok=True)
        if log_file is None:
            log_file = os.path.join(LOG_DIR, f"{name}.log")

        f = open(log_file, "a")
        proc = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            stdout=f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
        )
        self.processes[name] = {
            "proc": proc,
            "pid": proc.pid,
            "log": log_file,
            "file": f,
            "started": datetime.now().isoformat(),
            "cmd": " ".join(cmd),
        }
        print(f"  Started {name} (PID {proc.pid})")
        return proc.pid

    def is_alive(self, name):
        info = self.processes.get(name)
        if not info:
            return False
        return info["proc"].poll() is None

    def restart_if_dead(self, name):
        info = self.processes.get(name)
        if not info:
            return False
        if info["proc"].poll() is not None:
            print(f"  {name} died (exit code {info['proc'].returncode}), restarting...")
            info["file"].close()
            cmd = info["cmd"].split()
            self.start(name, cmd, info["log"])
            return True
        return False

    def stop_all(self):
        for name, info in self.processes.items():
            proc = info["proc"]
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                print(f"  Stopped {name} (PID {info['pid']})")
            info["file"].close()

    def status(self):
        status = {}
        for name, info in self.processes.items():
            alive = info["proc"].poll() is None
            status[name] = {
                "pid": info["pid"],
                "alive": alive,
                "started": info["started"],
                "log": info["log"],
            }
        return status

    def print_status(self):
        elapsed = time.time() - self.start_time
        hours = elapsed / 3600
        print(f"\n{'='*60}")
        print(f"  PROCESS STATUS ({hours:.2f}h elapsed)")
        print(f"{'='*60}")
        for name, info in self.processes.items():
            alive = info["proc"].poll() is None
            icon = "RUNNING" if alive else "STOPPED"
            print(f"  {name:30s} [{icon}] PID {info['pid']}")

        # Check output stats
        files_to_check = {
            "all_deductions.jsonl": "Total deductions",
            "meta_knowledge_graph.json": "Meta-graph",
            "synthesized_insights.json": "Insights",
            "meta_insights.json": "Meta-insights",
        }
        print(f"\n  Output files:")
        for fname, label in files_to_check.items():
            fpath = os.path.join(AUTORESEARCH_DIR, fname)
            if os.path.exists(fpath):
                size = os.path.getsize(fpath)
                if fname.endswith(".jsonl"):
                    with open(fpath) as f:
                        count = sum(1 for _ in f)
                    print(f"    {label:25s} {count:,} entries ({size/1024:.0f} KB)")
                else:
                    print(f"    {label:25s} {size/1024:.0f} KB")
            else:
                print(f"    {label:25s} not yet created")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Master continuous runner")
    parser.add_argument("--hours", type=float, default=0,
                        help="Hours to run (0 = forever)")
    parser.add_argument("--no-deductions", action="store_true",
                        help="Skip deduction generation (use existing)")
    parser.add_argument("--no-analysis", action="store_true",
                        help="Skip analysis loop")
    parser.add_argument("--no-metagraph", action="store_true",
                        help="Skip meta-graph optimization")
    args = parser.parse_args()

    max_seconds = args.hours * 3600 if args.hours > 0 else float("inf")

    print("=" * 60)
    print("  QURAN KNOWLEDGE GRAPH — Continuous Research Runner")
    print("=" * 60)
    print(f"  Time: {'forever' if max_seconds == float('inf') else f'{args.hours}h'}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    manager = ProcessManager()

    # Register signal handler for clean shutdown
    def shutdown(sig, frame):
        print("\n\nShutting down all processes...")
        manager.stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Start processes
    py = sys.executable

    if not args.no_deductions:
        manager.start("deduction_loop", [
            py, os.path.join(AUTORESEARCH_DIR, "infinite_deduction.py"),
            "--max-hours", "0",
        ])

    # Wait a bit for deductions to start accumulating
    if not args.no_deductions:
        print("  Waiting 30s for initial deductions...")
        time.sleep(30)

    if not args.no_analysis:
        analysis_script = os.path.join(AUTORESEARCH_DIR, "infinite_analysis.py")
        if os.path.exists(analysis_script):
            manager.start("analysis_loop", [
                py, analysis_script, "--max-hours", "0",
            ])
        else:
            print("  infinite_analysis.py not found, skipping analysis loop")

    # Meta-graph optimization runs in bursts (100 experiments, then pause)
    if not args.no_metagraph:
        # Wait for enough deductions
        time.sleep(30)
        manager.start("metagraph_opt", [
            py, os.path.join(AUTORESEARCH_DIR, "metagraph_loop.py"),
            "--max-experiments", "100",
        ])

    # Additional loops (start after initial data exists)
    time.sleep(10)
    for loop_name, script_name in [
        ("narrative_loop", "narrative_loop.py"),
        ("contradiction_loop", "contradiction_loop.py"),
        ("deepening_loop", "deepening_loop.py"),
        ("rhetorical_loop", "rhetorical_loop.py"),
        ("intertextuality_loop", "intertextuality_loop.py"),
        ("linguistics_loop", "linguistics_loop.py"),
    ]:
        script_path = os.path.join(AUTORESEARCH_DIR, script_name)
        if os.path.exists(script_path):
            manager.start(loop_name, [py, script_path, "--max-hours", "0"])
        else:
            print(f"  {script_name} not found, skipping")

    print("\n  All processes started. Monitoring...\n")

    # Monitor loop
    check_interval = 60  # seconds
    metagraph_runs = 0
    start = time.time()

    while True:
        elapsed = time.time() - start
        if elapsed >= max_seconds:
            print(f"\n  Time limit reached ({args.hours}h)")
            break

        time.sleep(check_interval)

        # Print status periodically
        if int(elapsed) % 300 < check_interval:  # every ~5 minutes
            manager.print_status()

        # Restart dead processes
        for name in list(manager.processes.keys()):
            if name == "metagraph_opt":
                # Meta-graph opt finishes after 100 experiments, restart it
                if not manager.is_alive(name):
                    metagraph_runs += 1
                    if metagraph_runs < 50:  # cap restarts
                        print(f"  Restarting metagraph_opt (run #{metagraph_runs + 1})")
                        manager.start("metagraph_opt", [
                            py, os.path.join(AUTORESEARCH_DIR, "metagraph_loop.py"),
                            "--max-experiments", "100",
                        ])
            else:
                manager.restart_if_dead(name)

    # Final status
    manager.print_status()
    print("\nStopping all processes...")
    manager.stop_all()

    print(f"\n  Continuous runner finished after {(time.time()-start)/3600:.2f}h")


if __name__ == "__main__":
    main()
