"""Keep a small CPU-only dashboard alive while its training process stops/resumes."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import urllib.request

import psutil

from AlphaZero.scripts.paths import DEFAULT_CONFIG, PROJECT_ROOT, run_paths
from AlphaZero.train.live import LiveTraining


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def process_identity(pid):
    process = psutil.Process(pid)
    value = dict(pid=pid, create_time=process.create_time(),
                 command=process.cmdline(), cwd=process.cwd())
    if sys.platform == "linux":
        value.update(boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                     start_ticks=int(Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[19]))
    return value


class TrainingDashboard:
    def __init__(self, config_path=DEFAULT_CONFIG, run_dir=None, *, performance=False,
                 performance_interval=5, performance_gpu_every=64, inference_timeout=300):
        self.config_path = str((PROJECT_ROOT / config_path).resolve())
        self.config, self.directory = run_paths(self.config_path, run_dir)
        self.options = (performance, performance_interval, performance_gpu_every, inference_timeout)
        self.lock = threading.Lock()
        self.process = None
        self.children = {}
        self.worker_url = None
        self.worker_state = {}
        self.stop_requested = threading.Event()
        self.stop_sent = False
        self.live = LiveTraining(self.config.telemetry_host, self.config.telemetry_port,
                                 self.config.telemetry_interval)
        self.live.enable_stop(self.stop)
        self.live.update(phase="stopped", stop_available=False, managed=True,
                         workers=self.config.num_workers, actor_processes=self.config.actor_processes,
                         config=self.config.to_dict())
        self.live.enable_resume(self.start)
        stop_file = self.directory / "training-stop.json"
        if stop_file.exists():
            saved = json.loads(stop_file.read_text())
            self.live.update(**{k: saved[k] for k in ("step", "model_step", "positions", "completed_games")},
                             actor_step=saved["model_step"], saved=saved.get("saved", False))
        self.live.update(resume_available=self.can_resume())

    def can_resume(self):
        return all((self.directory / name).is_file() for name in ("learner.pt", "latest.pt"))

    def start(self):
        with self.lock:
            if self.process is not None:
                raise RuntimeError("上一轮训练尚未完全退出，请等待。")
            if not self.can_resume():
                raise RuntimeError("没有可继续训练的存档。")
            # Refuse another owner even if it was launched outside this dashboard.
            if os.name == "posix":
                import fcntl
                with (self.directory / ".run.lock").open("a+") as stream:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            performance, interval, gpu_every, timeout = self.options
            command = [sys.executable, "-u", "-m", "AlphaZero.scripts.pipeline",
                       "--config", self.config_path, "--output-dir", str(self.directory),
                       "--resume", str(self.directory / "learner.pt"), "--telemetry-port", "0",
                       "--inference-timeout", str(timeout),
                       "--performance" if performance else "--no-performance",
                       "--performance-interval", str(interval), "--performance-gpu-every", str(gpu_every)]
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            self.stdout = self.directory / f"dashboard-training-{stamp}.log"
            self.stderr = self.directory / f"dashboard-training-{stamp}.stderr.log"
            with self.stdout.open("xb") as output, self.stderr.open("xb") as errors:
                self.process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL,
                    stdout=output, stderr=errors, start_new_session=True)
            self.children = {}
            self.worker_url = None
            self.worker_state = {}
            self.stop_requested.clear()
            self.stop_sent = False
            self.control = dict(process_identity(self.process.pid), status="starting",
                stdout=str(self.stdout), stderr=str(self.stderr), run_dir=str(self.directory))
            self.save_control()
            self.live.update(phase="starting", boards={}, terminal=False, errors=[], saved=False,
                stop_requested=False, stop_available=False, resume_available=False, resume_requested=True,
                performance=dict(enabled=performance), loss=None, value_loss=None, policy_loss=None)

    def save_control(self):
        write_json(self.directory / "dashboard-training-control.json", self.control)

    def stop(self):
        self.stop_requested.set()

    def tick(self):
        with self.lock:
            if self.process is None:
                return
            try:
                for child in psutil.Process(self.process.pid).children(recursive=True):
                    self.children[child.pid] = child.create_time()
            except psutil.NoSuchProcess:
                pass
            code = self.process.poll()
            if code is not None:
                remaining = []
                for pid, created in self.children.items():
                    try:
                        p = psutil.Process(pid)
                        if p.create_time() == created and p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                            remaining.append(pid)
                    except psutil.NoSuchProcess:
                        pass
                if remaining:
                    self.live.update(phase="stopping", stop_available=False, resume_available=False)
                    return
                saved = {}
                stop_file = self.directory / "training-stop.json"
                if stop_file.exists():
                    candidate = json.loads(stop_file.read_text())
                    if candidate.get("session_id") == self.worker_state.get("session_id"):
                        saved = candidate
                failed = code != 0
                self.control.update(status="failed" if failed else "stopped", exit_code=code,
                    ended_utc=datetime.now(timezone.utc).isoformat(), children_exited=True,
                    saved_step=saved.get("step"), saved_model_step=saved.get("model_step"))
                self.save_control()
                self.process = None
                self.live.update(phase="failed" if failed else "stopped", boards={}, terminal=False,
                    stop_available=False, stop_requested=False, resume_requested=False,
                    resume_available=self.can_resume(), saved=bool(saved.get("saved")))
                if saved:
                    self.live.update(step=saved["step"], actor_step=saved["model_step"],
                        completed_games=saved["completed_games"], positions=saved["positions"])
                if failed:
                    self.live.update(errors=[dict(error=f"训练进程已退出（{code}），可从最后存档继续；详情：{self.stderr.name}")])
                return
            if self.worker_url is None:
                match = re.search(r"^Training dashboard: (http://127\.0\.0\.1:\d+)$",
                                  self.stdout.read_text(errors="replace")[:8192], re.MULTILINE)
                if not match:
                    return
                self.worker_url = match[1]
            try:
                with urllib.request.urlopen(self.worker_url + "/api/snapshot", timeout=2) as response:
                    self.worker_state = json.load(response)
                state = {k: v for k, v in self.worker_state.items() if k not in (
                    "control_token", "stop_available", "stop_requested", "terminal", "hardware")}
                state.update(phase="stopping" if self.stop_requested.is_set() or self.worker_state.get("terminal")
                             else state["phase"], stop_available=bool(self.worker_state.get("stop_available")),
                             resume_available=False, resume_requested=False)
                self.live.update(**state)
                if self.control.get("session_id") != state.get("session_id"):
                    self.control.update(session_id=state.get("session_id"), run_id=state.get("run_id"), status="running")
                    self.save_control()
                if self.stop_requested.is_set() and not self.stop_sent:
                    request = urllib.request.Request(self.worker_url + "/api/stop",
                        data=json.dumps(dict(token=self.worker_state["control_token"])).encode(),
                        headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(request, timeout=2) as response:
                        self.stop_sent = response.status == 202
            except (OSError, ValueError):
                pass  # A closing/starting worker is not proof of failure; poll its actual process.

    def close(self):
        if self.process is not None:
            self.stop()
            deadline = time.monotonic() + 30
            while self.process is not None and time.monotonic() < deadline:
                self.tick()
                time.sleep(.1)
            if self.process is not None:
                raise RuntimeError("训练尚未退出，保留进程供人工处理。")
        self.live.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir")
    parser.add_argument("--start", action="store_true", help="resume immediately; otherwise show a stopped dashboard")
    parser.add_argument("--performance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--performance-interval", type=float, default=5)
    parser.add_argument("--performance-gpu-every", type=int, default=64)
    parser.add_argument("--inference-timeout", type=float, default=300)
    args = parser.parse_args()
    dashboard = TrainingDashboard(args.config, args.run_dir, performance=args.performance,
        performance_interval=args.performance_interval, performance_gpu_every=args.performance_gpu_every,
        inference_timeout=args.inference_timeout)
    write_json(dashboard.directory / "dashboard-control.json", process_identity(os.getpid()))
    print("Training control: http://%s:%s" % dashboard.live.address, flush=True)
    try:
        if args.start:
            dashboard.live.request_resume()
        while True:
            dashboard.tick()
            time.sleep(.2)
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.close()


if __name__ == "__main__":
    main()
