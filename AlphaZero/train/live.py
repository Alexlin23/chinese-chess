"""Read-only HTTP/SSE observation server. Slow viewers never block actors."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import os
import threading
import time
import psutil

class LiveTraining:
    def __init__(self, host="127.0.0.1", port=8765, interval=0.25):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.interval = interval
        self.finished = {}
        self.state = dict(phase="starting", boards={}, completed_games=0,
                          positions=0, step=0, loss=None, errors=[], hardware={},
                          started=time.time(), rule_status="development: CNXA2020 adjudication incomplete")
        owner = self
        root = Path(__file__).resolve().parents[2] / "frontend"
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def log_message(self, *args):
                pass
            def do_GET(self):
                if self.path == "/events":
                    self.connection.settimeout(5)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("X-Accel-Buffering", "no")
                    self.end_headers()
                    try:
                        while not owner.stop_event.is_set():
                            payload = json.dumps(owner.snapshot(), ensure_ascii=False, allow_nan=False)
                            self.wfile.write(("data: " + payload + "\n\n").encode())
                            self.wfile.flush()
                            owner.stop_event.wait(owner.interval)
                        self.wfile.write(("data: " + json.dumps(owner.snapshot(), ensure_ascii=False,
                            allow_nan=False) + "\n\n").encode())
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                    self.close_connection = True
                    return
                if self.path == "/api/snapshot":
                    data = json.dumps(owner.snapshot(), ensure_ascii=False, allow_nan=False).encode()
                    content_type = "application/json; charset=utf-8"
                else:
                    paths = {"/":"training.html", "/training.js":"training.js", "/training.css":"training.css"}
                    name = paths.get(self.path.split("?")[0])
                    if not name:
                        self.send_error(404)
                        return
                    data = (root / name).read_bytes()
                    content_type = ("text/html" if name.endswith("html") else
                                    "text/css" if name.endswith("css") else "text/javascript") + "; charset=utf-8"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.hardware_thread = threading.Thread(target=self._hardware, daemon=True)
        self.hardware_thread.start()

    @property
    def address(self):
        return self.server.server_address

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def event(self, event):
        with self.lock:
            if event["kind"] == "board":
                worker = event["worker"]
                if event["sequence"] > self.finished.get(worker, -1):
                    old = self.state["boards"].get(str(worker))
                    if old is None or (event["sequence"], event["ply"]) >= (old["sequence"], old["ply"]):
                        self.state["boards"][str(worker)] = event
            elif event["kind"] == "worker_error":
                self.state["errors"] = (self.state["errors"] + [event])[-20:]

    def complete(self, event, positions, games):
        with self.lock:
            worker = event["worker"]
            self.finished[worker] = max(self.finished.get(worker, -1), event["sequence"])
            board = self.state["boards"].get(str(worker))
            if board and board["episode"] == event["episode"]:
                del self.state["boards"][str(worker)]
            self.state.update(positions=positions, completed_games=games, last_result=dict(
                episode=event["episode"], reason=event["reason"], winner=event["winner"], ply=event["ply"]))

    def snapshot(self):
        with self.lock:
            result = json.loads(json.dumps(self.state, allow_nan=False))
        result["hardware"] = self.hardware_snapshot()
        return result

    def hardware_snapshot(self):
        with self.lock:
            return dict(self.state.get("hardware", {}))

    def _hardware(self):
        """Sample only host-safe CPU/RAM data; never query the GPU from WSL."""
        while not self.stop_event.is_set():
            memory = psutil.virtual_memory()
            with self.lock:
                self.state["hardware"].update(
                    cpu_percent=psutil.cpu_percent(),
                    ram_gb=memory.used / 1024**3,
                    ram_total_gb=memory.total / 1024**3,
                    memory_scope="WSL" if "WSL_DISTRO_NAME" in os.environ else "系统",
                    gpu_status="disabled",
                    gpu_sample_error="GPU监控已禁用：训练期间不查询GPU",
                )
            self.stop_event.wait(3)

    def close(self):
        phase = self.snapshot()["phase"]
        self.update(phase="failed" if phase == "failed" else "stopped", boards={}, terminal=True)
        self.stop_event.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.hardware_thread.join(3)
