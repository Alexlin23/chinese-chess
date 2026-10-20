import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from queue import Queue, Empty
import threading
import time

import pytest

from AlphaZero.engine import GameState
from AlphaZero.train.live import LiveTraining
from AlphaZero.train import live as live_module
from AlphaZero.train import self_play_worker as workers
from AlphaZero.train.runtime_diagnostics import InferenceStatus, InferenceWatch, report_failure
from AlphaZero.tests.test_search_budget_resume import tiny_config


def test_hardware_monitor_does_not_query_gpu(monkeypatch):
    live = object.__new__(LiveTraining)
    live.lock, live.state = threading.Lock(), {"hardware": {}}
    live.stop_event = threading.Event()
    def cpu():
        live.stop_event.set()
        return 25
    monkeypatch.setattr(live_module.psutil, "cpu_percent", cpu)
    live._hardware()
    hardware = live.state["hardware"]
    assert hardware["cpu_percent"] == 25
    assert hardware["gpu_status"] == "disabled"
    assert "gpu_percent" not in hardware and "vram_gb" not in hardware
    assert not hasattr(live_module, "subprocess")


def _blocked_owner(status, ready, release):
    status.mark("graph_capture", version=42, batch_size=8, versions=2)
    ready.set()
    release.wait(10)


def test_stage_and_heartbeat_readable_while_owner_is_blocked(tmp_path):
    context = mp.get_context("spawn")
    status, ready, release = InferenceStatus(context), context.Event(), context.Event()
    process = context.Process(target=_blocked_owner, args=(status, ready, release))
    process.start()
    watch = None
    try:
        assert ready.wait(8)
        snapshot = status.snapshot()
        assert snapshot["stage"] == "graph_capture" and snapshot["pid"] == process.pid
        assert snapshot["model_step"] == 42 and snapshot["completed_batches"] == 0
        watch = InferenceWatch(status, tmp_path, warning_seconds=0, session_id="test")
        watch.start()
        path = tmp_path/"inference-diagnostics.jsonl"
        deadline = time.monotonic()+3
        while (not path.exists() or not path.stat().st_size) and time.monotonic() < deadline:
            time.sleep(.01)
        record = json.loads(path.read_text().splitlines()[0])
        assert record["stalled"] and record["stage"] == "graph_capture"
        assert record["session_id"] == "test" and process.is_alive()
    finally:
        release.set()
        process.join(8)
        if watch: watch.close()
        if process.is_alive(): process.terminate(); process.join()


def test_request_timeout_stays_bounded_and_identifies_request(monkeypatch):
    clock = [0]
    monkeypatch.setattr(workers.time, "monotonic", lambda: clock[0])
    class Responses:
        def get(self, **kwargs):
            clock[0] += 2
            raise Empty
    evaluator = workers.BatchRemoteEvaluator(7, Queue(), Responses())
    evaluator.request_timeout = 3
    with pytest.raises(TimeoutError, match="after 3s: worker=7 request=1"):
        evaluator.evaluate(GameState.new_game())


def _failed_group(worker_ids, config, requests, responses, results, events, stop,
                  sequences, *args, failure_queue=None, request_timeout=300):
    try:
        raise TimeoutError("diagnostic-sentinel inference timeout")
    except Exception as exc:
        report_failure(failure_queue, "worker", exc, worker=worker_ids[0])
        stop.set()
        raise


def test_pipeline_records_actual_worker_error_before_process_exit(tmp_path, monkeypatch):
    module = importlib.import_module("AlphaZero.scripts.pipeline")
    monkeypatch.setattr(module, "self_play_group_fn", _failed_group)
    with pytest.raises(RuntimeError, match="diagnostic-sentinel"):
        module.pipeline(tiny_config(num_simulations=2), output_dir=tmp_path,
                        enable_monitor=False, runtime_diagnostics=True)
    record = json.loads((tmp_path/"runtime-failures.jsonl").read_text().splitlines()[0])
    assert record["source"] == "worker" and record["pid"] != os.getpid()
    assert "TimeoutError" in record["traceback"]
    assert record["inference_snapshot"]["stage"]


def test_pipeline_diagnostics_complete_without_gpu(tmp_path):
    module = importlib.import_module("AlphaZero.scripts.pipeline")
    config = tiny_config(num_simulations=2)
    assert module.pipeline(config, output_dir=tmp_path, enable_monitor=False,
                           runtime_diagnostics=True)["step"] == 2
    records = [json.loads(line) for line in (tmp_path/"inference-diagnostics.jsonl").read_text().splitlines()]
    assert records and records[0]["request_timeout_seconds"] == 300
    assert not (tmp_path/"runtime-failures.jsonl").read_text()


def test_analysis_excludes_stale_gpu_samples_but_keeps_cpu_readings(tmp_path, monkeypatch):
    import sys
    from references import analyze_performance
    rows = []
    for elapsed, status, gpu in [(61, "ok", 40), (62, "stale", 99), (63, "ok", 60)]:
        rows.append(dict(session_id="test", run_id="r", step=elapsed, elapsed_seconds=elapsed,
                         metrics={}, counters={}, hardware=dict(gpu_status=status, gpu_percent=gpu,
                         vram_gb=10, cpu_percent=elapsed, ram_gb=2)))
    (tmp_path/"performance.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    output = tmp_path/"analysis.json"
    monkeypatch.setattr(sys, "argv", ["analyze", "--run-dir", str(tmp_path), "--output", str(output)])
    analyze_performance.main()
    hardware = json.loads(output.read_text())["steady_window"]["hardware"]
    assert hardware["gpu_percent"]["samples"] == 2 and hardware["gpu_percent"]["mean"] == 50
    assert hardware["cpu_percent"]["samples"] == 3
