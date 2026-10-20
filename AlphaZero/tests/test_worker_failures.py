"""Real failures propagate without stage samplers or GPU management queries."""
import importlib
from queue import Queue, Empty
import threading
import pytest
from AlphaZero.engine import GameState
from AlphaZero.train.live import LiveTraining
from AlphaZero.train import live as live_module
from AlphaZero.train import self_play_worker as workers
from AlphaZero.train.inference_server import report_failure
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
    assert live.state["hardware"]["cpu_percent"] == 25
    assert not any("gpu" in key or "vram" in key for key in live.state["hardware"])
    assert not hasattr(live_module, "subprocess")


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
        raise TimeoutError("sentinel inference timeout")
    except Exception as exc:
        report_failure(failure_queue, "worker", exc, worker=worker_ids[0])
        stop.set()
        raise


def test_pipeline_propagates_actual_worker_error_before_process_exit(tmp_path, monkeypatch):
    module = importlib.import_module("AlphaZero.scripts.pipeline")
    monkeypatch.setattr(module, "self_play_group_fn", _failed_group)
    with pytest.raises(RuntimeError, match="sentinel inference timeout") as error:
        module.pipeline(tiny_config(num_simulations=2), output_dir=tmp_path, enable_monitor=False)
    assert "TimeoutError" in str(error.value)
