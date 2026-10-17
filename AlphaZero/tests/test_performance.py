"""Low-cost observers preserve RNG, search targets, SGD and checkpoint contracts."""
from dataclasses import replace
import json
from queue import Queue
import random
import sqlite3
import threading

import numpy as np
import pytest
import torch

from AlphaZero.engine import GameState
from AlphaZero.scripts.pipeline import pipeline
from AlphaZero.tests.test_mcts_correctness import ConstantEvaluator
from AlphaZero.train import self_play_worker as workers
from AlphaZero.train.config import AlphaZeroConfig
from AlphaZero.train.inference_server import BatchResponse
from AlphaZero.train.performance import Metric, Recorder, Hub
from AlphaZero.train.replay import ReplayBuffer, pack_sample
from AlphaZero.train.trainer import Trainer


def config():
    return AlphaZeroConfig(num_blocks=1, num_filters=4, batch_size=2, device="cpu", precision="fp32",
                           num_workers=1, actor_processes=1, num_simulations=8, max_game_ply=4,
                           checkpoint_interval=2, recovery_interval=1, max_optimizer_steps=2, max_iterations=1)


def test_metric_mean_counts_and_merged_p95_upper_bound():
    left, right, merged = Metric(), Metric(), Metric()
    for value in range(1, 101):
        (left if value <= 50 else right).add(value / 1000)
    merged.merge(left.raw())
    merged.merge(right.raw())
    result = merged.summary()
    assert result["count"] == 100 and result["total"] == pytest.approx(5.05)
    assert result["mean"] == pytest.approx(.0505)
    assert .095 <= result["p95_upper_approx"] <= .1
    extreme = Metric()
    extreme.add(100000)
    assert extreme.summary()["p95_upper_approx"] == 100000
    for value in (float("nan"), -1, float("inf")):
        with pytest.raises(ValueError):
            left.add(value)


def test_cumulative_drop_recovery_stale_packets_and_parallel_rates():
    clock = [0.0]
    local = Recorder("learner")
    hub = Hub(local, clock=lambda: clock[0])
    queue = Queue(maxsize=1)
    first, second = Recorder("worker:0", queue), Recorder("worker:1")
    first.increment("mcts.simulations", 800)
    first.observe("mcts.hand_wall", 10)
    first.flush(force=True)
    first.increment("mcts.simulations", 800)
    first.flush(force=True)  # Full queue drops this snapshot, not training data.
    stale = queue.get_nowait()
    first.flush(force=True)
    hub.receive(queue.get_nowait())
    hub.receive(stale)
    second.increment("mcts.simulations", 800)
    second.observe("mcts.hand_wall", 10)
    hub.receive(second.raw())
    local.increment("data.new_positions", 100)
    local.increment("data.completed_games", 2)
    clock[0] = 10
    result = hub.snapshot()
    assert result["metrics"]["mcts.hand_wall"]["total"] == 20  # Concurrent: elapsed remains 10.
    assert result["elapsed_seconds"] == 10
    assert result["rates"]["mcts_simulations_per_second"] == 240
    assert result["rates"]["new_positions_per_hour"] == 36000
    assert result["counters"]["monitor.dropped_reports"] == 1


def test_cpu_sgd_and_rng_are_identical_with_observer_enabled(tmp_path):
    torch.set_num_threads(1)
    state = GameState.new_game()
    replay = ReplayBuffer()
    replay.add_game("one", [pack_sample(state, state.legal_mask() / state.legal_mask().sum(), 0)] * 4)
    initial = Trainer(config()).model.state_dict()
    saved = []
    for enabled in (False, True):
        trainer = Trainer(config())
        trainer.model.load_state_dict(initial)
        if enabled:
            trainer.performance = Recorder("learner")
        torch.manual_seed(81)
        np.random.seed(81)
        random.seed(81)
        stats = [trainer.step(replay) for _ in range(3)]
        path = tmp_path / f"{enabled}.pt"
        trainer.save_checkpoint(path)
        payload = torch.load(path, weights_only=False)
        saved.append((stats, payload))
    replay.close()
    assert saved[0][0] == saved[1][0]
    before, after = saved[0][1], saved[1][1]
    assert before.keys() == after.keys()
    assert before["spec"] == after["spec"] and before["config"] == after["config"]
    assert before["python_rng_state"] == after["python_rng_state"]
    assert torch.equal(before["torch_rng_state"], after["torch_rng_state"])
    assert np.array_equal(before["numpy_rng_state"][1], after["numpy_rng_state"][1])
    assert before["numpy_rng_state"][0] == after["numpy_rng_state"][0]
    assert before["numpy_rng_state"][2:] == after["numpy_rng_state"][2:]
    for key in before["model_state_dict"]:
        assert torch.equal(before["model_state_dict"][key], after["model_state_dict"][key])
    for key, value in before["optimizer_state_dict"]["state"].items():
        assert torch.equal(value["momentum_buffer"], after["optimizer_state_dict"]["state"][key]["momentum_buffer"])


def test_worker_observer_does_not_change_seeded_moves_visits_or_samples(monkeypatch):
    class Evaluator(ConstantEvaluator):
        model_step = 12
    monkeypatch.setattr(workers, "BatchRemoteEvaluator", lambda *args: Evaluator())
    games = []
    for enabled in (False, True):
        stop = threading.Event()
        class OneGame(Queue):
            def put(self, message, **kwargs):
                super().put(message, **kwargs)
                stop.set()
        results = OneGame()
        workers.self_play_worker_fn(0, replace(config(), num_simulations=800), None, None,
                                    results, Queue(), stop, 3, Queue() if enabled else None)
        games.append(results.get_nowait())
    assert games[0] == games[1]


def test_rtt_includes_queue_and_response_wait_without_claiming_pure_ipc(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(workers.time, "perf_counter", lambda: clock[0])
    class Requests(Queue):
        def put(self, message, **kwargs):
            clock[0] += 1
            super().put(message, **kwargs)
    class Responses(Queue):
        def get(self, **kwargs):
            clock[0] += 3
            return super().get(**kwargs)
    state, responses = GameState.new_game(), Responses()
    policy = state.legal_mask().astype(np.float32)
    policy /= policy.sum()
    responses.put(BatchResponse(1, policy[None], np.zeros(1), 0, 1))
    evaluator = workers.BatchRemoteEvaluator(0, Requests(), responses)
    evaluator.performance = Recorder("worker:0")
    evaluator.evaluate(state)
    metric = evaluator.performance.raw()["metrics"]["communication.rtt_wall"]
    assert metric["total"] == 4 and evaluator.performance.neural_sync_total == 4


def test_cuda_sampling_reads_only_ready_events_and_never_synchronizes(monkeypatch):
    made = []
    class Event:
        ready = True
        def __init__(self, enable_timing):
            made.append(self)
        def record(self):
            pass
        def query(self):
            return self.ready
        def elapsed_time(self, other):
            return 2.5
    monkeypatch.setattr(torch.cuda, "Event", Event)
    observer = Recorder("gpu", gpu_every=4)
    sample = observer.cuda_sample("cuda", ("transfer", "compute"))
    sample.mark()
    sample.mark()
    sample.finish()
    assert observer.raw()["metrics"]["compute"]["total"] == .0025
    assert all(observer.cuda_sample("cuda", ("compute",)) is None for _ in range(3))
    sample = observer.cuda_sample("cuda", ("compute",))
    sample.mark()
    sample.events[-1].ready = False
    sample.finish()
    assert observer.raw()["counters"]["monitor.cuda_sample_not_ready"] == 1
    assert len(made) == 5


def test_pipeline_enable_disable_and_existing_checkpoint_resume(tmp_path):
    first_games, payloads = [], []
    for enabled in (False, True):
        directory = tmp_path / str(enabled)
        result = pipeline(config(), output_dir=directory, enable_monitor=False,
                          performance=enabled, performance_interval=.05)
        assert result["step"] == 2
        payloads.append(torch.load(directory / "learner.pt", weights_only=False))
        with sqlite3.connect(directory / "replay.sqlite") as db:
            first_games.append((db.execute("SELECT trace FROM games ORDER BY id LIMIT 1").fetchone(),
                                db.execute("SELECT planes,globals,actions,probabilities,value FROM positions WHERE game=1 ORDER BY id").fetchall()))
        assert (directory / "performance.json").exists() == enabled
        if enabled:
            report = json.loads((directory / "performance.json").read_text())
            assert report["metrics"]["sgd.update_wall"]["count"] == 2
            assert report["metrics"]["mcts.hand_wall"]["count"] >= 4
            assert report["metrics"]["inference.batch_size"]["count"] > 0
            assert report["counters"]["data.completed_games"] >= 1
    assert payloads[0].keys() == payloads[1].keys()
    assert payloads[0]["spec"] == payloads[1]["spec"]
    assert first_games[0] == first_games[1]
    # An uninstrumented checkpoint resumes with monitoring enabled, unchanged fingerprint.
    directory = tmp_path / "False"
    fingerprint = payloads[0]["fingerprint"]
    result = pipeline(replace(config(), max_optimizer_steps=4, max_iterations=2),
                      output_dir=directory, resume=directory / "learner.pt", enable_monitor=False,
                      performance=True, performance_interval=.05)
    assert result["step"] == 4
    assert torch.load(directory / "learner.pt", weights_only=False)["fingerprint"] == fingerprint
