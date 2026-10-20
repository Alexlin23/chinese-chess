"""Changing search effort must preserve learner state and reach new actors."""
from dataclasses import replace
import json
from queue import Queue
import sqlite3
import threading

import numpy as np
import pytest
import torch

from AlphaZero.engine import GameState
from AlphaZero.scripts.pipeline import pipeline
from AlphaZero.tests.test_mcts_correctness import ConstantEvaluator
from AlphaZero.train.config import AlphaZeroConfig
from AlphaZero.train import self_play_worker as workers
from AlphaZero.train.trainer import Trainer


def tiny_config(**kwargs):
    return AlphaZeroConfig(num_blocks=1, num_filters=4, batch_size=2,
        device="cpu", precision="fp32", num_workers=1, actor_processes=1,
        max_game_ply=4, checkpoint_interval=2, recovery_interval=1,
        max_optimizer_steps=2, max_iterations=1, **kwargs)


@pytest.mark.parametrize("budget,interval", [(1, 2), (3, 2), (32, 2), (800, 2), (400, 32000)])
def test_budget_change_restores_weights_momentum_step_and_rng(tmp_path, budget, interval):
    torch.set_num_threads(1)
    config = tiny_config(num_simulations=8)
    trainer = Trainer(config)
    state = GameState.new_game()
    class Batch:
        def __len__(self): return 2
        def sample(self, count):
            policy = state.legal_mask().astype(np.float32)
            policy /= policy.sum()
            return (np.stack([state.encode()] * count), np.stack([policy] * count),
                    np.zeros(count, np.float32), np.ones(count))
    trainer.step(Batch())
    path = tmp_path / "learner.pt"
    trainer.save_checkpoint(path, {"marker": 123})
    original = torch.load(path, weights_only=False)
    resumed = Trainer(replace(config, num_simulations=budget, checkpoint_interval=interval))
    assert resumed.load_checkpoint(path) == {"marker": 123}
    assert resumed.global_optimizer_step == 1
    assert resumed.config.num_simulations == budget
    assert resumed.config.fingerprint == config.fingerprint
    expected_changes = {"num_simulations": {"old": 8, "new": budget}}
    if interval != config.checkpoint_interval:
        expected_changes["checkpoint_interval"] = {"old": 2, "new": interval}
    assert resumed.resume_config_changes == expected_changes
    assert resumed.current_iteration == resumed.global_optimizer_step // interval
    for key, value in resumed.model.state_dict().items():
        assert torch.equal(value, original["model_state_dict"][key])
    restored = resumed.optimizer.state_dict()
    assert restored["param_groups"] == original["optimizer_state_dict"]["param_groups"]
    for key, value in restored["state"].items():
        assert torch.equal(value["momentum_buffer"],
                           original["optimizer_state_dict"]["state"][key]["momentum_buffer"])
    assert torch.equal(torch.get_rng_state(), original["torch_rng_state"])
    assert resumed.step(Batch())["step"] == 2


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError, match="num_simulations"):
        tiny_config(num_simulations=budget)


def test_resume_changed_publication_interval_uses_global_step_boundaries(tmp_path, monkeypatch):
    directory = tmp_path / "run"
    config = tiny_config(num_simulations=8)
    pipeline(config, output_dir=directory, enable_monitor=False)
    with sqlite3.connect(directory / "replay.sqlite") as db:
        previous_games = db.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    published = []
    export_actor = Trainer.export_actor
    def record_publish(self, path):
        published.append(self.global_optimizer_step)
        return export_actor(self, path)
    monkeypatch.setattr(Trainer, "export_actor", record_publish)
    resumed_config = replace(config, num_simulations=4, checkpoint_interval=4,
                             max_optimizer_steps=8, max_iterations=2,
                             min_replay_games=previous_games + 1)
    result = pipeline(resumed_config, output_dir=directory, enable_monitor=False,
                      resume=directory / "learner.pt")
    assert result["step"] == 8
    assert published == [2, 4, 8]  # Initial resume snapshot, then global multiples.
    checkpoint = torch.load(directory / "learner.pt", weights_only=False)
    assert checkpoint["current_iteration"] == 2
    assert checkpoint["config"]["checkpoint_interval"] == 4
    assert checkpoint["config"]["recovery_interval"] == 1
    records = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    session = [r for r in records if r.get("event") == "session_start"][-1]
    assert session["step"] == 2
    assert session["config_changes"]["checkpoint_interval"] == {"old": 2, "new": 4}
    assert session["config_changes"]["num_simulations"] == {"old": 8, "new": 4}
    with sqlite3.connect(directory / "replay.sqlite") as db:
        traces = db.execute("SELECT trace FROM games ORDER BY id LIMIT -1 OFFSET ?",
                            (previous_games,)).fetchall()
    assert traces and all(json.loads(row[0])["num_simulations"] == 4 for row in traces)


def test_unrelated_semantic_and_model_changes_still_rejected(tmp_path):
    config = tiny_config(num_simulations=8)
    path = tmp_path / "learner.pt"
    Trainer(config).save_checkpoint(path)
    for changed, message in [(replace(config, learning_rate=.1), "learning_rate"),
                             (replace(config, num_filters=8), "specification")]:
        with pytest.raises(ValueError, match=message):
            Trainer(changed).load_checkpoint(path)


def test_position_cap_change_resumes_legacy_checkpoint_without_resetting_sgd(tmp_path):
    config = tiny_config(num_simulations=8)
    original = Trainer(config)
    original.global_optimizer_step = 123
    parameter = next(original.model.parameters())
    original.optimizer.state[parameter]["momentum_buffer"] = torch.ones_like(parameter)
    path = tmp_path/"learner.pt"
    original.save_checkpoint(path)
    legacy = torch.load(path, weights_only=False)
    del legacy["config"]["replay_window_positions"]
    torch.save(legacy, path)
    resumed = Trainer(replace(config, replay_window_positions=100000))
    resumed.load_checkpoint(path)
    assert resumed.global_optimizer_step == 123
    assert resumed.resume_config_changes == {"replay_window_positions": {"old": 0, "new": 100000}}
    assert resumed.config.fingerprint == config.fingerprint
    for key, value in resumed.model.state_dict().items():
        assert torch.equal(value, legacy["model_state_dict"][key])
    momentum = resumed.optimizer.state[next(resumed.model.parameters())]["momentum_buffer"]
    assert torch.equal(momentum, torch.ones_like(momentum))
    assert torch.equal(torch.get_rng_state(), legacy["torch_rng_state"])
    actor = tmp_path/"actor.pt"
    resumed.export_actor(actor)
    assert "replay_window_positions" not in torch.load(actor, weights_only=False)["config"]
    resumed.save_checkpoint(path)
    assert torch.load(path, weights_only=False)["config"]["replay_window_positions"] == 100000


@pytest.mark.parametrize("cap", [-1, True, 2.5, 3])
def test_invalid_position_cap_rejected(cap):
    with pytest.raises(ValueError, match="replay_window_positions"):
        tiny_config(replay_window_positions=cap)


@pytest.mark.parametrize("budget", [1, 3, 17])
def test_worker_uses_exact_new_budget_on_every_move(monkeypatch, budget):
    config = tiny_config(num_simulations=budget)
    stop = threading.Event()
    class Evaluator(ConstantEvaluator):
        model_step = 2
    monkeypatch.setattr(workers, "BatchRemoteEvaluator", lambda *args: Evaluator())
    class OneGame(Queue):
        def put(self, item, **kwargs):
            super().put(item, **kwargs)
            stop.set()
    results, events = OneGame(), Queue()
    workers.self_play_worker_fn(0, config, None, None, results, events, stop)
    game = results.get_nowait()
    assert game["ply"] == 4
    assert json.loads(game["trace"])["num_simulations"] == budget
    boards = [events.get_nowait() for _ in range(events.qsize())]
    searched = [event["diagnostics"] for event in boards if event.get("diagnostics")]
    assert len(searched) == 3
    assert all(d["simulations"] == d["root_visits"] == budget for d in searched)
    for sample in game["samples"]:
        sample.validate()


def test_pipeline_resume_lower_then_higher_budget(tmp_path):
    directory = tmp_path / "run"
    config = tiny_config(num_simulations=8)
    previous_games = []
    run_id = None
    for index, budget in enumerate([8, 3, 11], start=1):
        config = replace(config, num_simulations=budget, max_optimizer_steps=index * 2,
                         max_iterations=index, min_replay_games=len(previous_games) + 1)
        result = pipeline(config, output_dir=directory, enable_monitor=False,
                          resume=directory / "learner.pt" if index > 1 else None)
        assert result["step"] == index * 2
        checkpoint = torch.load(directory / "learner.pt", weights_only=False)
        assert checkpoint["config"]["num_simulations"] == budget
        if run_id is not None:
            assert checkpoint["run_id"] == run_id
        run_id = checkpoint["run_id"]
        with sqlite3.connect(directory / "replay.sqlite") as db:
            games = db.execute("SELECT id,episode,trace FROM games ORDER BY id").fetchall()
        assert games[:len(previous_games)] == previous_games
        assert len(games) > len(previous_games)
        assert all(json.loads(g[2])["num_simulations"] == budget
                   for g in games[len(previous_games):])
        previous_games = games
    records = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    sessions = [r for r in records if r.get("event") == "session_start"]
    assert [s["step"] for s in sessions] == [0, 2, 4]
    assert [s["config"]["num_simulations"] for s in sessions] == [8, 3, 11]
    assert sessions[0]["config_changes"] == {}
    assert sessions[1]["config_changes"]["num_simulations"] == {"old": 8, "new": 3}
    assert sessions[2]["config_changes"]["num_simulations"] == {"old": 3, "new": 11}
    assert [r["num_simulations"] for r in records if "event" not in r] == [8, 8, 3, 3, 11, 11]
