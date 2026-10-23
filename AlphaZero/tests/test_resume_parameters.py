"""Same-directory horizon conversion, crash recovery and publication budgets."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest
import torch

from AlphaZero.scripts.pipeline import pipeline
from AlphaZero.tests.test_ply_limit_migration import source_run
from AlphaZero.train.config import AlphaZeroConfig
from AlphaZero.train.replay import ReplayBuffer
from AlphaZero.train import resume as adjustment
from AlphaZero.train.trainer import Trainer


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


@pytest.mark.parametrize("limit", [10, 20])
def test_horizon_change_keeps_directory_run_step_rng_and_replay_watermarks(source_run, limit):
    old = load(source_run / "learner.pt")
    config = replace(AlphaZeroConfig(**old["config"]), max_game_ply=limit)
    changes = adjustment.prepare_resume(source_run, config, source_run / "learner.pt")
    assert changes["max_game_ply"] == {"old": 12, "new": limit}
    new = load(source_run / "learner.pt")
    assert new["run_id"] == old["run_id"] and new["step"] == old["step"] == 1
    assert new["fingerprint"] == config.fingerprint
    for key, value in old["model_state_dict"].items():
        assert torch.equal(new["model_state_dict"][key], value)
    for key, value in old["optimizer_state_dict"]["state"].items():
        assert torch.equal(new["optimizer_state_dict"]["state"][key]["momentum_buffer"], value["momentum_buffer"])
    assert torch.equal(new["torch_rng_state"], old["torch_rng_state"])
    np.testing.assert_array_equal(new["numpy_rng_state"][1], old["numpy_rng_state"][1])
    actor = load(source_run / "latest.pt")
    assert actor["fingerprint"] == config.fingerprint and actor["step"] == 1
    replay = ReplayBuffer(source_run / "replay.sqlite")
    try:
        assert replay.run_id == old["run_id"] and replay.game_count == 1 and len(replay) == 8
        assert replay.generation == 2  # Last game was discarded; its ID must not be reused.
        assert replay.total_positions == 20 and replay.next_sequences() == {0: 2}
        assert new["extra"]["replay_generation"] == 2
        observed = [np.frombuffer(row[0], dtype="<f4")[1] for row in replay.db.execute("SELECT globals FROM positions ORDER BY id")]
        np.testing.assert_array_equal(observed, np.arange(8, dtype=np.float32) / limit)
        trainer = Trainer(config)
        trainer.load_checkpoint(source_run / "learner.pt")
        assert trainer.step(replay)["step"] == 2
    finally:
        replay.close()
    assert not (source_run / ".resume-adjustment.json").exists()
    records = [json.loads(line) for line in (source_run / "parameter-changes.jsonl").read_text().splitlines()]
    assert len(records) == 1 and records[0]["removed_games"] == 1
    assert records[0]["run_id"] == old["run_id"]


@pytest.mark.parametrize("field,value", [("learning_rate", .1), ("num_filters", 8)])
def test_unrelated_incompatible_parameters_do_not_modify_saved_run(source_run, field, value):
    original = (source_run / "learner.pt").read_bytes()
    config = replace(AlphaZeroConfig(**load(source_run / "learner.pt")["config"]),
                     max_game_ply=10, **{field: value})
    with pytest.raises(ValueError, match="semantics|specification"):
        adjustment.prepare_resume(source_run, config, source_run / "learner.pt")
    assert (source_run / "learner.pt").read_bytes() == original
    assert not (source_run / ".resume-adjustment.json").exists()


def test_invalid_replay_rolls_out_without_installing_new_checkpoint(source_run):
    original = (source_run / "learner.pt").read_bytes()
    with sqlite3.connect(source_run / "replay.sqlite") as db:
        db.execute("UPDATE positions SET globals=? WHERE id=1", (np.ones(3, dtype="<f4").tobytes(),))
    config = replace(AlphaZeroConfig(**load(source_run / "learner.pt")["config"]), max_game_ply=10)
    with pytest.raises(ValueError, match="observation"):
        adjustment.prepare_resume(source_run, config, source_run / "learner.pt")
    assert (source_run / "learner.pt").read_bytes() == original
    assert not (source_run / ".resume-adjustment.json").exists()


@pytest.mark.parametrize("power_loss", [False, True])
def test_partial_file_replacement_is_recoverable(source_run, monkeypatch, power_loss):
    digest = hashlib.sha256((source_run / "learner.pt").read_bytes()).hexdigest()
    config = AlphaZeroConfig(**load(source_run / "learner.pt")["config"])
    real_replace = adjustment.os.replace
    failed = False
    class PowerLoss(BaseException):
        pass
    def interrupted(source, target):
        nonlocal failed
        if not failed and Path(source).name == "learner.pt" and Path(target) == source_run / "learner.pt":
            failed = True
            raise PowerLoss() if power_loss else OSError("injected install failure")
        return real_replace(source, target)
    monkeypatch.setattr(adjustment.os, "replace", interrupted)
    with pytest.raises(PowerLoss if power_loss else OSError):
        adjustment.prepare_resume(source_run, replace(config, max_game_ply=10), source_run / "learner.pt")
    if power_loss:
        assert (source_run / ".resume-adjustment.json").exists()
        assert adjustment.prepare_resume(source_run, config, source_run / "learner.pt") == {}
    assert hashlib.sha256((source_run / "learner.pt").read_bytes()).hexdigest() == digest
    assert not (source_run / "latest.pt").exists()  # Source had no actor before adjustment.
    assert not (source_run / ".resume-adjustment.json").exists()
    replay = ReplayBuffer(source_run / "replay.sqlite")
    try:
        assert len(replay) == 20 and replay.game_count == 2
        replay.bind_spec(config.fingerprint)
    finally:
        replay.close()


def tiny_config(directory):
    return AlphaZeroConfig(num_blocks=1, num_filters=4, device="cpu", precision="fp32",
                           num_workers=1, actor_processes=1, num_simulations=1, batch_size=2,
                           max_game_ply=4, checkpoint_interval=2, recovery_interval=1,
                           max_optimizer_steps=8, max_iterations=4, checkpoint_dir=str(directory), torch_threads=1)


def test_publication_budget_counts_global_boundaries_after_resume(tmp_path, monkeypatch):
    directory = tmp_path / "same-run"
    config = tiny_config(directory)
    seen = []
    export = Trainer.export_actor
    def published(trainer, path):
        seen.append(trainer.global_optimizer_step)
        return export(trainer, path)
    monkeypatch.setattr(Trainer, "export_actor", published)
    assert pipeline(config, enable_monitor=False, stop_after_publications=1)["step"] == 2
    seen.clear()
    assert pipeline(replace(config, checkpoint_interval=3), resume=directory / "learner.pt",
                    enable_monitor=False, stop_after_publications=2)["step"] == 6
    assert seen == [2, 3, 6]  # Startup re-export does not consume the requested two publications.


def test_pipeline_changes_horizon_and_continues_in_same_run(tmp_path):
    directory = tmp_path / "same-run"
    config = tiny_config(directory)
    pipeline(config, enable_monitor=False, stop_after_publications=1)
    old = load(directory / "learner.pt")
    assert pipeline(replace(config, max_game_ply=3), resume=directory / "learner.pt",
                    enable_monitor=False, stop_after_publications=1)["step"] == 4
    saved = load(directory / "learner.pt")
    assert saved["run_id"] == old["run_id"] and saved["config"]["max_game_ply"] == 3
    sessions = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    starts = [row for row in sessions if row.get("event") == "session_start"]
    assert starts[-1]["config_changes"]["max_game_ply"] == {"old": 4, "new": 3}


@pytest.mark.parametrize("count", [0, -1, True])
def test_invalid_publication_budget_rejected_before_creating_run(tmp_path, count):
    directory = tmp_path / "invalid"
    with pytest.raises(ValueError, match="positive integer"):
        pipeline(tiny_config(directory), stop_after_publications=count)
    assert not directory.exists()
