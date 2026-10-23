"""Data-gated learning must survive FIFO, legacy checkpoints and resumed updates."""
from dataclasses import replace
import importlib
import json
import sqlite3

import numpy as np
import pytest
import torch

from AlphaZero.engine import GameState
from AlphaZero.scripts.pipeline import pipeline
from AlphaZero.tests.test_search_budget_resume import tiny_config
from AlphaZero.train.config import AlphaZeroConfig
from AlphaZero.train.replay import ReplayBuffer, pack_sample
from AlphaZero.train.trainer import Trainer
from AlphaZero.train.update_schedule import UpdateSchedule


def test_fifo_duplicate_and_mirror_do_not_create_or_remove_update_credit(tmp_path):
    replay = ReplayBuffer(tmp_path / "replay.sqlite", window_positions=2)
    try:
        state = GameState.new_game()
        policy = state.legal_mask().astype(np.float32)
        policy /= policy.sum()
        sample = pack_sample(state, policy, 0)
        gate = UpdateSchedule(4, replay.total_positions)
        replay.add_game("a", [sample] * 2)
        assert not gate.ready(replay.total_positions)
        replay.sample(16, mirror_augmentation=True)
        assert not replay.add_game("a", [sample] * 2)
        assert replay.total_positions == 2
        replay.add_game("b", [sample] * 2)
        assert len(replay) == 2 and replay.total_positions == 4
        gate.record_update(replay.total_positions)
        assert not gate.ready(replay.total_positions)
    finally:
        replay.close()


def test_restore_preserves_remainder_and_post_checkpoint_committed_games():
    gate = UpdateSchedule(4096, 90000)
    assert not gate.ready(94095)
    with pytest.raises(ValueError, match="not enough"):
        gate.record_update(94095)
    gate.record_update(95000)
    saved = gate.state_dict()
    restored = UpdateSchedule(4096, 99000, saved)
    assert restored.ready(99000)  # Includes 4000 positions committed after the checkpoint.
    restored.record_update(99000)
    assert restored.consumed_positions == 98192
    assert not restored.ready(99000)
    # A changed interval deliberately starts a new baseline; no historical windfall.
    changed = UpdateSchedule(512, 99000, saved)
    assert not changed.ready(99000)
    assert changed.ready(99512)
    with pytest.raises(ValueError, match="counter mismatch"):
        UpdateSchedule(4096, 90000, saved)


@pytest.mark.parametrize("invalid", [-1, True, 1.5, "4096"])
def test_invalid_config_interval(invalid):
    with pytest.raises(ValueError, match="new_positions_per_update"):
        AlphaZeroConfig(new_positions_per_update=invalid)


def test_disabled_does_not_limit_updates_or_consume_rng():
    state = np.random.get_state()
    gate = UpdateSchedule(0, 50000)
    for _ in range(10):
        gate.record_update(50000)
    np.testing.assert_array_equal(state[1], np.random.get_state()[1])
    assert state[2:] == np.random.get_state()[2:]


def test_cpu_pipeline_legacy_enablement_and_resume_credit(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    config = tiny_config(num_simulations=1)
    pipeline(config, output_dir=tmp_path, enable_monitor=False)
    path = tmp_path / "learner.pt"
    old = torch.load(path, map_location="cpu", weights_only=False)
    old["config"].pop("new_positions_per_update")
    old["extra"].pop("update_schedule", None)
    torch.save(old, path)
    with sqlite3.connect(tmp_path / "replay.sqlite") as db:
        baseline = int(db.execute("SELECT value FROM metadata WHERE key='total_positions'").fetchone()[0])
    seen = []
    original_step = Trainer.step
    def checked_step(trainer, replay):
        assert replay.total_positions >= baseline + 5 * (len(seen) + 1)
        seen.append(replay.total_positions)
        return original_step(trainer, replay)
    monkeypatch.setattr(Trainer, "step", checked_step)
    enabled = replace(config, new_positions_per_update=5, max_optimizer_steps=4, max_iterations=2)
    assert pipeline(enabled, resume=path, output_dir=tmp_path, enable_monitor=False)["step"] == 4
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["extra"]["update_schedule"] == dict(interval=5, consumed_positions=baseline + 10)
    assert saved["run_id"] == old["run_id"]
    actor = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=False)
    assert "new_positions_per_update" not in actor["config"]
    assert pipeline(replace(enabled, max_optimizer_steps=6, max_iterations=3), resume=path,
                    output_dir=tmp_path, enable_monitor=False)["step"] == 6
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["extra"]["update_schedule"]["consumed_positions"] == baseline + 20
    sessions = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
                if '"session_start"' in line]
    assert sessions[1]["config_changes"]["new_positions_per_update"] == {"old": 0, "new": 5}
    assert sessions[2]["config_changes"] == {
        "max_optimizer_steps": {"old": 4, "new": 6},
        "max_iterations": {"old": 2, "new": 3}}


def test_worker_failure_reaches_learner_while_waiting_for_data(tmp_path, monkeypatch):
    from AlphaZero.tests.test_worker_failures import _failed_group
    config = tiny_config(num_simulations=1)
    pipeline(config, output_dir=tmp_path, enable_monitor=False)
    module = importlib.import_module("AlphaZero.scripts.pipeline")
    monkeypatch.setattr(module, "self_play_group_fn", _failed_group)
    with pytest.raises(RuntimeError, match="sentinel inference timeout"):
        pipeline(replace(config, new_positions_per_update=4096, max_optimizer_steps=4,
                         max_iterations=2), resume=tmp_path / "learner.pt",
                 output_dir=tmp_path, enable_monitor=False)
