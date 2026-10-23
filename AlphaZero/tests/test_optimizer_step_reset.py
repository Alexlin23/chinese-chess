"""Rebasing the SGD schedule must not roll back weights, RNG, data or model versions."""
from dataclasses import replace
import hashlib

import numpy as np
import pytest
import torch

from AlphaZero.model.player import ModelPlayer
from AlphaZero.scripts.pipeline import pipeline, run_lock
from AlphaZero.scripts.reset_optimizer_step import reset_optimizer_step
from AlphaZero.tests.test_search_budget_resume import tiny_config
from AlphaZero.train.resume import prepare_resume
from AlphaZero.train.trainer import Trainer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reset_then_resume_preserves_state_and_monotonic_model_version(tmp_path):
    torch.set_num_threads(1)
    config = replace(tiny_config(num_simulations=1), lr_milestones=(2,),
                     max_optimizer_steps=4, max_iterations=2)
    pipeline(config, output_dir=tmp_path, enable_monitor=False)
    path = tmp_path / "learner.pt"
    old = torch.load(path, map_location="cpu", weights_only=False)
    unchanged = {name: digest(tmp_path / name) for name in ("latest.pt", "replay.sqlite", "process.pid")}
    record = reset_optimizer_step(tmp_path, 1)
    new = torch.load(path, map_location="cpu", weights_only=False)
    assert new["step"] == 1 and new["optimizer_step_offset"] == 3
    assert new["current_iteration"] == 0 and record["model_step"] == 4
    assert record["learning_rate"] == .2
    assert {name: digest(tmp_path / name) for name in unchanged} == unchanged
    for name in old["model_state_dict"]:
        assert torch.equal(old["model_state_dict"][name], new["model_state_dict"][name])
    for index, state in old["optimizer_state_dict"]["state"].items():
        for key, value in state.items():
            assert torch.equal(value, new["optimizer_state_dict"]["state"][index][key])
    assert new["optimizer_state_dict"]["param_groups"][0]["lr"] == .2
    assert new["python_rng_state"] == old["python_rng_state"]
    assert torch.equal(new["torch_rng_state"], old["torch_rng_state"])
    for a, b in zip(new["numpy_rng_state"], old["numpy_rng_state"]):
        np.testing.assert_array_equal(a, b)
    assert new["run_id"] == old["run_id"]
    assert ModelPlayer(path, device="cpu", simulations=1).step == 4
    result = pipeline(config, resume=path, output_dir=tmp_path, enable_monitor=False,
                      stop_after_publications=1)
    assert result["step"] == 2
    continued = torch.load(path, map_location="cpu", weights_only=False)
    actor = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=False)
    assert continued["step"] == 2 and continued["optimizer_step_offset"] == 3
    assert actor["step"] == 5 and "optimizer_step_offset" not in actor
    assert continued["extra"]["optimizer_step_changes"][-1] == record
    assert ModelPlayer(tmp_path / "latest.pt", device="cpu", simulations=1).step == 5
    # Horizon conversion must compare historical actor versions against the monotonic counter.
    updated = replace(config, max_game_ply=3)
    with run_lock(tmp_path):
        prepare_resume(tmp_path, updated, path)
    converted = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=False)
    assert converted["step"] == 5
    loaded = Trainer(updated, "cpu")
    loaded.load_checkpoint(path)
    assert loaded.global_optimizer_step == 2 and loaded.model_step == 5


@pytest.mark.parametrize("target", [0, -1, True, 1.5])
def test_invalid_target_rejected_before_accessing_run(tmp_path, target):
    with pytest.raises(ValueError, match="positive integer"):
        reset_optimizer_step(tmp_path / "missing", target)


def test_locked_run_cannot_be_modified(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another learner"):
            reset_optimizer_step(tmp_path, 1)
