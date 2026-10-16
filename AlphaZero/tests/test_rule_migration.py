"""Exercise explicit rule migration with a real SGD checkpoint and replay."""
import hashlib
import json
import sqlite3

import numpy as np
import pytest
import torch

from AlphaZero.engine import GameState, Move
from AlphaZero.scripts.migrate_natural_limit import migrate, OLD_RULE
from AlphaZero.train.config import AlphaZeroConfig, RULE_VERSION
from AlphaZero.train.replay import ReplayBuffer, VERSION, pack_sample
from AlphaZero.train.trainer import Trainer


@pytest.fixture
def old_run(tmp_path):
    source = tmp_path / "old"
    source.mkdir()
    config = AlphaZeroConfig(num_blocks=1, num_filters=4, device="cpu", precision="fp32",
                             batch_size=2, max_game_ply=4, checkpoint_dir=str(source))
    replay = ReplayBuffer(source / "replay.sqlite")
    replay.bind_spec(config.fingerprint)
    state, samples = GameState.new_game(4), []
    moves = [[9, 1, 7, 2], [0, 1, 2, 2], [7, 2, 9, 1], [2, 2, 0, 1]]
    for move in moves:
        samples.append(pack_sample(state, state.legal_mask() / state.legal_mask().sum(), 0))
        state = state.apply(Move(*move))
    assert state.game_result().reason == "max_ply"
    replay.add_game("w0-g2", samples, worker=0, sequence=2, reason="max_ply",
                    trace=json.dumps(dict(moves=moves, num_simulations=800)))
    trainer = Trainer(config)
    trainer.run_id = replay.run_id
    trainer.step(replay)
    trainer.save_checkpoint(source / "learner.pt", replay.recovery_marker())
    replay.close()
    payload = torch.load(source / "learner.pt", map_location="cpu", weights_only=False)
    payload["spec"]["rule"] = OLD_RULE
    for key in ("natural_limit_rounds", "natural_limit_check_cap", "natural_limit_claim"):
        del payload["spec"][key]
    payload["fingerprint"] = hashlib.sha256(json.dumps(payload["spec"], sort_keys=True).encode()).hexdigest()
    torch.save(payload, source / "learner.pt")
    with sqlite3.connect(source / "replay.sqlite") as db:
        db.execute("UPDATE metadata SET value=? WHERE key='version'", (VERSION.replace(RULE_VERSION, OLD_RULE),))
        db.execute("UPDATE metadata SET value=? WHERE key='fingerprint'", (payload["fingerprint"],))
    return source, config, payload


def test_migration_preserves_training_and_source_and_can_take_next_sgd_step(old_run, tmp_path):
    source, config, original = old_run
    digest = hashlib.sha256((source / "learner.pt").read_bytes()).hexdigest()
    destination = tmp_path / "new"
    report = migrate(source, destination)
    assert report["kept_games"] == 1 and report["kept_positions"] == 4
    assert not report["rejected"]
    assert hashlib.sha256((source / "learner.pt").read_bytes()).hexdigest() == digest
    payload = torch.load(destination / "learner.pt", map_location="cpu", weights_only=False)
    assert payload["step"] == original["step"] == 1
    assert payload["run_id"] != original["run_id"]
    for key, value in original["model_state_dict"].items():
        assert torch.equal(value, payload["model_state_dict"][key])
    for key, state in original["optimizer_state_dict"]["state"].items():
        assert torch.equal(state["momentum_buffer"], payload["optimizer_state_dict"]["state"][key]["momentum_buffer"])
    assert torch.equal(original["torch_rng_state"], payload["torch_rng_state"])
    trainer = Trainer(config)
    trainer.load_checkpoint(destination / "learner.pt")
    replay = ReplayBuffer(destination / "replay.sqlite")
    try:
        trace = json.loads(replay.db.execute("SELECT trace FROM games").fetchone()[0])
        assert trace["rule_migration"]["policy_search_rule"] == OLD_RULE
        assert replay.next_sequences() == {0: 3}
        assert np.isfinite(trainer.step(replay)["loss"])
        assert trainer.global_optimizer_step == 2
    finally:
        replay.close()


@pytest.mark.parametrize("corruption", ["observation", "identity"])
def test_migration_rejects_corruption_without_publishing_destination(old_run, tmp_path, corruption):
    source, _, _ = old_run
    with sqlite3.connect(source / "replay.sqlite") as db:
        if corruption == "observation":
            db.execute("UPDATE positions SET globals=? WHERE id=1", (np.ones(3, dtype="<f4").tobytes(),))
        else:
            db.execute("UPDATE metadata SET value='wrong-run' WHERE key='run_id'")
    destination = tmp_path / "invalid"
    with pytest.raises(ValueError, match="observation|identity"):
        migrate(source, destination)
    assert not destination.exists()


def test_migration_requires_stopped_source(old_run, tmp_path):
    import fcntl
    source, _, _ = old_run
    with (source / ".run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="stop the source"):
            migrate(source, tmp_path / "locked")


def test_changed_terminal_is_excluded_and_sequence_is_preserved(old_run, tmp_path):
    source, _, _ = old_run
    with sqlite3.connect(source / "replay.sqlite") as db:
        db.execute("UPDATE games SET reason='no_capture'")
    destination = tmp_path / "filtered"
    report = migrate(source, destination)
    assert report["kept_games"] == 0 and len(report["rejected"]) == 1
    replay = ReplayBuffer(destination / "replay.sqlite")
    try:
        assert replay.next_sequences() == {0: 3}
        assert len(replay) == 0
    finally:
        replay.close()
