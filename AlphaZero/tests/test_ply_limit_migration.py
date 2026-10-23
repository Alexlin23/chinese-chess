"""Horizon changes preserve learner recovery and convert retained observations."""
import hashlib
import json
import sqlite3

import numpy as np
import pytest
import torch

from AlphaZero.engine import GameState, Move
from AlphaZero.scripts.migrate_ply_limit import migrate
from AlphaZero.train.config import AlphaZeroConfig
from AlphaZero.train.replay import ReplayBuffer, pack_sample
from AlphaZero.train.trainer import Trainer


@pytest.fixture
def source_run(tmp_path):
    source = tmp_path / "source"
    config = AlphaZeroConfig(num_blocks=1, num_filters=4, device="cpu", precision="fp32",
                             batch_size=2, max_game_ply=12, checkpoint_dir=str(source))
    replay = ReplayBuffer(source / "replay.sqlite")
    replay.bind_spec(config.fingerprint)
    short = [[9,1,7,2], [0,1,2,2], [7,2,9,1], [2,2,0,1]] * 2
    long = [[6,c,5,c] if side == 0 else [3,c,4,c] for c in range(0,9,2) for side in range(2)]
    long += [[9,1,7,2], [0,1,2,2]]
    for sequence, moves in enumerate((short, long)):
        state, samples = GameState.new_game(12), []
        for move in moves:
            mask = state.legal_mask()
            samples.append(pack_sample(state, mask / mask.sum(), 0))
            state = state.apply(Move(*move))
        result = state.game_result()
        assert result is not None and result.winner is None
        replay.add_game(str(sequence), samples, sequence=sequence, reason=result.reason,
                        trace=json.dumps(dict(moves=moves)))
    trainer = Trainer(config)
    trainer.run_id = replay.run_id
    trainer.step(replay)
    trainer.save_checkpoint(source / "learner.pt", replay.recovery_marker())
    replay.close()
    return source


def test_horizon_migration_preserves_sgd_and_transforms_only_retained_globals(source_run, tmp_path):
    digest = hashlib.sha256((source_run / "learner.pt").read_bytes()).hexdigest()
    destination = tmp_path / "target"
    report = migrate(source_run, destination, 10)
    assert (report["kept_games"], report["kept_positions"], len(report["rejected"])) == (1, 8, 1)
    assert report["rejected"][0]["positions"] == 12
    original = torch.load(source_run / "learner.pt", map_location="cpu", weights_only=False)
    payload = torch.load(destination / "learner.pt", map_location="cpu", weights_only=False)
    assert payload["step"] == original["step"] == 1
    assert payload["run_id"] != original["run_id"]
    assert payload["fingerprint"] != original["fingerprint"]
    assert payload["config"]["max_game_ply"] == 10
    for key, value in original["model_state_dict"].items():
        assert torch.equal(value, payload["model_state_dict"][key])
    for key, state in original["optimizer_state_dict"]["state"].items():
        assert torch.equal(state["momentum_buffer"], payload["optimizer_state_dict"]["state"][key]["momentum_buffer"])
    assert torch.equal(original["torch_rng_state"], payload["torch_rng_state"])
    assert hashlib.sha256((source_run / "learner.pt").read_bytes()).hexdigest() == digest
    replay = ReplayBuffer(destination / "replay.sqlite")
    try:
        globals_ = [np.frombuffer(row[0], dtype="<f4")[1] for row in replay.db.execute("SELECT globals FROM positions ORDER BY id")]
        np.testing.assert_array_equal(globals_, np.arange(8, dtype=np.float32) / 10)
        assert replay.next_sequences() == {0: 2}
        trace = json.loads(replay.db.execute("SELECT trace FROM games").fetchone()[0])
        assert trace["ply_limit_migration"]["policy_search_max_game_ply"] == 12
        trainer = Trainer(AlphaZeroConfig(**payload["config"]))
        trainer.load_checkpoint(destination / "learner.pt")
        assert np.isfinite(trainer.step(replay)["loss"])
        assert trainer.global_optimizer_step == 2
    finally:
        replay.close()


def test_corrupt_source_observation_does_not_publish_migration(source_run, tmp_path):
    with sqlite3.connect(source_run / "replay.sqlite") as db:
        db.execute("UPDATE positions SET globals=? WHERE id=1", (np.ones(3, dtype="<f4").tobytes(),))
    destination = tmp_path / "bad"
    with pytest.raises(ValueError, match="observation"):
        migrate(source_run, destination, 10)
    assert not destination.exists()


def test_300_ply_draw_boundary_and_terminal_priority():
    assert not GameState(GameState.new_game().board, True, move_count=299, max_ply=300).is_terminal()
    result = GameState(GameState.new_game().board, True, move_count=300, max_ply=300).game_result()
    assert result.winner is None and result.reason == "max_ply"
    board = GameState.new_game().board.copy()
    board[board == -1] = 0
    result = GameState(board, True, move_count=300, max_ply=300).game_result()
    assert result.winner == 1 and result.reason == "king_captured"
