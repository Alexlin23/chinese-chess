"""v2 disk replay: complete-game FIFO, schema checks and exact observation targets."""
import numpy as np
import pytest
from AlphaZero.engine import GameState
from AlphaZero.train.replay import ReplayBuffer, pack_sample

def sample_for(state, value=1):
    policy = state.legal_mask().astype(np.float32)
    policy /= policy.sum()
    return pack_sample(state, policy, value)

def test_history_roundtrip_disk_and_uniform_targets(tmp_path):
    state = GameState.new_game()
    state = state.apply(state.legal_moves()[0])
    sample = sample_for(state, -1)
    path = tmp_path / "replay.sqlite"
    replay = ReplayBuffer(path, window_games=2)
    replay.add_game("first", [sample], worker=1, sequence=7)
    x, p, z, weights = replay.sample(3)
    np.testing.assert_array_equal(x[0], state.encode())
    np.testing.assert_allclose(p[0], state.canonical_policy(
        state.legal_mask().astype(np.float32)/state.legal_mask().sum()), atol=1e-7)
    assert np.all(z == -1) and np.all(weights == 1)
    assert not replay.add_game("first", [sample])
    replay.close()
    replay = ReplayBuffer(path, window_games=2)
    assert replay.next_sequences()[1] == 8
    assert len(replay) == 1
    replay.close()

def test_fifo_evicts_complete_games(tmp_path):
    replay = ReplayBuffer(tmp_path/"r.sqlite", window_games=2)
    sample = sample_for(GameState.new_game())
    replay.add_game("a", [sample]*3, worker=0, sequence=0)
    replay.add_game("b", [sample]*2, worker=0, sequence=1)
    replay.add_game("c", [sample]*4, worker=0, sequence=2)
    assert len(replay) == 6 and replay.game_count == 2
    assert replay.next_sequences()[0] == 3
    assert np.isfinite(replay.sample(64)[0]).all()
    replay.close()

def test_reject_negative_and_nonuniform_weight():
    state = GameState.new_game()
    p = state.legal_mask().astype(np.float32)
    p /= p.sum()
    p[0] = -0.1
    with pytest.raises(ValueError):
        pack_sample(state, p, 0)
    with pytest.raises(ValueError):
        pack_sample(state, state.legal_mask()/state.legal_mask().sum(), 0, 2)

def test_empty_replay_and_schema_rejection(tmp_path):
    path = tmp_path/"empty.sqlite"
    replay = ReplayBuffer(path)
    with pytest.raises(ValueError):
        replay.sample(1)
    replay.db.execute("UPDATE metadata SET value='legacy' WHERE key='version'")
    replay.db.commit()
    replay.close()
    with pytest.raises(ValueError, match="incompatible"):
        ReplayBuffer(path)


def test_position_cap_evicts_whole_games_and_preserves_exact_boundary(tmp_path):
    replay = ReplayBuffer(tmp_path/"r.sqlite", window_positions=5)
    sample = sample_for(GameState.new_game())
    replay.add_game("a", [sample]*3)
    replay.add_game("b", [sample]*2)
    assert len(replay) == 5 and replay.game_count == 2
    replay.add_game("c", [sample]*4)
    assert len(replay) == 4 and replay.game_count == 1
    assert replay.db.execute("SELECT episode FROM games").fetchall() == [("c",)]
    replay.add_game("d", [sample])
    assert len(replay) == 5 and replay.game_count == 2
    assert replay.total_positions == 10 and replay.generation == 4
    assert replay.high - replay.low + 1 == len(replay)
    assert np.isfinite(replay.sample(32)[0]).all()
    replay.close()


def test_shrinking_position_window_preserves_resume_metadata_and_newest_targets(tmp_path):
    path = tmp_path/"r.sqlite"
    replay = ReplayBuffer(path)
    state = GameState.new_game()
    for sequence, size in enumerate([3, 4, 5]):
        replay.add_game(str(sequence), [sample_for(state, sequence-1)]*size,
                        worker=7, sequence=sequence, actor_end=sequence*100)
    marker, run_id = replay.recovery_marker(), replay.run_id
    replay.close()
    replay = ReplayBuffer(path, window_positions=8)
    assert len(replay) == 12  # Opening alone must not trim before resume validation.
    assert replay.enforce_limits() == dict(removed_games=2, removed_positions=7,
                                          retained_games=1, retained_positions=5)
    assert replay.recovery_marker() == marker and replay.run_id == run_id
    assert replay.total_positions == 12 and replay.next_sequences() == {7: 3}
    assert replay.max_actor_step == 200
    assert np.all(replay.sample(32)[2] == 1)
    replay.close()
    replay = ReplayBuffer(path, window_games=2, window_positions=8)
    assert replay.enforce_limits()["removed_positions"] == 0
    replay.add_game("next", [sample_for(state)]*2, worker=7, sequence=3)
    assert len(replay) == 7 and replay.generation == 4
    replay.add_game("last", [sample_for(state)]*2, worker=7, sequence=4)
    assert len(replay) == 4 and replay.game_count == 2
    assert replay.high - replay.low + 1 == len(replay)
    assert not replay.db.execute("PRAGMA foreign_key_check").fetchall()
    replay.close()


def test_oversized_game_rejected_without_removing_existing_data():
    replay = ReplayBuffer(window_positions=3)
    sample = sample_for(GameState.new_game())
    replay.add_game("kept", [sample]*2)
    with pytest.raises(ValueError, match="position cap"):
        replay.add_game("too-long", [sample]*4)
    assert len(replay) == 2 and replay.game_count == 1 and replay.generation == 1
    replay.close()
