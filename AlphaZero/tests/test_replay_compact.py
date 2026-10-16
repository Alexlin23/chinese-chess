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
