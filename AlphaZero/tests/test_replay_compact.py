"""Compact replay keeps a byte budget and round-trips without pickle."""
import numpy as np
import pytest
from AlphaZero.engine.constants import POLICY_SIZE
from AlphaZero.engine.state import GameState
from AlphaZero.train.replay import ReplayBuffer, pack_sample


def test_compact_sample_roundtrip_and_budget(tmp_path):
    state = GameState.new_game()
    actions = state.legal_actions()
    policy = np.zeros(POLICY_SIZE, dtype=np.float32)
    policy[actions[:2]] = [0.25, 0.75]
    sample = pack_sample(state, policy, np.array([1, 0, 0]), 2.0)
    assert sample.nbytes < 512
    replay = ReplayBuffer(max_size=100, max_bytes=sample.nbytes * 2)
    for _ in range(3):
        replay.add_packed(sample)
    assert len(replay) == 2
    assert replay.byte_size <= replay.max_bytes
    states, policies, wdls, weights = replay.sample(2)
    np.testing.assert_array_equal(states[0], state.encode())
    np.testing.assert_allclose(policies[0, actions[:2]], [0.25, 0.75], atol=1e-4)
    np.testing.assert_array_equal(wdls[0], [1, 0, 0])
    assert np.all(weights == 2.0)

    path = tmp_path / "replay.npz"
    replay.save(path)
    restored = ReplayBuffer.load(path, max_size=100, max_bytes=sample.nbytes * 2)
    assert len(restored) == 2
    assert restored.byte_size == replay.byte_size
    np.testing.assert_array_equal(restored.sample(1)[0][0], state.encode())


def test_empty_replay_can_be_saved_and_loaded(tmp_path):
    path = tmp_path / "empty.npz"
    ReplayBuffer().save(path)
    assert len(ReplayBuffer.load(path)) == 0
    with pytest.raises(ValueError, match="experiment mode"):
        ReplayBuffer.load(path, experiment_mode='heuristic')
