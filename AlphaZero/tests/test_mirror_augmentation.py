"""Reflection must preserve full history, legal policy targets and recovery."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from AlphaZero.engine import GameState, Move
from AlphaZero.engine.move import ActionEncoder
from AlphaZero.model.player import ModelPlayer
from AlphaZero.train.config import AlphaZeroConfig
from AlphaZero.train.replay import ReplayBuffer, _MIRROR_ACTION_IDS, pack_sample
from AlphaZero.train.trainer import Trainer


def reflect_move(move):
    return Move(move.from_row, 8 - move.from_col, move.to_row, 8 - move.to_col)


@pytest.fixture
def positions():
    state, reflected = GameState.new_game(300), GameState.new_game(300)
    rng = np.random.default_rng(20261020)
    pairs = []
    for _ in range(12):
        moves = state.legal_moves()
        move = moves[int(rng.integers(len(moves)))]
        assert reflected.is_legal(reflect_move(move))
        state = state.apply(move)
        reflected = reflected.apply(reflect_move(move))
        assert not state.is_terminal()
        pairs.append((state, reflected))
    return pairs


def policy_for(state):
    actions = np.flatnonzero(state.legal_mask())
    policy = np.zeros(8100, np.float32)
    policy[actions] = np.arange(1, len(actions) + 1, dtype=np.float32)
    return policy / policy.sum()


@pytest.fixture
def replay(positions):
    # Both colours, with a populated history and deliberately asymmetric targets.
    pairs = [positions[8], positions[9]]
    buffer = ReplayBuffer()
    for index, (state, _) in enumerate(pairs):
        buffer.add_game(str(index), [pack_sample(state, policy_for(state), (-.75, .5)[index])])
    yield buffer, pairs
    buffer.close()


def test_all_action_mappings_are_bijective_involutions():
    assert len(np.unique(_MIRROR_ACTION_IDS)) == 8100
    np.testing.assert_array_equal(_MIRROR_ACTION_IDS[_MIRROR_ACTION_IDS], np.arange(8100))
    for action in range(8100):
        assert _MIRROR_ACTION_IDS[action] == ActionEncoder.encode(reflect_move(ActionEncoder.decode(action)))


def test_reflected_legal_trajectory_has_matching_history_and_masks(positions):
    assert {state.current_player for state, _ in positions} == {-1, 1}
    for state, reflected in positions:
        np.testing.assert_array_equal(reflected.encode(), state.encode()[:, :, ::-1])
        np.testing.assert_array_equal(reflected.canonical_mask(), state.canonical_mask()[_MIRROR_ACTION_IDS])
    state, reflected = GameState.new_game(300), GameState.new_game(300)
    cycle = [Move(9,1,7,2), Move(0,1,2,2), Move(7,2,9,1), Move(2,2,0,1)]
    for move in (cycle * 2)[:7]:
        state = state.apply(move)
        reflected = reflected.apply(reflect_move(move))
    assert np.any(state.encode()[14:128:16])  # Nonzero repetition history planes.
    np.testing.assert_array_equal(reflected.encode(), state.encode()[:, :, ::-1])


@pytest.mark.parametrize("factory", [np.random.default_rng, np.random.RandomState])
def test_uniform_virtual_pool_reflects_targets_without_expanding_batch_or_storage(replay, factory):
    buffer, pairs = replay
    before = buffer.db.execute("SELECT * FROM positions ORDER BY id").fetchall()
    expected = factory(71)
    draw = expected.integers if hasattr(expected, "integers") else expected.randint
    ids, flips = draw(1, 3, size=64), draw(0, 2, size=64)
    assert set(flips) == {0, 1}
    x, pi, z, weights = buffer.sample(64, factory(71), mirror_augmentation=True)
    assert x.shape == (64, 131, 10, 9) and pi.shape == (64, 8100)
    assert x.flags.c_contiguous and pi.flags.c_contiguous
    for slot, (sample_id, flip) in enumerate(zip(ids, flips)):
        original, reflected = pairs[sample_id - 1]
        target_state = reflected if flip else original
        target_pi = original.canonical_policy(policy_for(original))
        if flip:
            target_pi = target_pi[_MIRROR_ACTION_IDS]
        np.testing.assert_array_equal(x[slot], target_state.encode())
        np.testing.assert_array_equal(pi[slot], target_pi)
        assert target_state.canonical_mask()[np.flatnonzero(pi[slot])].all()
        assert z[slot] == (-.75, .5)[sample_id - 1] and weights[slot] == 1
    np.testing.assert_allclose(pi.sum(axis=1), 1, atol=1e-6)
    assert len(buffer) == 2
    assert buffer.db.execute("SELECT * FROM positions ORDER BY id").fetchall() == before


@pytest.mark.parametrize("factory", [np.random.default_rng, np.random.RandomState])
def test_disabled_augmentation_preserves_samples_and_rng_consumption(replay, factory):
    buffer, pairs = replay
    rng, expected = factory(57), factory(57)
    draw = expected.integers if hasattr(expected, "integers") else expected.randint
    ids = draw(1, 3, size=32)
    x, pi, _, _ = buffer.sample(32, rng)
    for slot, sample_id in enumerate(ids):
        state = pairs[sample_id - 1][0]
        np.testing.assert_array_equal(x[slot], state.encode())
        np.testing.assert_array_equal(pi[slot], state.canonical_policy(policy_for(state)))
    np.testing.assert_array_equal(rng.random(16), expected.random(16))


def test_checkpoint_restores_augmented_sampling_rng_exactly(replay):
    buffer, _ = replay
    previous = np.random.get_state()
    try:
        np.random.seed(300)
        saved = np.random.get_state()
        expected = buffer.sample(16, mirror_augmentation=True)
        np.random.set_state(saved)
        actual = buffer.sample(16, mirror_augmentation=True)
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)
    finally:
        np.random.set_state(previous)


@pytest.mark.parametrize("bad", [0, 1, None, "true"])
def test_augmentation_flag_requires_boolean(bad):
    with pytest.raises(ValueError, match="mirror_augmentation"):
        AlphaZeroConfig(mirror_augmentation=bad)


def test_legacy_resume_retains_sgd_state_and_publishes_compatible_actor(replay, tmp_path):
    torch.set_num_threads(1)
    buffer, _ = replay
    config = AlphaZeroConfig(num_blocks=1, num_filters=4, batch_size=4, max_game_ply=300,
                             device="cpu", precision="fp32", replay_window_positions=100000)
    original = Trainer(config)
    original.run_id = buffer.run_id
    original.step(buffer)
    checkpoint = tmp_path / "learner.pt"
    original.save_checkpoint(checkpoint, buffer.recovery_marker())
    legacy = torch.load(checkpoint, map_location="cpu", weights_only=False)
    legacy["config"].pop("mirror_augmentation")
    torch.save(legacy, checkpoint)
    updated_config = replace(config, mirror_augmentation=True, replay_window_positions=50000)
    assert updated_config.fingerprint == config.fingerprint
    restored = Trainer(updated_config)
    restored.load_checkpoint(checkpoint)
    assert restored.global_optimizer_step == 1
    assert restored.resume_config_changes == {
        "mirror_augmentation": {"old": False, "new": True},
        "replay_window_positions": {"old": 100000, "new": 50000}}
    for key, value in legacy["model_state_dict"].items():
        assert torch.equal(value, restored.model.state_dict()[key])
    state = restored.optimizer.state_dict()
    for key, value in legacy["optimizer_state_dict"]["state"].items():
        assert torch.equal(value["momentum_buffer"], state["state"][key]["momentum_buffer"])
    np.testing.assert_array_equal(np.random.get_state()[1], legacy["numpy_rng_state"][1])
    calls = []
    sample = buffer.sample
    def record_sample(n, **kwargs):
        calls.append((n, kwargs))
        return sample(n, **kwargs)
    buffer.sample = record_sample
    try:
        assert np.isfinite(restored.step(buffer)["loss"])
        assert calls == [(4, {"mirror_augmentation": True})]
    finally:
        buffer.sample = sample
    actor = tmp_path / "latest.pt"
    restored.export_actor(actor)
    published = torch.load(actor, map_location="cpu", weights_only=False)
    assert "mirror_augmentation" not in published["config"]
    assert "replay_window_positions" not in published["config"]
    assert ModelPlayer(actor, device="cpu", simulations=1).step == 2
    restored.save_checkpoint(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["config"]["mirror_augmentation"] is True
