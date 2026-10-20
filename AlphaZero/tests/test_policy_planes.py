"""Independent geometry, coordinate, network and compatibility contracts."""
import hashlib
import json
from queue import Queue
import sqlite3

import numpy as np
import pytest
import torch

from AlphaZero.engine import GameState, Move, ActionEncoder, BLACK
from AlphaZero.engine.constants import POLICY_PLANES, POLICY_SIZE
from AlphaZero.engine.move import ACTION_DELTAS
from AlphaZero.model.network import PolicyValueNetwork
from AlphaZero.model.neural_eval import NeuralEvaluator
from AlphaZero.model.player import ModelPlayer
from AlphaZero.train.config import AlphaZeroConfig, ACTION_VERSION
from AlphaZero.train.inference_server import BatchResponse
from AlphaZero.train.self_play_worker import BatchRemoteEvaluator
from AlphaZero.train.replay import ReplayBuffer, VERSION, pack_sample
from AlphaZero.train.trainer import Trainer


def rotated(move):
    return Move(9 - move.from_row, 8 - move.from_col, 9 - move.to_row, 8 - move.to_col)


def test_geometry_covers_all_and_only_xiangqi_displacements_without_collisions():
    assert POLICY_PLANES == 50 and POLICY_SIZE == 4500
    assert len(ACTION_DELTAS) == len(set(ACTION_DELTAS)) == 50
    encoded = set()
    for origin in range(90):
        fr, fc = divmod(origin, 9)
        for target in range(90):
            tr, tc = divmod(target, 9)
            dr, dc = abs(tr - fr), abs(tc - fc)
            supported = ((dr == 0) != (dc == 0) or (dr, dc) in ((1, 2), (2, 1), (2, 2), (1, 1)))
            move = Move(fr, fc, tr, tc)
            action = ActionEncoder.encode(move)
            if supported:
                assert 0 <= action < 4500 and action not in encoded
                encoded.add(action)
                assert ActionEncoder.decode(action) == move
            else:
                assert action == -1
    for action in range(4500):
        assert (ActionEncoder.decode(action) is not None) == (action in encoded)


@pytest.mark.parametrize("move,plane", [
    (Move(9, 4, 0, 4), 8), (Move(0, 4, 9, 4), 17),
    (Move(5, 8, 5, 0), 25), (Move(5, 0, 5, 8), 33),
    (Move(9, 1, 7, 0), 34), (Move(0, 1, 2, 2), 41),
    (Move(9, 2, 7, 0), 42), (Move(0, 2, 2, 4), 45),
    (Move(9, 3, 8, 4), 47), (Move(0, 3, 1, 4), 49),
    (Move(6, 0, 5, 0), 0), (Move(3, 0, 4, 0), 9),
])
def test_versioned_plane_order_and_origin_layout(move, plane):
    assert ActionEncoder.encode(move) == plane * 90 + move.from_row * 9 + move.from_col


def test_all_slots_rotate_origins_and_directions_and_rotation_is_an_involution():
    seen = set()
    for action in range(4500):
        other = ActionEncoder.rotate_action(action)
        seen.add(other)
        assert ActionEncoder.rotate_action(other) == action
        move = ActionEncoder.decode(action)
        assert ActionEncoder.decode(other) == (rotated(move) if move else None)
    assert len(seen) == 4500
    policy = np.arange(9000, dtype=np.float32).reshape(2, 4500)
    transformed = ActionEncoder.rotate_policy(policy)
    np.testing.assert_array_equal(ActionEncoder.rotate_policy(transformed), policy)
    assert not np.array_equal(transformed, policy[:, ::-1])


@pytest.mark.parametrize("index", [-1, 4500, True, 1.5, np.nan])
def test_invalid_indices_are_not_decoded(index):
    assert ActionEncoder.decode(index) is None
    assert ActionEncoder.rotate_action(index) == -1


def test_legal_masks_are_lossless_on_reachable_positions_and_both_colors():
    rng = np.random.default_rng(732)
    colors = set()
    for _ in range(4):
        state = GameState.new_game()
        for _ in range(60):
            if state.is_terminal():
                break
            colors.add(state.current_player)
            moves, actions = state.legal_moves(), state.legal_actions()
            assert len(set(actions)) == len(moves) == int(state.legal_mask().sum())
            assert {ActionEncoder.decode(action) for action in actions} == set(moves)
            expected = ActionEncoder.legal_mask(moves if state.turn else [rotated(move) for move in moves])
            np.testing.assert_array_equal(state.canonical_mask(), expected)
            state = state.apply(moves[int(rng.integers(len(moves)))])
    assert colors == {-1, 1}


@pytest.mark.parametrize("channels_last", [False, True])
def test_policy_head_is_two_spatial_convolutions_and_logits_use_plane_major_order(channels_last):
    model = PolicyValueNetwork(num_blocks=1, num_filters=4).eval()
    convs = [layer for layer in model.policy_head if isinstance(layer, torch.nn.Conv2d)]
    assert [(layer.in_channels, layer.out_channels, layer.kernel_size) for layer in convs] == [
        (4, 4, (3, 3)), (4, 50, (3, 3))]
    assert not any(isinstance(layer, torch.nn.Linear) for layer in model.policy_head.modules())
    captured = []
    handle = model.policy_head.register_forward_hook(lambda module, inputs, output: captured.append(output.clone()))
    x = torch.randn(2, 131, 10, 9)
    if channels_last:
        model.to(memory_format=torch.channels_last)
        x = x.contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        logits, value = model(x)
    handle.remove()
    assert logits.shape == (2, 4500) and value.shape == (2,)
    assert captured[0].shape == (2, 50, 10, 9)
    assert torch.equal(logits.reshape(2, 50, 10, 9), captured[0])
    with torch.no_grad():
        convs[-1].weight.zero_()
        convs[-1].bias.fill_(-2)
        logits, _ = model(x)
    assert torch.equal(logits, torch.full_like(logits, -2))  # No output ReLU/softmax.


def test_black_network_remote_inference_and_replay_share_the_same_action_transform():
    state = GameState(GameState.new_game().board, BLACK)
    move = Move(3, 0, 4, 0)
    absolute_action, canonical_action = 837, 62
    assert ActionEncoder.encode(move) == absolute_action
    assert ActionEncoder.encode(rotated(move)) == canonical_action

    class FixedNetwork(torch.nn.Module):
        def evaluate_batch(self, states, legal_masks, forward_model=None):
            assert legal_masks.shape == (1, 4500) and legal_masks[0, canonical_action]
            policy = np.zeros((1, 4500), np.float32)
            policy[0, canonical_action] = 1
            return policy, None, np.zeros(1, np.float32)

    policy, _ = NeuralEvaluator(FixedNetwork()).evaluate(state)
    assert policy.argmax() == absolute_action and state.legal_mask()[policy.argmax()]
    requests, responses = Queue(), Queue()
    canonical = np.zeros((1, 4500), np.float32)
    canonical[0, canonical_action] = 1
    responses.put(BatchResponse(1, canonical, np.zeros(1), 0, 5))
    remote = BatchRemoteEvaluator(0, requests, responses)
    remote_policy, _ = remote.evaluate(state)
    np.testing.assert_array_equal(remote_policy, policy)
    np.testing.assert_array_equal(requests.get_nowait().masks[0], state.canonical_mask())
    sample = pack_sample(state, policy, -1)
    assert np.frombuffer(sample.actions, dtype="<u2").tolist() == [canonical_action]
    replay = ReplayBuffer()
    try:
        replay.add_game("black", [sample])
        states, policies, values, _ = replay.sample(2)
        assert states.shape == (2, 131, 10, 9) and policies.shape == (2, 4500)
        assert (policies.argmax(1) == canonical_action).all() and (values == -1).all()
        assert state.absolute_policy(policies[0]).argmax() == absolute_action
    finally:
        replay.close()


def test_old_flat_checkpoint_and_replay_are_rejected_at_version_boundary(tmp_path):
    config = AlphaZeroConfig(num_blocks=1, num_filters=4, device="cpu", precision="fp32")
    trainer = Trainer(config)
    path = tmp_path / "flat.pt"
    trainer.save_checkpoint(path)
    payload = torch.load(path, weights_only=False)
    payload["spec"]["action"] = "absolute-from-to-8100-v1"
    del payload["spec"]["policy_head"], payload["spec"]["policy_planes"]
    payload["fingerprint"] = hashlib.sha256(json.dumps(payload["spec"], sort_keys=True).encode()).hexdigest()
    payload["step"] = 100
    torch.save(payload, path)
    with pytest.raises(ValueError, match="specification mismatch"):
        trainer.load_checkpoint(path)
    with pytest.raises(ValueError, match="规格不匹配"):
        ModelPlayer(path)
    from AlphaZero.train.benchmark import checkpoint_agent
    with pytest.raises(ValueError, match="规格不匹配"):
        checkpoint_agent(str(path))
    replay_path = tmp_path / "old.sqlite"
    ReplayBuffer(replay_path).close()
    with sqlite3.connect(replay_path) as db:
        db.execute("UPDATE metadata SET value=? WHERE key='version'",
                   (VERSION.replace(ACTION_VERSION, "absolute-from-to-8100-v1"),))
    with pytest.raises(ValueError, match="incompatible replay"):
        ReplayBuffer(replay_path)
    from AlphaZero.scripts.migrate_natural_limit import migrate, OLD_RULE
    source = tmp_path / "flat-v2-run"
    source.mkdir()
    payload["spec"]["rule"] = OLD_RULE
    for key in ("natural_limit_rounds", "natural_limit_check_cap", "natural_limit_claim"):
        del payload["spec"][key]
    payload["fingerprint"] = hashlib.sha256(json.dumps(payload["spec"], sort_keys=True).encode()).hexdigest()
    torch.save(payload, source / "learner.pt")
    with pytest.raises(ValueError, match="unchanged architecture"):
        migrate(source, tmp_path / "invalid-migration")
    assert not (tmp_path / "invalid-migration").exists()


def test_padding_targets_and_wrong_size_search_priors_are_rejected():
    state = GameState.new_game()
    pi = np.zeros(4500, np.float32)
    pi[0] = 1  # Upward step from (0,0) leaves the board.
    assert ActionEncoder.decode(0) is None
    with pytest.raises(ValueError, match="off-board"):
        pack_sample(state, pi, 0)
    from AlphaZero.search import MCTS
    class FlatEvaluator:
        def evaluate(self, state):
            return np.zeros(8100, np.float32), 0
    with pytest.raises(ValueError, match="shape"):
        MCTS(FlatEvaluator(), num_simulations=1).search(state)


def test_real_800_search_selfplay_training_checkpoint_and_black_play(tmp_path):
    from AlphaZero.scripts.pipeline import pipeline
    from AlphaZero.train.benchmark import checkpoint_agent
    config = AlphaZeroConfig(num_blocks=1, num_filters=8, batch_size=2,
                             device="cpu", precision="fp32", num_simulations=800,
                             num_workers=2, actor_processes=1, inference_batch_size=4,
                             max_game_ply=4, checkpoint_interval=2, recovery_interval=1,
                             max_optimizer_steps=2, max_iterations=1)
    directory = tmp_path / "planes"
    result = pipeline(config, output_dir=directory, enable_monitor=False)
    assert result["step"] == 2 and result["games"] >= 1
    replay = ReplayBuffer(directory / "replay.sqlite")
    try:
        _, targets, _, _ = replay.sample(2)
        assert targets.shape == (2, 4500)
        assert all(json.loads(row[0])["num_simulations"] == 800
                   for row in replay.db.execute("SELECT trace FROM games"))
    finally:
        replay.close()
    player = ModelPlayer(directory / "latest.pt", simulations=16)
    black = GameState(GameState.new_game().board, BLACK)
    move, diagnostics = player.choose(black)
    assert black.is_legal(move) and diagnostics["simulations"] == 16
    agent = checkpoint_agent(str(directory / "latest.pt"), simulations=4)
    assert black.is_legal(agent(black, np.random.default_rng(5)))
