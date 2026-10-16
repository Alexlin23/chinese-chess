"""CNXA 2020 claimant-specific check allowance and automatic training convention."""
import numpy as np
import pytest
from AlphaZero.engine import GameState, Move, RED, BLACK
from AlphaZero.search import MCTS
from AlphaZero.tests.test_mcts_correctness import ConstantEvaluator


def state_at(plies, checks=(0, 0), turn=RED, board=None, **kwargs):
    if board is None:
        board = GameState.new_game().board
    return GameState(board, turn, move_count=plies, no_capture_count=plies,
                     no_capture_checks=checks, **kwargs)


@pytest.mark.parametrize("plies,checks,effective,draw", [
    (119, (0, 0), 59, False), (120, (0, 0), 60, True),
    (120, (10, 0), 60, True), (120, (11, 0), 59, False),
    (120, (15, 0), 55, False), (124, (13, 0), 59, False),
    (130, (15, 0), 60, True), (120, (0, 15), 60, True),
    (128, (15, 15), 59, False), (130, (15, 15), 60, True),
])
@pytest.mark.parametrize("turn", [RED, BLACK])
def test_claimant_cap_is_per_side_and_counts_rounds_not_plies(plies, checks, effective, draw, turn):
    counts = checks if turn == RED else checks[::-1]
    state = state_at(plies, counts, turn)
    assert state.natural_limit_rounds(turn) == effective
    assert state.natural_limit_status()["eligible"] == draw
    result = state.game_result()
    assert (result is not None) == draw
    if draw:
        assert result.reason == "no_capture" and result.is_draw


def test_extra_check_moves_are_added_once_and_opponent_can_qualify_on_next_turn():
    board = np.zeros((10, 9), np.int8)
    board[9, 3], board[0, 4], board[1, 0] = 1, -1, 5
    state = state_at(118, (10, 0), board=board)
    state = state.apply(Move(1, 0, 1, 4))
    assert state.no_capture_checks == (11, 0)
    state = state.apply(Move(0, 4, 0, 5))
    assert state.no_capture_count == 120
    assert not state.is_terminal()  # Red is the automatic claimant; effective 59.
    state = state.apply(Move(1, 4, 2, 4))
    assert state.no_capture_count == 121
    assert state.current_player == BLACK
    assert state.natural_limit_rounds(BLACK) == 60
    assert state.game_result().reason == "no_capture"


def test_checking_capture_resets_both_sides_without_counting_the_capture_check():
    board = np.zeros((10, 9), np.int8)
    board[9, 3], board[0, 4], board[1, 0], board[1, 4] = 1, -1, 5, -7
    state = state_at(118, (12, 8), board=board)
    next_state = state.apply(Move(1, 0, 1, 4))
    assert next_state.is_in_check()
    assert next_state.no_capture_count == 0
    assert next_state.no_capture_checks == (0, 0)
    assert next_state.natural_limit_rounds(RED) == next_state.natural_limit_rounds(BLACK) == 0
    assert state.no_capture_checks == (12, 8)  # Parent is immutable.


def test_history_derivation_only_counts_the_interval_since_last_capture():
    history = tuple((side, True) for _ in range(30) for side in (RED, BLACK))
    history += ((RED, False), (BLACK, True), (RED, True), (BLACK, False))
    state = GameState(GameState.new_game().board, RED, move_count=64,
                      no_capture_count=4, check_history=history)
    assert state.no_capture_checks == (1, 1)


@pytest.mark.parametrize("checks", [(-1, 0), (61, 0), (0, 61), (True, 0), (1.0, 0), (0,)])
def test_invalid_imported_counters_rejected(checks):
    with pytest.raises(ValueError, match="check counts"):
        state_at(120, checks)


def test_missing_check_history_is_not_assumed_to_mean_zero_checks():
    with pytest.raises(ValueError, match="check history"):
        GameState(GameState.new_game().board, RED, no_capture_count=120)


def test_tree_reuse_distinguishes_same_board_with_different_check_counts():
    before = state_at(100, (10, 10))
    after = state_at(100, (11, 10))
    search = MCTS(ConstantEvaluator(), num_simulations=3, reuse_tree=True)
    search.search(before, add_exploration_noise=False)
    search._reuse_state, search._reuse_root = before, search.last_root
    search.search(after, add_exploration_noise=False)
    assert search.diagnostics["reused_visits"] == 0


def test_old_rule_checkpoint_and_replay_require_explicit_migration(tmp_path):
    import hashlib
    import json
    import sqlite3
    import torch
    from AlphaZero.train.config import AlphaZeroConfig, RULE_VERSION
    from AlphaZero.train.trainer import Trainer
    from AlphaZero.train.replay import ReplayBuffer, VERSION
    config = AlphaZeroConfig(num_blocks=1, num_filters=4, device="cpu", precision="fp32")
    trainer = Trainer(config)
    path = tmp_path / "old.pt"
    trainer.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["spec"]["rule"] = "xiangqi-development-v2"
    for key in ("natural_limit_rounds", "natural_limit_check_cap", "natural_limit_claim"):
        del payload["spec"][key]
    payload["fingerprint"] = hashlib.sha256(json.dumps(payload["spec"], sort_keys=True).encode()).hexdigest()
    torch.save(payload, path)
    with pytest.raises(ValueError, match="specification mismatch"):
        trainer.load_checkpoint(path)
    replay_path = tmp_path / "old.sqlite"
    ReplayBuffer(replay_path).close()
    with sqlite3.connect(replay_path) as conn:
        conn.execute("UPDATE metadata SET value=? WHERE key='version'",
                     (VERSION.replace(RULE_VERSION, "xiangqi-development-v2"),))
    with pytest.raises(ValueError, match="incompatible replay"):
        ReplayBuffer(replay_path)
