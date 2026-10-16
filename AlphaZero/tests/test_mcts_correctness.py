"""Regression tests for value perspective, leaf batching and game outcomes."""

import numpy as np

from AlphaZero.engine.constants import KING, POLICY_SIZE, RED, ROOK
from AlphaZero.engine.move import ActionEncoder, Move
from AlphaZero.engine.state import GameState
from AlphaZero.search.node import MCTSNode
from AlphaZero.search.tree import MCTS


class ConstantEvaluator:
    def __init__(self):
        self.batch_sizes = []

    def evaluate(self, state):
        return np.zeros(POLICY_SIZE, dtype=np.float32), 0.0

    def evaluate_batch(self, states):
        # Transpositions with different histories are different rule states.
        keys = [(s.board.tobytes(), s.turn, s.position_history) for s in states]
        assert len(keys) == len(set(keys)), "same unexpanded leaf evaluated twice"
        self.batch_sizes.append(len(states))
        return (np.zeros((len(states), POLICY_SIZE), dtype=np.float32),
                np.zeros(len(states), dtype=np.float32))


def test_parent_prefers_child_that_is_bad_for_next_player():
    root = MCTSNode(prior=1.0, visit_count=20)
    root.children = {
        1: MCTSNode(prior=0.5, visit_count=10, total_value=-8.0),
        2: MCTSNode(prior=0.5, visit_count=10, total_value=8.0),
    }
    assert root.select_child(c_puct=1.5) == 1


def test_virtual_loss_and_backup_preserve_player_perspectives():
    search = MCTS(ConstantEvaluator(), virtual_loss=3.0)
    root = MCTSNode(prior=1.0)
    child = MCTSNode(prior=1.0, visit_count=3.0, total_value=3.0)
    root.children[1] = child
    # Emulate a reservation, followed by a winning outcome for the child.
    child.visit_count += search.virtual_loss
    child.total_value += search.virtual_loss
    search._backup([(root, 1), (child, -1)], 1.0, root)
    assert child.visit_count == 4.0
    assert child.total_value == 4.0
    assert root.visit_count == 1
    assert root.total_value == -1.0


def test_search_finds_immediate_king_capture_without_duplicate_leaves():
    board = np.zeros((10, 9), dtype=np.int8)
    board[0, 4] = -KING
    board[1, 4] = ROOK
    board[9, 4] = KING
    state = GameState(board, RED)
    evaluator = ConstantEvaluator()
    search = MCTS(evaluator, num_simulations=48, batch_size=8)
    move, policy = search.select_move(
        state, temperature=0.0, add_exploration_noise=False)
    assert move == Move(1, 4, 0, 4)
    assert state.apply(move).game_result().winner == RED
    # Greedy action selection does not turn the training target into one-hot.
    assert policy.argmax() == ActionEncoder.encode(move)
    assert np.isclose(policy.sum(), 1.0)
    assert len(evaluator.batch_sizes) > 0


def test_search_noise_can_be_disabled():
    board = np.zeros((10, 9), dtype=np.int8)
    board[0, 4] = -KING
    board[1, 4] = ROOK
    board[9, 4] = KING
    state = GameState(board, RED)
    search = MCTS(ConstantEvaluator(), num_simulations=16, batch_size=4)
    np.random.seed(1)
    first = search.search(state, temperature=0.0, add_exploration_noise=False)
    np.random.seed(999)
    second = search.search(state, temperature=0.0, add_exploration_noise=False)
    np.testing.assert_array_equal(first, second)
