"""Compare both existing Xiangqi move generators on reachable positions."""
import numpy as np

from AlphaZero.engine.state import GameState
from backend import chess_rules


RED_TYPES = {1: "帥", 2: "仕", 3: "相", 4: "馬",
             5: "車", 6: "炮", 7: "兵"}
BLACK_TYPES = {1: "將", 2: "士", 3: "象", 4: "馬",
               5: "車", 6: "砲", 7: "卒"}


def backend_board(board):
    return [[None if piece == 0 else {
        "type": (RED_TYPES if piece > 0 else BLACK_TYPES)[abs(int(piece))],
        "color": "r" if piece > 0 else "b",
    } for piece in row] for row in board]


def backend_moves(state):
    board = backend_board(state.board)
    turn = "r" if state.turn else "b"
    result = set()
    for row in range(10):
        for col in range(9):
            piece = board[row][col]
            if piece is None or piece["color"] != turn:
                continue
            for target in chess_rules.get_valid_moves(board, row, col, turn):
                result.add((row, col, target["row"], target["col"]))
    return result


def test_reachable_positions_match_web_rules():
    rng = np.random.default_rng(42)
    state = GameState.new_game()
    for _ in range(20):
        if state.is_terminal():
            break
        engine = {(m.from_row, m.from_col, m.to_row, m.to_col)
                  for m in state.legal_moves()}
        assert engine == backend_moves(state)
        moves = state.legal_moves()
        state = state.apply(moves[int(rng.integers(len(moves)))])
