"""Web piece names at the boundary of the shared, history-aware engine."""
import numpy as np
from AlphaZero.engine import GameState, RED

RED_NAMES = ("", "帥", "仕", "相", "馬", "車", "炮", "兵")
BLACK_NAMES = ("", "將", "士", "象", "馬", "車", "砲", "卒")


def numeric_board(board):
    if len(board) != 10 or any(len(row) != 9 for row in board):
        raise ValueError("棋盘必须为10行9列")
    result = np.zeros((10, 9), np.int8)
    for r, row in enumerate(board):
        for c, piece in enumerate(row):
            if piece is None:
                continue
            if hasattr(piece, "model_dump"):
                piece = piece.model_dump()
            color = piece["color"]
            if color not in ("r", "b"):
                raise ValueError("棋子颜色必须为r或b")
            names = RED_NAMES if color == "r" else BLACK_NAMES
            try:
                kind = names.index(piece["type"])
            except ValueError as exc:
                raise ValueError("棋子名称与颜色不匹配") from exc
            if kind == 0:
                raise ValueError("棋子名称不能为空")
            result[r, c] = kind if color == "r" else -kind
    return result


def web_board(state):
    return [[None if p == 0 else dict(type=(RED_NAMES if p > 0 else BLACK_NAMES)[abs(p)],
                                    color="r" if p > 0 else "b")
             for p in row] for row in state.board.tolist()]


def status(state):
    result = state.game_result()
    if result is None:
        return "ongoing"
    return "draw" if result.is_draw else ("red_win" if result.winner == RED else "black_win")


def initial_state(board, turn="r", max_ply=None):
    """Explicit game origin only. Existing games must load their stored history."""
    return GameState(numeric_board(board), turn, max_ply=max_ply)
