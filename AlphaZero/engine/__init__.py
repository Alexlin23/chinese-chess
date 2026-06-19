"""AlphaZero 优化规则引擎 — 中国象棋

完全自包含模块，不依赖 backend.chess_rules。
使用 numpy int8 矩阵 + 原地模拟/撤销，专为 MCTS 优化。
"""
from .constants import (ROWS, COLS,
                         KING, ADVISOR, ELEPHANT, KNIGHT, ROOK, CANNON, PAWN,
                         RED, BLACK)
from .fast_chess import (
    create_initial_board,
    get_raw_moves,
    get_valid_moves,
    is_valid_move,
    make_move,
    undo_move,
    is_in_check,
    check_game_result,
    has_any_valid_move,
)
from .move import Move, ActionEncoder
from .state import GameState

__all__ = [
    "ROWS", "COLS",
    "KING", "ADVISOR", "ELEPHANT", "KNIGHT", "ROOK", "CANNON", "PAWN",
    "RED", "BLACK",
    "create_initial_board",
    "get_raw_moves",
    "get_valid_moves",
    "is_valid_move",
    "make_move",
    "undo_move",
    "is_in_check",
    "check_game_result",
    "has_any_valid_move",
    "Move", "ActionEncoder",
    "GameState",
]
