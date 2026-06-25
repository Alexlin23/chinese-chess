"""中国象棋 AlphaZero 引擎模块"""
from .constants import (
    ROWS, COLS, BOARD_SIZE, POLICY_SIZE,
    INPUT_CHANNELS, WDL_SIZE, MAX_GAME_PLY, REPETITION_LIMIT,
    EMPTY, KING, ADVISOR, ELEPHANT, KNIGHT, ROOK, CANNON, PAWN,
    RED, BLACK, WDL_WIN, WDL_DRAW, WDL_LOSS,
    PIECE_VALUES, INIT_LAYOUT
)
from .move import Move, ActionEncoder
from .state import GameState, GameResult
from .repetition import position_key, update_repetition, is_repetition_draw
from .fast_chess import create_initial_board, get_valid_moves, is_in_check
