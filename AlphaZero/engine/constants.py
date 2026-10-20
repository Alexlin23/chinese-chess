"""中国象棋 AlphaZero 全局常量"""
import numpy as np

# ── 棋盘维度 ──
ROWS: int = 10
COLS: int = 9
BOARD_SIZE: int = ROWS * COLS          # 90

# ── 动作空间 ──
# Plane-major NCHW: action = displacement_plane * 90 + from_square.
POLICY_PLANES: int = 50
POLICY_SIZE: int = POLICY_PLANES * BOARD_SIZE  # 4500, including off-board padding.

# ── 网络维度 ──
INPUT_CHANNELS: int = 131  # 8 * (14 pieces + 2 repetitions) + 3 global planes
WDL_SIZE: int = 3

# ── 游戏规则 ──
MAX_GAME_PLY: int = 512
REPETITION_LIMIT: int = 3
NATURAL_LIMIT_ROUNDS: int = 60
NATURAL_LIMIT_CHECK_CAP: int = 10

# ── 棋子编码 (int8) ──
EMPTY = 0
KING = 1
ADVISOR = 2
ELEPHANT = 3
KNIGHT = 4
ROOK = 5
CANNON = 6
PAWN = 7

# ── 颜色 ──
RED = 1
BLACK = -1

# ── WDL 索引 ──
WDL_WIN = 0
WDL_DRAW = 1
WDL_LOSS = 2

# ── 棋子价值 (启发式评估用) ──
PIECE_VALUES = np.array([0, 0, 2.0, 2.0, 4.0, 9.0, 4.5, 1.0], dtype=np.float32)

# ── 初始布局 (row, col, ptype, color) ──
INIT_LAYOUT = [
    # 黑方 (上方, row 0-4)
    (0, 0, ROOK, BLACK), (0, 1, KNIGHT, BLACK), (0, 2, ELEPHANT, BLACK),
    (0, 3, ADVISOR, BLACK), (0, 4, KING, BLACK), (0, 5, ADVISOR, BLACK),
    (0, 6, ELEPHANT, BLACK), (0, 7, KNIGHT, BLACK), (0, 8, ROOK, BLACK),
    (2, 1, CANNON, BLACK), (2, 7, CANNON, BLACK),
    (3, 0, PAWN, BLACK), (3, 2, PAWN, BLACK), (3, 4, PAWN, BLACK),
    (3, 6, PAWN, BLACK), (3, 8, PAWN, BLACK),
    # 红方 (下方, row 5-9)
    (9, 0, ROOK, RED), (9, 1, KNIGHT, RED), (9, 2, ELEPHANT, RED),
    (9, 3, ADVISOR, RED), (9, 4, KING, RED), (9, 5, ADVISOR, RED),
    (9, 6, ELEPHANT, RED), (9, 7, KNIGHT, RED), (9, 8, ROOK, RED),
    (7, 1, CANNON, RED), (7, 7, CANNON, RED),
    (6, 0, PAWN, RED), (6, 2, PAWN, RED), (6, 4, PAWN, RED),
    (6, 6, PAWN, RED), (6, 8, PAWN, RED),
]
