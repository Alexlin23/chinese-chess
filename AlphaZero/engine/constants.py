"""中国象棋 AlphaZero 全局常量"""
import numpy as np

# ── 棋盘维度 ──
ROWS: int = 10
COLS: int = 9
BOARD_SIZE: int = ROWS * COLS          # 90

# ── 动作空间 ──
# action = from_square * 90 + to_square
POLICY_SIZE: int = BOARD_SIZE * BOARD_SIZE  # 8100

# ── 网络维度 ──
INPUT_CHANNELS: int = 18
WDL_SIZE: int = 3

# ── 游戏规则 ──
MAX_GAME_PLY: int = 300
REPETITION_LIMIT: int = 3

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
