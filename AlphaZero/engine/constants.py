"""棋子编码常量 — 中国象棋 AlphaZero 引擎"""

ROWS = 10
COLS = 9

# ── 棋子类型 (绝对值) ──
KING = 1       # 將/帥
ADVISOR = 2    # 士/仕
ELEPHANT = 3   # 象/相
KNIGHT = 4     # 馬
ROOK = 5       # 車
CANNON = 6     # 砲/炮
PAWN = 7       # 卒/兵

# ── 颜色符号 ──
RED = 1        # 正数 = 红方
BLACK = -1     # 负数 = 黑方

# ── 初始布局: (row, col, piece_type, color_sign) ──
INIT_LAYOUT = [
    # 黑方 (row 0-4)
    (0, 0, ROOK,    BLACK), (0, 1, KNIGHT, BLACK), (0, 2, ELEPHANT, BLACK),
    (0, 3, ADVISOR, BLACK), (0, 4, KING,    BLACK),
    (0, 5, ADVISOR, BLACK), (0, 6, ELEPHANT, BLACK),
    (0, 7, KNIGHT,  BLACK), (0, 8, ROOK,    BLACK),
    (2, 1, CANNON,  BLACK), (2, 7, CANNON,  BLACK),
    (3, 0, PAWN,    BLACK), (3, 2, PAWN,    BLACK), (3, 4, PAWN, BLACK),
    (3, 6, PAWN,    BLACK), (3, 8, PAWN,    BLACK),
    # 红方 (row 5-9)
    (9, 0, ROOK,    RED), (9, 1, KNIGHT, RED), (9, 2, ELEPHANT, RED),
    (9, 3, ADVISOR, RED), (9, 4, KING,    RED),
    (9, 5, ADVISOR, RED), (9, 6, ELEPHANT, RED),
    (9, 7, KNIGHT,  RED), (9, 8, ROOK,    RED),
    (7, 1, CANNON,  RED), (7, 7, CANNON,  RED),
    (6, 0, PAWN,    RED), (6, 2, PAWN,    RED), (6, 4, PAWN, RED),
    (6, 6, PAWN,    RED), (6, 8, PAWN,    RED),
]
