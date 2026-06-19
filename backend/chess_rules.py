"""中国象棋规则引擎 - 棋子走法、合法性校验、胜负判定"""
from typing import Optional

ROWS = 10
COLS = 9

# 初始布局 [col, row, type, color]
INIT_LAYOUT = [
    [0,0,"車","b"],[1,0,"馬","b"],[2,0,"象","b"],[3,0,"士","b"],[4,0,"將","b"],
    [5,0,"士","b"],[6,0,"象","b"],[7,0,"馬","b"],[8,0,"車","b"],
    [1,2,"砲","b"],[7,2,"砲","b"],
    [0,3,"卒","b"],[2,3,"卒","b"],[4,3,"卒","b"],[6,3,"卒","b"],[8,3,"卒","b"],
    [0,9,"車","r"],[1,9,"馬","r"],[2,9,"相","r"],[3,9,"仕","r"],[4,9,"帥","r"],
    [5,9,"仕","r"],[6,9,"相","r"],[7,9,"馬","r"],[8,9,"車","r"],
    [1,7,"炮","r"],[7,7,"炮","r"],
    [0,6,"兵","r"],[2,6,"兵","r"],[4,6,"兵","r"],[6,6,"兵","r"],[8,6,"兵","r"],
]


def create_initial_board() -> list[list[Optional[dict]]]:
    """创建初始棋盘"""
    board = [[None] * COLS for _ in range(ROWS)]
    for c, r, piece_type, color in INIT_LAYOUT:
        # 创建棋子字典，包含类型和颜色两个键
        board[r][c] = {"type": piece_type, "color": color}
    return board


def in_bounds(r: int, c: int) -> bool:
    """检查坐标是否在棋盘范围内"""
    return 0 <= r < ROWS and 0 <= c < COLS


def get_raw_moves(board: list, r: int, c: int) -> list[dict]:
    """获取棋子的原始走法（不含合法性校验）"""
    piece = board[r][c]
    if not piece:
        return []

    ptype = piece["type"]
    color = piece["color"]
    moves = []

    if ptype == "車":
        moves = _moves_rook(board, r, c)
    elif ptype == "馬":
        moves = _moves_knight(board, r, c)
    elif ptype in ("象", "相"):
        moves = _moves_elephant(board, r, c, color)
    elif ptype in ("士", "仕"):
        moves = _moves_advisor(board, r, c, color)
    elif ptype in ("將", "帥"):
        moves = _moves_king(board, r, c, color)
    elif ptype in ("砲", "炮"):
        moves = _moves_cannon(board, r, c)
    elif ptype in ("卒", "兵"):
        moves = _moves_pawn(board, r, c, color)

    # 过滤：不能吃己方棋子
    result = []
    for m in moves:
        target = board[m["row"]][m["col"]]
        # 目标位置为空或敌方棋子时，该走法有效
        if target is None or target["color"] != color:
            # 标记是否为吃子走法
            m["capture"] = target is not None  # 标记是否为吃子走法，用于前端显示吃子提示或音效触发；若无此标记则数组中该字段不存在
            result.append(m)

    return result


def get_valid_moves(board: list, r: int, c: int, turn: str) -> list[dict]:
    """获取合法走法（含将军校验）"""
    piece = board[r][c]
    if not piece or piece["color"] != turn:
        return []

    raw = get_raw_moves(board, r, c)
    valid = []
    for m in raw:
        to_r, to_c = m["row"], m["col"]
        # 原地模拟走棋（省深拷贝），检查走后己方是否被将
        captured = board[to_r][to_c]
        board[to_r][to_c] = piece
        board[r][c] = None
        if not _is_in_check(board, turn):
            valid.append(m)
        # 撤销模拟
        board[r][c] = piece
        board[to_r][to_c] = captured
    return valid


def is_valid_move(board: list, from_pos: dict, to_pos: dict, turn: str) -> bool:
    """校验走法是否合法"""
    fr, fc = from_pos["row"], from_pos["col"]
    tr, tc = to_pos["row"], to_pos["col"]
    # 检查起点和终点坐标是否在棋盘范围内
    if not in_bounds(fr, fc) or not in_bounds(tr, tc):
        return False
    piece = board[fr][fc]
    if not piece or piece["color"] != turn:
        return False
    moves = get_valid_moves(board, fr, fc, turn)
    # 检查目标位置是否在合法走法列表中
    return any(m["row"] == tr and m["col"] == tc for m in moves)


def make_move(board: list, from_pos: dict, to_pos: dict) -> tuple[list, Optional[dict]]:
    """执行走棋，返回 (新棋盘, 被吃棋子)"""
    fr, fc = from_pos["row"], from_pos["col"]
    tr, tc = to_pos["row"], to_pos["col"]
    new_board = _copy_board(board)
    captured = new_board[tr][tc]
    new_board[tr][tc] = new_board[fr][fc]
    new_board[fr][fc] = None
    return new_board, captured


def check_game_result(board: list, turn: str) -> str:
    """检测游戏结果
    返回: "red_win" / "black_win" / "draw" / "ongoing"
    """
    has_red_king = False
    has_black_king = False
    # 遍历整个棋盘，查找双方的将/帅是否还存在
    for r in range(ROWS):
        for c in range(COLS):
            p = board[r][c]
            if p:
                # 找到红方帅（帥）
                if p["type"] == "帥":
                    has_red_king = True
                # 找到黑方将（將）
                elif p["type"] == "將":
                    has_black_king = True

    if not has_red_king:
        return "black_win"
    if not has_black_king:
        return "red_win"

    # 检查当前方是否无子可动（困毙）
    if not _has_any_valid_move(board, turn):
        # 无子可动 = 输
        return "black_win" if turn == "r" else "red_win"

    return "ongoing"


def is_in_check(board: list, color: str) -> bool:
    """公开接口：检查某方是否被将军"""
    return _is_in_check(board, color)


# ============================================================
#  内部走法实现
# ============================================================

def _moves_rook(board, r, c):
    moves = []
    # 車可以沿四个方向移动：右、左、下、上
    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
        # 每个方向最多走9步（棋盘最大跨度）
        for i in range(1, 10):
            # 计算目标位置坐标
            nr, nc = r + dr*i, c + dc*i
            # 如果超出棋盘边界，停止该方向的搜索
            if not in_bounds(nr, nc):
                break
            # 将该位置加入可走列表
            moves.append({"row": nr, "col": nc})
            # 如果该位置有棋子（无论敌我），車只能走到这里（吃子或阻挡），停止继续延伸
            if board[nr][nc]:
                break
    return moves


def _moves_knight(board, r, c):
    moves = []
    jumps = [
        (1, 2, 0, 1), (1, -2, 0, -1), (-1, 2, 0, 1), (-1, -2, 0, -1),
        (2, 1, 1, 0), (2, -1, 1, 0), (-2, 1, -1, 0), (-2, -1, -1, 0),
    ]
    for dr, dc, br, bc in jumps:
        nr, nc = r + dr, c + dc
        block_r, block_c = r + br, c + bc
        blocked = in_bounds(block_r, block_c) and board[block_r][block_c]
        if in_bounds(nr, nc) and not blocked:
            moves.append({"row": nr, "col": nc})
    return moves


def _moves_elephant(board, r, c, color):
    moves = []
    min_r = 5 if color == "r" else 0
    max_r = 9 if color == "r" else 4
    for dr, dc in [(2,2),(2,-2),(-2,2),(-2,-2)]:
        nr, nc = r + dr, c + dc
        eye_r, eye_c = r + dr//2, c + dc//2
        if in_bounds(nr, nc) and min_r <= nr <= max_r and not board[eye_r][eye_c]:
            moves.append({"row": nr, "col": nc})
    return moves


def _moves_advisor(board, r, c, color):
    moves = []
    min_r = 7 if color == "r" else 0
    max_r = 9 if color == "r" else 2
    for dr, dc in [(1,1),(1,-1),(-1,1),(-1,-1)]:
        nr, nc = r + dr, c + dc
        if in_bounds(nr, nc) and min_r <= nr <= max_r and 3 <= nc <= 5:
            moves.append({"row": nr, "col": nc})
    return moves


def _moves_king(board, r, c, color):
    moves = []
    min_r = 7 if color == "r" else 0
    max_r = 9 if color == "r" else 2
    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
        nr, nc = r + dr, c + dc
        if in_bounds(nr, nc) and min_r <= nr <= max_r and 3 <= nc <= 5:
            moves.append({"row": nr, "col": nc})
    return moves


def _moves_cannon(board, r, c):
    moves = []
    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
        jumped = False
        for i in range(1, 10):
            nr, nc = r + dr*i, c + dc*i
            if not in_bounds(nr, nc):
                break
            if not jumped:
                if board[nr][nc]:
                    jumped = True
                else:
                    moves.append({"row": nr, "col": nc})
            else:
                if board[nr][nc]:
                    moves.append({"row": nr, "col": nc})
                    break
    return moves


def _moves_pawn(board, r, c, color):
    moves = []
    if color == "r":
        if r - 1 >= 0:
            moves.append({"row": r - 1, "col": c})
        if r <= 4:  # 已过河
            if c - 1 >= 0:
                moves.append({"row": r, "col": c - 1})
            if c + 1 < COLS:
                moves.append({"row": r, "col": c + 1})
    else:
        if r + 1 < ROWS:
            moves.append({"row": r + 1, "col": c})
        if r >= 5:  # 已过河
            if c - 1 >= 0:
                moves.append({"row": r, "col": c - 1})
            if c + 1 < COLS:
                moves.append({"row": r, "col": c + 1})
    return moves


# ============================================================
#  辅助函数
# ============================================================

def _copy_board(board):
    return [
        [dict(cell) if cell else None for cell in row]
        for row in board
    ]


def _find_king(board, color):
    king_type = "帥" if color == "r" else "將"
    for r in range(ROWS):
        for c in range(COLS):
            p = board[r][c]
            if p and p["type"] == king_type and p["color"] == color:
                return (r, c)
    return None


def _is_in_check(board, color):
    """检查 color 方是否被将军"""
    # 一次扫描同时定位双方将/帅，省去两次 _find_king
    my_king_type = "帥" if color == "r" else "將"
    enemy_king_type = "將" if color == "r" else "帥"
    enemy_color = "b" if color == "r" else "r"

    my_king = None
    enemy_king = None
    for r in range(ROWS):
        for c in range(COLS):
            p = board[r][c]
            if p:
                t = p["type"]
                if t == my_king_type and p["color"] == color:
                    my_king = (r, c)
                elif t == enemy_king_type and p["color"] == enemy_color:
                    enemy_king = (r, c)

    if not my_king:
        return True  # 将/帅已被吃，视为被将

    kr, kc = my_king

    # 检查所有敌方棋子是否能攻击到将/帅位置
    for r in range(ROWS):
        for c in range(COLS):
            p = board[r][c]
            if p and p["color"] == enemy_color:
                raw = get_raw_moves(board, r, c)
                for m in raw:
                    if m["row"] == kr and m["col"] == kc:
                        return True

    # 将帅对面检测（复用已定位的 enemy_king）
    if enemy_king and enemy_king[1] == kc:
        min_r = min(kr, enemy_king[0])
        max_r = max(kr, enemy_king[0])
        for check_r in range(min_r + 1, max_r):
            if board[check_r][kc]:
                return False
        return True

    return False


def _has_any_valid_move(board, color):
    """检查某方是否有任何合法走法"""
    for r in range(ROWS):
        for c in range(COLS):
            p = board[r][c]
            if p and p["color"] == color:
                if get_valid_moves(board, r, c, color):
                    return True
    return False
