"""优化的中国象棋规则引擎 — numpy int8 矩阵 + 原地模拟/撤销

专为 AlphaZero MCTS 设计，目标:
  - is_in_check:  ~500ns (反向射线检测)
  - get_valid_moves: ~5μs (原地模拟+撤销)
  - has_any_valid_move: ~10μs (numpy 向量化早停)

棋盘编码:
  正数(红): 1=帥 2=仕 3=相 4=馬 5=車 6=炮 7=兵
  负数(黑): -1=將 -2=士 -3=象 -4=馬 -5=車 -6=砲 -7=卒
  0=空
"""
import numpy as np
from typing import Optional
from .constants import (ROWS, COLS,
                         KING, ADVISOR, ELEPHANT, KNIGHT, ROOK, CANNON, PAWN,
                         RED, BLACK, INIT_LAYOUT)


# ============================================================
#  A. 棋盘创建
# ============================================================

def create_initial_board() -> np.ndarray:
    """创建初始棋盘，返回 numpy int8 (10,9) 矩阵。"""
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    for r, c, ptype, color in INIT_LAYOUT:
        board[r, c] = color * ptype
    return board


# ============================================================
#  B. 原始走法生成
# ============================================================

def get_raw_moves(board: np.ndarray, r: int, c: int) -> list[dict]:
    """获取棋子的原始走法（不含将军校验）。

    Args:
        board: numpy int8 (10,9)
        r, c: 棋子位置

    Returns:
        list[dict]: 每项 {"row": int, "col": int, "capture": bool}
    """
    cell = board[r, c]
    if cell == 0:
        return []

    ptype = abs(cell)
    is_red = cell > 0

    # 分发到各棋子类型的走法生成函数
    if ptype == ROOK:
        raw = _raw_rook(board, r, c)
    elif ptype == KNIGHT:
        raw = _raw_knight(board, r, c)
    elif ptype == ELEPHANT:
        raw = _raw_elephant(board, r, c, is_red)
    elif ptype == ADVISOR:
        raw = _raw_advisor(board, r, c, is_red)
    elif ptype == KING:
        raw = _raw_king(board, r, c, is_red)
    elif ptype == CANNON:
        raw = _raw_cannon(board, r, c)
    elif ptype == PAWN:
        raw = _raw_pawn(board, r, c, is_red)
    else:
        return []

    # 过滤：不能吃己方棋子
    result = []
    for m in raw:
        target = board[m["row"], m["col"]]
        if target == 0:
            m["capture"] = False
            result.append(m)
        elif (target > 0) != is_red:  # 异色 = 敌方
            m["capture"] = True
            result.append(m)
        # 同色跳过（不能吃己方）
    return result


def _raw_rook(board, r, c):
    moves = []
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        for i in range(1, 10):
            nr, nc = r + dr * i, c + dc * i
            if not _in_bounds(nr, nc):
                break
            moves.append({"row": nr, "col": nc})
            if board[nr, nc] != 0:
                break
    return moves


def _raw_knight(board, r, c):
    moves = []
    # (dr, dc, leg_r, leg_c) — 目标偏移 + 马腿偏移
    jumps = [
        (1, 2, 0, 1), (1, -2, 0, -1), (-1, 2, 0, 1), (-1, -2, 0, -1),
        (2, 1, 1, 0), (2, -1, 1, 0), (-2, 1, -1, 0), (-2, -1, -1, 0),
    ]
    for dr, dc, lr, lc in jumps:
        nr, nc = r + dr, c + dc
        leg_r, leg_c = r + lr, c + lc
        if not _in_bounds(nr, nc):
            continue
        if board[leg_r, leg_c] == 0:  # 马腿未被蹩
            moves.append({"row": nr, "col": nc})
    return moves


def _raw_elephant(board, r, c, is_red):
    moves = []
    min_r, max_r = (5, 9) if is_red else (0, 4)
    for dr, dc in [(2, 2), (2, -2), (-2, 2), (-2, -2)]:
        nr, nc = r + dr, c + dc
        eye_r, eye_c = r + dr // 2, c + dc // 2
        if (_in_bounds(nr, nc) and min_r <= nr <= max_r
                and board[eye_r, eye_c] == 0):
            moves.append({"row": nr, "col": nc})
    return moves


def _raw_advisor(board, r, c, is_red):
    moves = []
    min_r, max_r = (7, 9) if is_red else (0, 2)
    for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        nr, nc = r + dr, c + dc
        if (_in_bounds(nr, nc) and min_r <= nr <= max_r
                and 3 <= nc <= 5):
            moves.append({"row": nr, "col": nc})
    return moves


def _raw_king(board, r, c, is_red):
    moves = []
    min_r, max_r = (7, 9) if is_red else (0, 2)
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if (_in_bounds(nr, nc) and min_r <= nr <= max_r
                and 3 <= nc <= 5):
            moves.append({"row": nr, "col": nc})
    return moves


def _raw_cannon(board, r, c):
    moves = []
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        jumped = False
        for i in range(1, 10):
            nr, nc = r + dr * i, c + dc * i
            if not _in_bounds(nr, nc):
                break
            if not jumped:
                if board[nr, nc] != 0:
                    jumped = True  # 找到炮架
                else:
                    moves.append({"row": nr, "col": nc})
            else:
                if board[nr, nc] != 0:
                    moves.append({"row": nr, "col": nc})  # 翻山吃子
                    break
    return moves


def _raw_pawn(board, r, c, is_red):
    moves = []
    if is_red:
        # 红兵向前（上）
        if r - 1 >= 0:
            moves.append({"row": r - 1, "col": c})
        # 过河后可以左右走
        if r <= 4:
            if c - 1 >= 0:
                moves.append({"row": r, "col": c - 1})
            if c + 1 < COLS:
                moves.append({"row": r, "col": c + 1})
    else:
        # 黑卒向前（下）
        if r + 1 < ROWS:
            moves.append({"row": r + 1, "col": c})
        # 过河后可以左右走
        if r >= 5:
            if c - 1 >= 0:
                moves.append({"row": r, "col": c - 1})
            if c + 1 < COLS:
                moves.append({"row": r, "col": c + 1})
    return moves


# ============================================================
#  C. 将军检测 (反向射线)
# ============================================================

def is_in_check(board: np.ndarray, color) -> bool:
    """检查 color 方是否被将军。

    使用反向射线检测：从将/帅位置出发，检查是否有敌方棋子可以攻击到。

    Args:
        board: numpy int8 (10,9)
        color: True/1/"r"=红方, False/-1/"b"=黑方

    Returns:
        bool: True 如果被将军或将/帅已被吃
    """
    if isinstance(color, str):
        color_bool = (color == "r")
    elif isinstance(color, bool):
        color_bool = color
    else:
        color_bool = (color > 0)

    king_val = KING if color_bool else -KING

    # 找将/帅位置
    positions = np.where(board == king_val)
    if len(positions[0]) == 0:
        return True  # 将/帅已被吃

    kr, kc = positions[0][0], positions[1][0]
    enemy_sign = -1 if color_bool else 1  # 敌方棋子符号

    # 1) 4 方向射线扫描 (車/砲)
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        if _check_ray(board, kr, kc, dr, dc, enemy_sign):
            return True

    # 2) 对面将检测（仅同列）
    if _check_facing_kings(board, kr, kc, enemy_sign):
        return True

    # 3) 8 个马位检测
    if _check_knight_threats(board, kr, kc, enemy_sign):
        return True

    # 4) 兵/卒威胁检测
    if _check_pawn_threats(board, kr, kc, enemy_sign, color_bool):
        return True

    return False


def _check_ray(board, kr, kc, dr, dc, enemy_sign):
    """沿 (dr,dc) 方向扫描，检测車/砲。

    逻辑：
    - 遇到的第一子如果是敵車(5) → 将军
    - 第一子成为炮架，继续扫描
    - 遇到的第二子如果是敵砲(6) → 将军
    - 对面将在调用处单独处理
    """
    for step in range(1, 10):
        nr = kr + dr * step
        nc = kc + dc * step
        if not _in_bounds(nr, nc):
            return False

        piece = board[nr, nc]
        if piece == 0:
            continue

        # 第一子：敵車直接将军
        if piece * enemy_sign > 0 and abs(piece) == ROOK:
            return True

        # 此子成为炮架，继续扫描找炮
        for step2 in range(step + 1, 10):
            nr2 = kr + dr * step2
            nc2 = kc + dc * step2
            if not _in_bounds(nr2, nc2):
                break
            piece2 = board[nr2, nc2]
            if piece2 == 0:
                continue
            # 第二子：敵砲则将军
            if piece2 * enemy_sign > 0 and abs(piece2) == CANNON:
                return True
            break  # 第二子不是砲，被阻挡

        return False  # 只有一个炮架，没有后面的砲

    return False


def _check_facing_kings(board, kr, kc, enemy_sign):
    """对面将 + 相邻将检测。

    1. 敵将相邻（同行或同列差1步）→ 将军
    2. 敵将在同列且无遮挡（对面将）→ 将军
    """
    enemy_king_val = KING if enemy_sign == 1 else -KING
    positions = np.where(board == enemy_king_val)
    if len(positions[0]) == 0:
        return False

    ekr, ekc = positions[0][0], positions[1][0]

    # 相邻检测（敵将可在同行或同列一步之遥）
    if abs(kr - ekr) + abs(kc - ekc) == 1:
        return True

    # 对面将（同列且无遮挡）
    if ekc != kc:
        return False

    min_r, max_r = min(kr, ekr), max(kr, ekr)
    for check_r in range(min_r + 1, max_r):
        if board[check_r, kc] != 0:
            return False
    return True


def _check_knight_threats(board, kr, kc, enemy_sign):
    """检查 8 个马位是否有敵方马能将军（含蹩马脚检测）。

    从将/帅位置出发，检查每个可能的敌方马位置。
    马走"日"字：先直走再斜走，蹩脚点在直走的第一步。
    (dr, dc) = 马相对将的位置，(lr, lc) = 蹩脚点相对将的位置。
    """
    knight_targets = [
        (-2, -1, -1, -1), (-2, +1, -1, +1),
        (+2, -1, +1, -1), (+2, +1, +1, +1),
        (-1, -2, -1, -1), (-1, +2, -1, +1),
        (+1, -2, +1, -1), (+1, +2, +1, +1),
    ]
    for dr, dc, lr, lc in knight_targets:
        nr = kr + dr
        nc = kc + dc
        leg_r = kr + lr
        leg_c = kc + lc
        if not _in_bounds(nr, nc):
            continue
        if board[leg_r, leg_c] != 0:
            continue  # 蹩马脚
        target = board[nr, nc]
        if target * enemy_sign > 0 and abs(target) == KNIGHT:
            return True
    return False


def _check_pawn_threats(board, kr, kc, enemy_sign, king_is_red):
    """检查敵方兵/卒是否能将军。

    从将/帅视角：
    - 红帅被黑卒攻：黑卒从上方来 (kr-1, kc)，或同行侧方 (kr, kc±1) 且卒已过河 (kr>=5)
    - 黑将被红兵攻：红兵从下方来 (kr+1, kc)，或同行侧方 (kr, kc±1) 且兵已过河 (kr<=4)
    """
    if king_is_red:
        # 黑卒从上方来
        if kr > 0:
            p = board[kr - 1, kc]
            if p * enemy_sign > 0 and abs(p) == PAWN:
                return True
        # 黑卒侧面攻击（已过河，卒所在行 ≥ 5）
        if kr >= 5:
            for dc in (-1, 1):
                nc = kc + dc
                if 0 <= nc < COLS:
                    p = board[kr, nc]
                    if p * enemy_sign > 0 and abs(p) == PAWN:
                        return True
    else:
        # 红兵从下方来
        if kr < ROWS - 1:
            p = board[kr + 1, kc]
            if p * enemy_sign > 0 and abs(p) == PAWN:
                return True
        # 红兵侧面攻击（已过河，兵所在行 ≤ 4）
        if kr <= 4:
            for dc in (-1, 1):
                nc = kc + dc
                if 0 <= nc < COLS:
                    p = board[kr, nc]
                    if p * enemy_sign > 0 and abs(p) == PAWN:
                        return True
    return False


# ============================================================
#  D. 合法走法
# ============================================================

def get_valid_moves(board: np.ndarray, r: int, c: int, turn) -> list[dict]:
    """获取合法走法（含将军校验）。

    对每个原始走法执行原地模拟 → is_in_check → 撤销。
    """
    if isinstance(turn, str):
        turn_bool = (turn == "r")
    else:
        turn_bool = bool(turn)

    cell = board[r, c]
    if cell == 0:
        return []
    if (cell > 0) != turn_bool:  # 不是当前走棋方的棋子
        return []

    raw = get_raw_moves(board, r, c)
    valid = []
    for m in raw:
        tr, tc = m["row"], m["col"]
        # 原地模拟走棋
        captured = board[tr, tc]
        board[tr, tc] = cell
        board[r, c] = 0
        if not is_in_check(board, turn_bool):
            valid.append(m)
        # 撤销
        board[r, c] = cell
        board[tr, tc] = captured
    return valid


def is_valid_move(board: np.ndarray, from_pos, to_pos, turn) -> bool:
    """校验单步走法是否合法（定点校验，不枚举全部走法）。"""
    if isinstance(from_pos, dict):
        fr, fc = from_pos["row"], from_pos["col"]
    else:
        fr, fc = from_pos
    if isinstance(to_pos, dict):
        tr, tc = to_pos["row"], to_pos["col"]
    else:
        tr, tc = to_pos

    if not _in_bounds(fr, fc) or not _in_bounds(tr, tc):
        return False

    cell = board[fr, fc]
    if cell == 0:
        return False

    if isinstance(turn, str):
        turn_bool = (turn == "r")
    else:
        turn_bool = bool(turn)

    if (cell > 0) != turn_bool:
        return False

    # ① 原始走法规则检查
    raw = get_raw_moves(board, fr, fc)
    if not any(m["row"] == tr and m["col"] == tc for m in raw):
        return False

    # ② 走后己方帅是否安全
    captured = board[tr, tc]
    board[tr, tc] = cell
    board[fr, fc] = 0
    safe = not is_in_check(board, turn_bool)
    board[fr, fc] = cell
    board[tr, tc] = captured
    return safe


# ============================================================
#  E. 走棋执行（原地 + 撤销）
# ============================================================

def make_move(board: np.ndarray, from_pos, to_pos) -> dict:
    """执行走棋（原地修改棋盘）。返回 undo_info 用于撤销。

    Args:
        board: numpy int8 (10,9) — 原地修改
        from_pos: {"row": r, "col": c} 或 (r, c)
        to_pos:   {"row": r, "col": c} 或 (r, c)

    Returns:
        dict: undo_info = {"fr", "fc", "tr", "tc", "piece", "captured"}
    """
    if isinstance(from_pos, dict):
        fr, fc = from_pos["row"], from_pos["col"]
    else:
        fr, fc = from_pos
    if isinstance(to_pos, dict):
        tr, tc = to_pos["row"], to_pos["col"]
    else:
        tr, tc = to_pos

    piece = int(board[fr, fc])
    captured = int(board[tr, tc])
    board[tr, tc] = piece
    board[fr, fc] = 0

    return {"fr": fr, "fc": fc, "tr": tr, "tc": tc,
            "piece": piece, "captured": captured}


def undo_move(board: np.ndarray, undo_info: dict) -> None:
    """撤销一步走棋。必须按 LIFO 顺序调用。"""
    board[undo_info["fr"], undo_info["fc"]] = undo_info["piece"]
    board[undo_info["tr"], undo_info["tc"]] = undo_info["captured"]


# ============================================================
#  F. 游戏状态
# ============================================================

def check_game_result(board: np.ndarray, turn) -> str:
    """检测游戏结果。

    Returns:
        "red_win" / "black_win" / "draw" / "ongoing"
    """
    if isinstance(turn, str):
        turn_bool = (turn == "r")
    else:
        turn_bool = bool(turn)

    has_red_king = np.any(board == KING)
    has_black_king = np.any(board == -KING)

    if not has_red_king:
        return "black_win"
    if not has_black_king:
        return "red_win"

    # 当前方无子可动 = 困毙
    if not has_any_valid_move(board, turn_bool):
        return "black_win" if turn_bool else "red_win"

    return "ongoing"


def has_any_valid_move(board: np.ndarray, color) -> bool:
    """检查 color 方是否有任何合法走法。

    使用 numpy 向量化定位所有己方棋子，早停。
    """
    if isinstance(color, str):
        color_bool = (color == "r")
    else:
        color_bool = bool(color)

    sign = 1 if color_bool else -1
    # numpy 向量化：找到所有己方棋子
    rows, cols = np.where(board * sign > 0)
    for r, c in zip(rows, cols):
        if get_valid_moves(board, r, c, color_bool):
            return True
    return False


# ============================================================
#  内部辅助
# ============================================================

def _in_bounds(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS
