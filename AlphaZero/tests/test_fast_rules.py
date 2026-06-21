"""全面对比测试：fast_chess vs backend.chess_rules

确保优化引擎的行为与现有规则引擎 100% 一致。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from backend import chess_rules as old
from AlphaZero.engine import fast_chess as new
from AlphaZero.engine.constants import ROWS, COLS

# ============================================================
#  转换辅助函数
# ============================================================

# dict type → numeric (absolute value)
_TYPE_TO_NUM = {
    "帥": 1, "仕": 2, "相": 3, "馬": 4, "車": 5, "炮": 6, "兵": 7,
    "將": 1, "士": 2, "象": 3,           "砲": 6, "卒": 7,
}
# numeric → dict (red)
_NUM_TO_RED = {1: "帥", 2: "仕", 3: "相", 4: "馬", 5: "車", 6: "炮", 7: "兵"}
# numeric → dict (black)
_NUM_TO_BLACK = {1: "將", 2: "士", 3: "象", 4: "馬", 5: "車", 6: "砲", 7: "卒"}


def old_to_new(old_board):
    """将 backend 的 list[list[dict]] 转为 numpy int8 (10,9)。"""
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    for r in range(ROWS):
        for c in range(COLS):
            p = old_board[r][c]
            if p:
                sign = 1 if p["color"] == "r" else -1
                board[r, c] = sign * _TYPE_TO_NUM[p["type"]]
    return board


def new_to_old(board):
    """将 numpy int8 (10,9) 转为 backend 的 list[list[dict]]。"""
    result = [[None] * COLS for _ in range(ROWS)]
    for r in range(ROWS):
        for c in range(COLS):
            v = int(board[r, c])
            if v > 0:
                result[r][c] = {"type": _NUM_TO_RED[v], "color": "r"}
            elif v < 0:
                result[r][c] = {"type": _NUM_TO_BLACK[-v], "color": "b"}
    return result


def moves_to_set(moves):
    """将走法列表转为 {(row, col, capture)} 集合，便于无序对比。"""
    return {(m["row"], m["col"], m.get("capture", False)) for m in moves}


# ============================================================
#  Test 1: 初始棋盘一致性
# ============================================================

def test_initial_board_identical():
    old_board = old.create_initial_board()
    new_board = new.create_initial_board()
    new_as_old = new_to_old(new_board)

    for r in range(ROWS):
        for c in range(COLS):
            o = old_board[r][c]
            n = new_as_old[r][c]
            assert type(o) == type(n), f"Type mismatch at ({r},{c}): {o} vs {n}"
            if o is not None:
                assert o["type"] == n["type"], \
                    f"Piece type mismatch at ({r},{c}): {o['type']} vs {n['type']}"
                assert o["color"] == n["color"], \
                    f"Color mismatch at ({r},{c}): {o['color']} vs {n['color']}"


# ============================================================
#  Test 2: 所有棋子的 raw_moves
# ============================================================

def test_raw_moves_all_pieces_initial():
    """逐个位置对比 raw_moves。"""
    old_board = old.create_initial_board()
    new_board = new.create_initial_board()

    total_checked = 0
    for r in range(ROWS):
        for c in range(COLS):
            p = old_board[r][c]
            if p:
                old_moves = moves_to_set(old.get_raw_moves(old_board, r, c))
                new_moves = moves_to_set(new.get_raw_moves(new_board, r, c))
                assert old_moves == new_moves, \
                    f"Raw moves differ for {p['type']}({p['color']}) at ({r},{c}):\n" \
                    f"  old: {sorted(old_moves)}\n  new: {sorted(new_moves)}"
                total_checked += 1

    assert total_checked == 32, f"Expected 32 pieces, found {total_checked}"


# ============================================================
#  Test 3: 所有棋子的 valid_moves
# ============================================================

@pytest.mark.parametrize("turn", ["r", "b"])
def test_valid_moves_all_pieces_initial(turn):
    """逐个位置对比 valid_moves。"""
    old_board = old.create_initial_board()
    new_board = new.create_initial_board()

    total_checked = 0
    for r in range(ROWS):
        for c in range(COLS):
            p = old_board[r][c]
            if p:
                old_moves = moves_to_set(old.get_valid_moves(old_board, r, c, turn))
                new_moves = moves_to_set(new.get_valid_moves(new_board, r, c, turn))
                assert old_moves == new_moves, \
                    f"Valid moves differ for {p['type']}({p['color']}) at ({r},{c}), turn={turn}:\n" \
                    f"  old: {sorted(old_moves)}\n  new: {sorted(new_moves)}"
                total_checked += 1

    assert total_checked == 32


# ============================================================
#  Test 4: is_in_check
# ============================================================

def test_is_in_check_initial():
    """初始棋盘双方都不被将。"""
    old_board = old.create_initial_board()
    new_board = new.create_initial_board()
    assert not old.is_in_check(old_board, "r")
    assert not new.is_in_check(new_board, "r")
    assert not old.is_in_check(old_board, "b")
    assert not new.is_in_check(new_board, "b")


def test_is_in_check_bool_and_str():
    """测试 bool 和 str 两种 color 参数。"""
    board = new.create_initial_board()
    assert is_in_check(board, True) == is_in_check(board, "r")
    assert is_in_check(board, False) == is_in_check(board, "b")


def test_is_in_check_rook_threat():
    """車直线将军。"""
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[9, 4] = KING    # 红帅
    board[0, 4] = -KING   # 黑将
    board[3, 4] = -ROOK   # 黑車在同列，红帅被将
    assert new.is_in_check(board, "r")
    assert not new.is_in_check(board, "b")  # 黑方没有被将


def test_is_in_check_cannon_threat():
    """砲翻山将军。"""
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[9, 4] = KING    # 红帅
    board[0, 4] = -KING   # 黑将
    board[2, 4] = -CANNON  # 黑砲
    board[5, 4] = PAWN     # 炮架
    assert new.is_in_check(board, "r")
    assert not new.is_in_check(board, "b")


def test_is_in_check_knight_threat():
    """馬将军。"""
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[9, 4] = KING     # 红帅
    board[0, 4] = -KING    # 黑将
    board[7, 3] = -KNIGHT  # 黑馬在 (7,3)，可以走到 (9,4) 将军
    # 马腿 (8,3) 为空
    assert new.is_in_check(board, "r")


def test_is_in_check_knight_blocked():
    """蹩马脚，不能将军。"""
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[9, 4] = KING     # 红帅
    board[0, 3] = -KING    # 黑将 (错开列，避免对面将)
    board[7, 3] = -KNIGHT  # 黑馬
    board[8, 3] = PAWN     # 蹩马脚
    assert not new.is_in_check(board, "r")


def test_is_in_check_pawn_threat():
    """兵将军。"""
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[9, 4] = KING    # 红帅
    board[0, 4] = -KING   # 黑将
    board[8, 4] = -PAWN   # 黑卒在帅正上方
    assert new.is_in_check(board, "r")


def test_is_in_check_facing_kings():
    """对面将。"""
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[9, 4] = KING    # 红帅
    board[0, 4] = -KING   # 黑将，同列无遮挡
    assert new.is_in_check(board, "r")
    assert new.is_in_check(board, "b")


def test_is_in_check_facing_kings_blocked():
    """对面将被阻挡则不将军。"""
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[9, 4] = KING    # 红帅
    board[0, 4] = -KING   # 黑将
    board[5, 4] = PAWN    # 中间有兵阻挡（非車，不会将军）
    assert not new.is_in_check(board, "r")
    assert not new.is_in_check(board, "b")


def test_is_in_check_king_missing():
    """帅被吃视为被将。"""
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[0, 4] = -KING   # 只有黑将
    assert new.is_in_check(board, "r")


# ============================================================
#  Test 5: check_game_result
# ============================================================

def test_game_result_initial():
    """初始棋盘进行中。"""
    old_board = old.create_initial_board()
    new_board = new.create_initial_board()
    for turn in ["r", "b"]:
        assert old.check_game_result(old_board, turn) == \
               new.check_game_result(new_board, turn) == "ongoing"


def test_game_result_checkmate():
    """对比双方引擎对多个杀棋/非杀棋局面的判定一致性。"""
    # 场景1: 初始局面
    board1 = new.create_initial_board()
    old_b1 = old.create_initial_board()
    assert new.check_game_result(board1, "r") == old.check_game_result(old_b1, "r")
    assert new.check_game_result(board1, "b") == old.check_game_result(old_b1, "b")

    # 场景2: 红帅被吃 → 黑胜
    board2 = np.zeros((ROWS, COLS), dtype=np.int8)
    board2[0, 4] = -KING
    assert new.check_game_result(board2, "r") == "black_win"
    old_b2 = new_to_old(board2)
    assert old.check_game_result(old_b2, "r") == new.check_game_result(board2, "r")

    # 场景3: 构造困毙/将死，只验证新旧引擎一致
    board3 = np.zeros((ROWS, COLS), dtype=np.int8)
    board3[0, 4] = -KING   # 黑将
    board3[2, 4] = ROOK    # 红車将军
    board3[9, 4] = KING    # 红帅
    old_b3 = new_to_old(board3)
    for turn in ["r", "b"]:
        r_new = new.check_game_result(board3, turn)
        r_old = old.check_game_result(old_b3, turn)
        assert r_new == r_old, f"board3 turn={turn}: new={r_new}, old={r_old}"


# ============================================================
#  Test 6: make_move + undo_move
# ============================================================

def test_make_and_undo():
    """走棋后撤销，应恢复原状。"""
    board = new.create_initial_board()
    original = board.copy()

    undo = new.make_move(board, (9, 1), (7, 2))  # 马二进三
    assert not np.array_equal(board, original), "Board should change after move"
    assert board[7, 2] == 4  # 红马在目标位置

    new.undo_move(board, undo)
    assert np.array_equal(board, original), "Board should be restored after undo"


def test_make_move_with_capture():
    """吃子走棋后撤销。"""
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[9, 4] = KING     # 红帅
    board[0, 4] = -KING    # 黑将
    board[5, 0] = ROOK     # 红車
    board[5, 3] = -PAWN    # 黑卒
    original = board.copy()

    undo = new.make_move(board, (5, 0), (5, 3))  # 車吃卒
    assert board[5, 3] == ROOK
    assert board[5, 0] == 0

    new.undo_move(board, undo)
    assert np.array_equal(board, original)


def test_make_move_dict_args():
    """测试 dict 格式参数。"""
    board = new.create_initial_board()
    original = board.copy()

    undo = new.make_move(board, {"row": 9, "col": 1}, {"row": 7, "col": 2})
    new.undo_move(board, undo)
    assert np.array_equal(board, original)


# ============================================================
#  Test 7: is_valid_move 定点校验
# ============================================================

def test_is_valid_move_simple():
    """基本合法走法。"""
    board = new.create_initial_board()
    assert new.is_valid_move(board, (9, 1), (7, 2), "r")  # 马二进三
    assert not new.is_valid_move(board, (9, 1), (7, 1), "r")  # 蹩马脚


def test_is_valid_move_out_of_bounds():
    """越界走法。"""
    board = new.create_initial_board()
    assert not new.is_valid_move(board, (-1, 0), (0, 0), "r")
    assert not new.is_valid_move(board, (0, 0), (10, 0), "r")


def test_is_valid_move_wrong_turn():
    """走错方的棋子。"""
    board = new.create_initial_board()
    assert not new.is_valid_move(board, (0, 0), (0, 1), "r")  # 黑車在 (0,0)，轮红方


# ============================================================
#  Test 8: 随机对局收敛 (关键测试)
# ============================================================

def test_random_game_convergence():
    """50 局随机对弈，每步后对比两套引擎的 valid_moves。"""
    rng = np.random.RandomState(42)

    for game_idx in range(50):
        old_b = old.create_initial_board()
        new_b = new.create_initial_board()
        turn = "r"

        for move_num in range(300):  # 防止死循环
            # 用旧引擎收集所有合法走法
            all_moves = []
            for r in range(ROWS):
                for c in range(COLS):
                    p = old_b[r][c]
                    if p and p["color"] == turn:
                        moves = old.get_valid_moves(old_b, r, c, turn)
                        for m in moves:
                            all_moves.append((r, c, m["row"], m["col"]))

            if not all_moves:
                break  # 困毙/将死

            # 用新引擎也收集，对比
            for r in range(ROWS):
                for c in range(COLS):
                    p = old_b[r][c]
                    if p and p["color"] == turn:
                        old_m = moves_to_set(
                            old.get_valid_moves(old_b, r, c, turn))
                        new_m = moves_to_set(
                            new.get_valid_moves(new_b, r, c, turn))
                        assert old_m == new_m, \
                            f"[Game {game_idx}, move {move_num}] " \
                            f"Valid moves diverged for {p['type']}({p['color']}) at ({r},{c}):\n" \
                            f"  old: {sorted(old_m)}\n  new: {sorted(new_m)}"

            # 随机选一步
            fr, fc, tr, tc = all_moves[rng.randint(len(all_moves))]

            # 旧引擎走棋
            old_b, _ = old.make_move(old_b,
                                     {"row": fr, "col": fc},
                                     {"row": tr, "col": tc})

            # 新引擎走棋（原地）
            new.make_move(new_b, (fr, fc), (tr, tc))

            # 对比棋盘状态
            new_as_old = new_to_old(new_b)
            for r in range(ROWS):
                for c in range(COLS):
                    o = old_b[r][c]
                    n = new_as_old[r][c]
                    if o is None:
                        assert n is None, \
                            f"[Game {game_idx}, move {move_num}] " \
                            f"At ({r},{c}): old=None, new={n}"
                    else:
                        assert n is not None, \
                            f"[Game {game_idx}, move {move_num}] " \
                            f"At ({r},{c}): old={o}, new=None"
                        assert o["type"] == n["type"] and o["color"] == n["color"], \
                            f"[Game {game_idx}, move {move_num}] " \
                            f"At ({r},{c}): old={o}, new={n}"

            turn = "b" if turn == "r" else "r"

            # 终局检测
            if old.check_game_result(old_b, turn) != "ongoing":
                new_result = new.check_game_result(new_b, turn)
                old_result = old.check_game_result(old_b, turn)
                assert new_result == old_result, \
                    f"[Game {game_idx}, move {move_num}] " \
                    f"Result mismatch: old={old_result}, new={new_result}"
                break


# ============================================================
#  Test 9: 专项边缘用例
# ============================================================

def test_horse_leg_block():
    """蹩马脚：马腿有子时不能走。"""
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[9, 4] = KING    # 红帅
    board[5, 4] = KNIGHT  # 红馬
    board[6, 4] = PAWN    # 蹩马脚（垂直方向）
    board[0, 0] = -KING   # 黑将

    moves = new.get_raw_moves(board, 5, 4)
    move_positions = {(m["row"], m["col"]) for m in moves}
    # 马腿 (6,4) 被蹩，不能走 (7,3) 和 (7,5)
    assert (7, 3) not in move_positions, "Horse leg blocked: should NOT reach (7,3)"
    assert (7, 5) not in move_positions, "Horse leg blocked: should NOT reach (7,5)"
    # 但可以走 (4,2) (4,6) (6,2) (6,6) (3,3) (3,5)
    assert (3, 3) in move_positions
    assert (3, 5) in move_positions
    assert (4, 2) in move_positions


def test_elephant_eye_block():
    """塞象眼：象眼有子时不能走。"""
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[9, 4] = KING     # 红帅
    board[7, 4] = ELEPHANT  # 红相
    board[6, 5] = PAWN     # 塞象眼——擋住 (5,6) 方向
    board[0, 0] = -KING    # 黑将

    moves = new.get_raw_moves(board, 7, 4)
    move_positions = {(m["row"], m["col"]) for m in moves}
    assert (5, 6) not in move_positions, "Elephant eye blocked: should NOT reach (5,6)"
    # 其他方向正常
    assert (5, 2) in move_positions
    assert (9, 2) in move_positions
    assert (9, 6) in move_positions


def test_elephant_cannot_cross_river():
    """象不能过河。"""
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[9, 4] = KING     # 红帅
    board[5, 2] = ELEPHANT  # 红相在河边界
    board[0, 0] = -KING    # 黑将

    moves = new.get_raw_moves(board, 5, 2)
    move_positions = {(m["row"], m["col"]) for m in moves}
    # 只能走到 7,0 和 7,4 (不会越过 row 4)
    for r, c in move_positions:
        assert r >= 5, f"Elephant at (5,2) crossed river to ({r},{c})"


def test_cannon_jump():
    """砲必须翻山吃子。"""
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[9, 4] = KING     # 红帅
    board[7, 1] = CANNON   # 红炮
    board[7, 4] = PAWN     # 炮架
    board[7, 7] = -ROOK    # 黑車（炮架后面的目标）
    board[0, 0] = -KING    # 黑将

    moves = new.get_raw_moves(board, 7, 1)
    move_set = {(m["row"], m["col"], m["capture"]) for m in moves}

    # 可以空走到炮架之前
    assert (7, 2, False) in move_set
    assert (7, 3, False) in move_set
    # 可以翻山吃子
    assert (7, 7, True) in move_set
    # 不能落在炮架上（己方棋子）
    assert (7, 4, False) not in move_set
    assert (7, 4, True) not in move_set
    # 不能落在炮架和目標之间
    assert (7, 5, False) not in move_set
    assert (7, 6, False) not in move_set


def test_king_facing_illegal():
    """将帅对面是非法的（走完后不能对面）。"""
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[0, 4] = -KING   # 黑将在上 (0,4)
    board[9, 3] = KING    # 红帅在 (9,3)
    # 同列无任何阻挡：红帅走到 (9,4) 会与黑将对面

    old_b = new_to_old(board)
    # 红帅试图走到 (9,4) 会与黑将对面 → 非法
    assert not new.is_valid_move(board, (9, 3), (9, 4), "r")
    assert not old.is_valid_move(old_b, {"row": 9, "col": 3},
                                 {"row": 9, "col": 4}, "r")


def test_advisor_stays_in_palace():
    """仕不能出宫。"""
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[9, 4] = KING     # 红帅
    board[8, 4] = ADVISOR  # 红仕
    board[0, 0] = -KING    # 黑将

    moves = new.get_raw_moves(board, 8, 4)
    move_positions = {(m["row"], m["col"]) for m in moves}
    # 仕只能在 (7,3), (7,5), (9,3), (9,5) 中
    for r, c in move_positions:
        assert 7 <= r <= 9, f"Advisor left palace: ({r},{c})"
        assert 3 <= c <= 5, f"Advisor left palace: ({r},{c})"


def test_king_stays_in_palace():
    """帅不能出宫。"""
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[9, 4] = KING    # 红帅
    board[0, 0] = -KING   # 黑将

    moves = new.get_raw_moves(board, 9, 4)
    move_positions = {(m["row"], m["col"]) for m in moves}
    for r, c in move_positions:
        assert 7 <= r <= 9, f"King left palace: ({r},{c})"
        assert 3 <= c <= 5, f"King left palace: ({r},{c})"


# ============================================================
#  Test 10: 性能基准（非断言，仅输出）
# ============================================================

def test_performance_benchmark():
    """输出性能基准数据。"""
    import time

    board = new.create_initial_board()
    n = 50_000

    # is_in_check
    t0 = time.perf_counter()
    for _ in range(n):
        new.is_in_check(board, True)
    is_check_time = (time.perf_counter() - t0) / n * 1e9
    print(f"\n  is_in_check: {is_check_time:.0f} ns/call")

    # get_raw_moves (马)
    t0 = time.perf_counter()
    for _ in range(n):
        new.get_raw_moves(board, 9, 1)
    raw_time = (time.perf_counter() - t0) / n * 1e6
    print(f"  get_raw_moves(knight): {raw_time:.1f} μs/call")

    # get_valid_moves (马)
    t0 = time.perf_counter()
    for _ in range(n):
        new.get_valid_moves(board, 9, 1, "r")
    valid_time = (time.perf_counter() - t0) / n * 1e6
    print(f"  get_valid_moves(knight): {valid_time:.1f} μs/call")

    # has_any_valid_move
    t0 = time.perf_counter()
    for _ in range(n):
        new.has_any_valid_move(board, True)
    any_time = (time.perf_counter() - t0) / n * 1e6
    print(f"  has_any_valid_move: {any_time:.1f} μs/call")

    # make_move + undo_move
    t0 = time.perf_counter()
    for _ in range(n):
        undo = new.make_move(board, (9, 1), (7, 2))
        new.undo_move(board, undo)
    move_time = (time.perf_counter() - t0) / n * 1e6
    print(f"  make+undo cycle: {move_time:.1f} μs/call")

    # 不做硬断言，仅汇报
    assert is_check_time < 1e6, f"is_in_check too slow: {is_check_time:.0f}ns"


# ============================================================
#  辅助（避免重复 import）
# ============================================================

from AlphaZero.engine.constants import KING, ADVISOR, ELEPHANT, KNIGHT, ROOK, CANNON, PAWN
from AlphaZero.engine.fast_chess import is_in_check
