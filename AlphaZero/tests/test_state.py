"""GameState 状态机测试"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from AlphaZero.engine import GameState, Move


def test_new_game():
    """新对局初始状态正确。"""
    state = GameState.new_game()
    assert len(state.legal_moves()) == 44
    assert not state.is_terminal()
    assert state.result() is None
    assert not state.is_in_check()


def test_apply_move():
    """走棋后切换轮到对方。"""
    state = GameState.new_game()
    moves = state.legal_moves()
    m = moves[0]
    s2 = state.apply(m)
    assert s2.move_count == 1
    assert len(s2.legal_moves()) == 44  # 黑方也有44步


def test_is_legal():
    """走法合法性检测。"""
    state = GameState.new_game()
    # 合法走法
    assert state.is_legal(Move(9, 1, 7, 2))  # 马二进三
    # 非法走法
    assert not state.is_legal(Move(9, 0, 9, 0))  # 原地不动
    assert not state.is_legal(Move(6, 0, 4, 0))  # 兵只能进一


def test_terminal_detection():
    """终局检测。"""
    # 红帅被吃
    board_no_red_king = np.zeros((10, 9), dtype=np.int8)
    board_no_red_king[0, 4] = -1  # 只有黑将
    state = GameState(board_no_red_king, "r")
    assert state.is_terminal()
    assert state.result() == -1.0  # 红输

    # 黑将被吃
    board_no_black_king = np.zeros((10, 9), dtype=np.int8)
    board_no_black_king[9, 4] = 1  # 只有红帅
    state2 = GameState(board_no_black_king, "b")
    assert state2.is_terminal()
    assert state2.result() == 1.0  # 红赢


def test_encoding_shape():
    """神经网络编码维度正确。"""
    state = GameState.new_game()
    enc = state.encode()
    assert enc.shape == (18, 10, 9)
    assert enc.dtype == np.float32
    assert 0.0 <= enc.min() <= enc.max() <= 1.0


def test_encoding_consistency():
    """编码在走棋后合理变化。"""
    s1 = GameState.new_game()
    e1 = s1.encode()

    # 走一步后编码应不同
    moves = s1.legal_moves()
    s2 = s1.apply(moves[0])
    e2 = s2.encode()

    # 通道内容应该不同（己方/对方交换）
    assert not np.array_equal(e1, e2), "Encodings should differ after move"

    # 步数通道应增加
    assert e2[15].max() > e1[15].max(), "Step count should increase"


def test_apply_preserves_original():
    """apply 不修改原状态。"""
    s1 = GameState.new_game()
    board_copy = s1.board.copy()
    moves = s1.legal_moves()
    s2 = s1.apply(moves[0])

    # s1 的棋盘不应改变
    assert np.array_equal(s1.board, board_copy), "Original board should not mutate"
    # s2 的棋盘不同
    assert not np.array_equal(s2.board, board_copy), "New state should have different board"
