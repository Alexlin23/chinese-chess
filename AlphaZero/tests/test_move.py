"""Move 编解码测试"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from AlphaZero.engine.move import Move, ActionEncoder


def test_move_immutable():
    """Move 是不可变值对象。"""
    m1 = Move(0, 0, 1, 1)
    m2 = Move(0, 0, 1, 1)
    m3 = Move(0, 0, 1, 2)
    assert m1 == m2
    assert m1 != m3
    assert hash(m1) == hash(m2)
    assert hash(m1) != hash(m3)


def test_encode_decode_roundtrip():
    """encode → decode 往返一致性。"""
    for idx in range(ActionEncoder.POLICY_SIZE):
        move = ActionEncoder.decode(idx)
        assert move is not None, f"Decode returned None for idx={idx}"
        idx2 = ActionEncoder.encode(move)
        assert idx == idx2, f"Roundtrip failed: {idx} → {move} → {idx2}"


def test_encode_invalid():
    """无效走法返回 -1。"""
    # (0,0) → (0,0) 不在查找表中
    m = Move(0, 0, 0, 0)
    assert ActionEncoder.encode(m) == -1


def test_policy_size():
    """策略向量大小合理。"""
    assert 2000 < ActionEncoder.POLICY_SIZE < 3000, \
        f"POLICY_SIZE={ActionEncoder.POLICY_SIZE} out of expected range"


def test_legal_mask():
    """legal_mask 只标记合法走法。"""
    from AlphaZero.engine.state import GameState
    state = GameState.new_game()
    mask = ActionEncoder.legal_mask(state)
    legal = ActionEncoder.legal_indices(state)

    assert mask.sum() > 0, "Should have legal moves"
    assert mask.sum() == len(legal), "Mask and indices count mismatch"
    # 初局 44 步合法走法
    assert mask.sum() == 44, f"Expected 44 legal moves, got {mask.sum()}"
    assert len(legal) == 44


def test_all_legal_moves_encodable():
    """所有合法走法都能编码。"""
    from AlphaZero.engine.state import GameState
    state = GameState.new_game()
    for move in state.legal_moves():
        idx = ActionEncoder.encode(move)
        assert idx >= 0, f"Legal move {move} not encodable!"
        decoded = ActionEncoder.decode(idx)
        assert decoded == move, f"Encode/decode mismatch: {move} vs {decoded}"
