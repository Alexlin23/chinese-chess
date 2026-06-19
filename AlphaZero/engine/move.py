"""走法表示 — Move 值对象 + ActionEncoder 双向映射"""
from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass(frozen=True)
class Move:
    """不可变走法值对象"""
    from_row: int
    from_col: int
    to_row: int
    to_col: int

    def __repr__(self):
        return f"Move({self.from_row},{self.from_col}→{self.to_row},{self.to_col})"


# ============================================================
#  ActionEncoder — 走法 ↔ 策略向量索引
# ============================================================

class ActionEncoder:
    """双向映射：Move ↔ 策略向量索引。

    预计算所有合理的 (from→to) 组合，建立双向查找表。
    策略向量 size = len(lookup)，每个索引对应一个特定走法。
    """

    # 类变量，首次 build_lookup() 时填充
    _lookup: list[tuple] = None         # [idx] → (fr,fc,tr,tc)
    _reverse: dict = None               # (fr,fc,tr,tc) → idx
    POLICY_SIZE: int = 0

    @classmethod
    def build_lookup(cls) -> None:
        """预计算映射表。在模块加载时自动调用。"""
        if cls._lookup is not None:
            return

        moves_set = set()

        for r in range(10):
            for c in range(9):
                # 車/砲走法：同行同列
                for nc in range(9):
                    if nc != c:
                        moves_set.add((r, c, r, nc))
                for nr in range(10):
                    if nr != r:
                        moves_set.add((r, c, nr, c))

                # 馬走法：8 个日字
                for dr, dc in [(2, 1), (2, -1), (-2, 1), (-2, -1),
                               (1, 2), (1, -2), (-1, 2), (-1, -2)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 10 and 0 <= nc < 9:
                        moves_set.add((r, c, nr, nc))

                # 士/将走法：对角线 + 直线（含宫内）
                for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1),
                               (0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 10 and 0 <= nc < 9:
                        moves_set.add((r, c, nr, nc))

                # 象走法：田字对角
                for dr, dc in [(2, 2), (2, -2), (-2, 2), (-2, -2)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 10 and 0 <= nc < 9:
                        moves_set.add((r, c, nr, nc))

        cls._lookup = sorted(moves_set)
        cls._reverse = {(fr, fc, tr, tc): i
                        for i, (fr, fc, tr, tc) in enumerate(cls._lookup)}
        cls.POLICY_SIZE = len(cls._lookup)

    @classmethod
    def encode(cls, move: Move) -> int:
        """走法 → 索引"""
        return cls._reverse.get((move.from_row, move.from_col,
                                  move.to_row, move.to_col), -1)

    @classmethod
    def decode(cls, index: int) -> Optional[Move]:
        """索引 → 走法"""
        if 0 <= index < len(cls._lookup):
            fr, fc, tr, tc = cls._lookup[index]
            return Move(fr, fc, tr, tc)
        return None

    @classmethod
    def legal_mask(cls, state: 'GameState') -> np.ndarray:
        """返回 (POLICY_SIZE,) bool 数组，标记当前局面合法走法索引。"""
        mask = np.zeros(cls.POLICY_SIZE, dtype=bool)
        for move in state.legal_moves():
            idx = cls.encode(move)
            if idx >= 0:
                mask[idx] = True
        return mask

    @classmethod
    def legal_indices(cls, state: 'GameState') -> np.ndarray:
        """返回合法走法索引列表。"""
        indices = []
        for move in state.legal_moves():
            idx = cls.encode(move)
            if idx >= 0:
                indices.append(idx)
        return np.array(indices, dtype=np.int32)


# 模块加载时自动构建
ActionEncoder.build_lookup()
