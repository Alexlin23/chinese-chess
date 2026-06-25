"""动作编码 — Move 值对象 + ActionEncoder 8100 动作空间

action = from_square * 90 + to_square
from_square = from_row * 9 + from_col
to_square = to_row * 9 + to_col
"""
from dataclasses import dataclass
import numpy as np
from typing import Optional

from .constants import ROWS, COLS, BOARD_SIZE, POLICY_SIZE


@dataclass(frozen=True)
class Move:
    """不可变走法值对象"""
    from_row: int
    from_col: int
    to_row: int
    to_col: int

    @property
    def from_square(self) -> int:
        return self.from_row * COLS + self.from_col

    @property
    def to_square(self) -> int:
        return self.to_row * COLS + self.to_col

    def __repr__(self):
        return f"Move({self.from_row},{self.from_col}→{self.to_row},{self.to_col})"


class ActionEncoder:
    """双向映射：Move ↔ 动作索引 (0..8099)

    action = from_square * 90 + to_square
    """

    @staticmethod
    def in_bounds(row: int, col: int) -> bool:
        return 0 <= row < ROWS and 0 <= col < COLS

    @staticmethod
    def encode(move: Move) -> int:
        """走法 → 索引，越界返回 -1"""
        if not ActionEncoder.in_bounds(move.from_row, move.from_col):
            return -1
        if not ActionEncoder.in_bounds(move.to_row, move.to_col):
            return -1
        return move.from_square * BOARD_SIZE + move.to_square

    @staticmethod
    def decode(action: int) -> Optional[Move]:
        """索引 → 走法，越界返回 None"""
        if action < 0 or action >= POLICY_SIZE:
            return None
        from_square = action // BOARD_SIZE
        to_square = action % BOARD_SIZE
        from_row, from_col = divmod(from_square, COLS)
        to_row, to_col = divmod(to_square, COLS)
        return Move(from_row, from_col, to_row, to_col)

    @staticmethod
    def legal_mask(legal_moves: list) -> np.ndarray:
        """返回 (8100,) bool 数组，标记合法走法"""
        mask = np.zeros(POLICY_SIZE, dtype=bool)
        for move in legal_moves:
            action = ActionEncoder.encode(move)
            if 0 <= action < POLICY_SIZE:
                mask[action] = True
        return mask

    @staticmethod
    def legal_actions(legal_moves: list) -> np.ndarray:
        """返回合法动作索引数组"""
        actions = []
        for move in legal_moves:
            action = ActionEncoder.encode(move)
            if 0 <= action < POLICY_SIZE:
                actions.append(action)
        return np.asarray(actions, dtype=np.int32)

    @staticmethod
    def mask_logits(logits: np.ndarray, legal_moves: list) -> np.ndarray:
        """对 logits 应用合法动作掩码，非法动作设为 -inf

        Args:
            logits: (8100,) 原始 logits
            legal_moves: 合法走法列表

        Returns:
            masked_logits: (8100,) 掩码后的 logits
        """
        masked = np.full_like(logits, -np.inf)
        mask = ActionEncoder.legal_mask(legal_moves)
        masked[mask] = logits[mask]
        return masked
