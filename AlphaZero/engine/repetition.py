"""重复局面检测 — 支持三次重复平局"""
import numpy as np
from .constants import ROWS, COLS


def position_key(board: np.ndarray, turn: bool) -> bytes:
    """生成局面哈希 key，包含棋盘和轮次信息

    Args:
        board: (10, 9) int8 棋盘
        turn: True=红方走, False=黑方走

    Returns:
        bytes: 可哈希的局面 key
    """
    turn_byte = b"r" if turn else b"b"
    return board.tobytes() + turn_byte


def update_repetition(counts: dict, key: bytes) -> dict:
    """更新重复计数，返回新字典（不修改原字典）

    Args:
        counts: {key: count} 字典
        key: 局面 key

    Returns:
        新的 counts 字典
    """
    new_counts = counts.copy()
    new_counts[key] = new_counts.get(key, 0) + 1
    return new_counts


def is_repetition_draw(counts: dict, key: bytes, limit: int = 3) -> bool:
    """检查当前局面是否达到重复次数限制

    Args:
        counts: {key: count} 字典
        key: 当前局面 key
        limit: 重复次数限制，默认 3

    Returns:
        bool: 是否达到限制
    """
    return counts.get(key, 0) >= limit
