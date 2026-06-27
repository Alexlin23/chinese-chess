"""基于子力估值的启发式 AI

只计算棋子价值差，选择最优走法。
棋子价值参考中国象棋标准估值。
"""
import random
import numpy as np
from .constants import PIECE_VALUES


def material_score(board: np.ndarray, is_red: bool) -> float:
    """计算当前方子力优势。

    Args:
        board: (10, 9) int8 棋盘数组
        is_red: True=红方视角, False=黑方视角

    Returns:
        子力优势值（己方总价值 - 对方总价值）
    """
    abs_board = np.abs(board)
    values = PIECE_VALUES[abs_board]
    red_total = values[board > 0].sum()
    black_total = values[board < 0].sum()
    if is_red:
        return float(red_total - black_total)
    else:
        return float(black_total - red_total)


def heuristic_move(state) -> object:
    """选择子力得分最高的走法，同分时随机选择。

    Args:
        state: GameState 当前局面

    Returns:
        Move | None: 最佳走法，无合法走法时返回 None
    """
    legal_moves = state.legal_moves()
    if not legal_moves:
        return None

    best_score = -float('inf')
    best_moves = []

    for move in legal_moves:
        new_state = state.apply(move)
        # 从当前走棋方的视角评估子力优势
        score = material_score(new_state.board, state.turn)

        if score > best_score:
            best_score = score
            best_moves = [move]
        elif abs(score - best_score) < 1e-6:
            best_moves.append(move)

    return random.choice(best_moves)
