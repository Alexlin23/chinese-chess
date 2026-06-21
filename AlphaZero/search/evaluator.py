"""局面评估器 — 接口定义 + 随机评估器 + 启发式评估器"""
from typing import Protocol
import numpy as np
from ..engine.move import ActionEncoder
from ..engine.state import GameState
from ..engine.constants import KING, ADVISOR, ELEPHANT, KNIGHT, ROOK, CANNON, PAWN


class Evaluator(Protocol):
    """神经网络评估器接口协议。

    所有评估器必须实现 evaluate()，返回 (policy_probs, value)。
    使用 Protocol 而非 ABC，允许任何兼容对象作为评估器。
    """

    def evaluate(self, state: GameState) -> tuple[np.ndarray, float]:
        """
        评估一个局面。

        Args:
            state: 待评估局面

        Returns:
            (policy_probs, value):
              - policy_probs: shape (POLICY_SIZE,) float32，走法先验概率
              - value: float [-1, +1]，局面评分
        """
        ...


class RandomEvaluator:
    """随机评估器 — 用于 MCTS 骨架测试。

    返回均匀分布的合法走法概率 + 随机价值。
    不学习，仅验证 MCTS 搜索流程正确性。
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def evaluate(self, state: GameState) -> tuple[np.ndarray, float]:
        """随机评估：合法走法等概率，价值随机。"""
        policy = np.zeros(ActionEncoder.POLICY_SIZE, dtype=np.float32)
        legal = ActionEncoder.legal_indices(state)
        if len(legal) > 0:
            policy[legal] = 1.0 / len(legal)
        else:
            policy[:] = 0.0

        value = float(self.rng.uniform(-1.0, 1.0))
        return policy, value


class HeuristicEvaluator:
    """启发式评估器 — 物质 + 位置 + 将军奖励 + 均匀策略。

    用于 AlphaZero 冷启动。
    提供足够强的价值信号，使 MCTS 在 50-100 模拟下
    就能找到杀棋路径，产出有真实终局结果的训练数据。
    """

    # ── 棋子基础价值 ──
    PIECE_VALUE = np.array([0, 0, 2.0, 2.0, 4.0, 9.0, 4.5, 1.0])
    # 索引: 0=空, 1=将, 2=士, 3=象, 4=马, 5=车, 6=炮, 7=兵

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def evaluate(self, state: GameState) -> tuple[np.ndarray, float]:
        """启发式评估 — 均匀策略 + 价值评分。"""
        policy = np.zeros(ActionEncoder.POLICY_SIZE, dtype=np.float32)
        legal = ActionEncoder.legal_indices(state)
        if len(legal) > 0:
            policy[legal] = 1.0 / len(legal)
        else:
            policy[:] = 0.0
        value = self.value(state)
        return policy, value

    def value(self, state: GameState) -> float:
        """快速价值评分 — 跳过 policy 计算，纯启发式评估。"""
        return self._evaluate_value(state)

    def _evaluate_value(self, state: GameState) -> float:
        """综合局面评估：物质 + 位置 + 移动力 + 将军奖励。"""
        board = state.board
        abs_board = np.abs(board)

        # ── 1. 物质得分 ──
        material = self.PIECE_VALUE[abs_board].sum()  # 总价值
        # 红方 material: board > 0 → +value; 黑方: board < 0 → -value
        red_score = np.sum(self.PIECE_VALUE[abs_board] * (board > 0))
        black_score = np.sum(self.PIECE_VALUE[abs_board] * (board < 0))
        total = red_score - black_score

        # ── 2. 兵/卒过河奖励 ──
        pawn_mask = abs_board == PAWN
        crossed = ((board > 0) & pawn_mask & (np.arange(10)[:, None] <= 4)) | \
                  ((board < 0) & pawn_mask & (np.arange(10)[:, None] >= 5))
        pawn_bonus = crossed.sum() * 1.0
        total += pawn_bonus * (1 if state.turn else -1)

        # ── 3. 将军奖励（鼓励探索将军局面） ──
        from ..engine.fast_chess import is_in_check
        opponent_color = not state.turn
        if is_in_check(board, opponent_color):
            # 当前方正在将军对方 → 大加分
            total += (8.0 if state.turn else -8.0)
        if is_in_check(board, state.turn):
            # 当前方被将军 → 大扣分
            total += (-8.0 if state.turn else 8.0)

        # ── 4. 位置奖励 (简化) ──
        # 车在开阔线 / 马在中心 等
        pos_bonus = self._position_bonus(board, state.turn)
        total += pos_bonus

        # 翻转到当前走棋方视角
        if not state.turn:
            total = -total

        return float(np.clip(total / 60.0, -1.0, 1.0))

    def _position_bonus(self, board: np.ndarray, turn: bool) -> float:
        """简化位置奖励。

        车在开放线 + 炮在对方半场 + 马在中心。
        """
        bonus = 0.0
        abs_b = np.abs(board)
        rows, cols = np.where(board != 0)

        for r, c in zip(rows, cols):
            cell = board[r, c]
            ptype = abs(cell)
            is_red = cell > 0

            if ptype == ROOK:
                # 车在开阔列（该列无兵/卒阻挡）
                col_vals = board[:, c]
                if np.sum((abs_b[:, c] == PAWN) & (col_vals * (1 if is_red else -1) < 0)) == 0:
                    bonus += 1.5 if is_red else -1.5

            elif ptype == KNIGHT:
                # 马在中心区域 [2..7, 2..6]
                if 2 <= r <= 7 and 2 <= c <= 6:
                    bonus += 0.8 if is_red else -0.8

            elif ptype == CANNON:
                # 炮在对方半场
                in_opponent_half = (is_red and r <= 4) or (not is_red and r >= 5)
                if in_opponent_half:
                    bonus += 0.6 if is_red else -0.6

            elif ptype == PAWN:
                # 中兵/卒加分
                if c in (3, 4, 5):
                    crossed = (is_red and r <= 4) or (not is_red and r >= 5)
                    if crossed:
                        bonus += 0.5 if is_red else -0.5

        return bonus
