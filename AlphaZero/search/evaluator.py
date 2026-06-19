"""局面评估器 — 接口定义 + 随机评估器"""
from typing import Protocol
import numpy as np
from ..engine.move import ActionEncoder
from ..engine.state import GameState


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
            # 无合法走法 → 已结束，不应调用 evaluate
            policy[:] = 0.0

        value = float(self.rng.uniform(-1.0, 1.0))
        return policy, value
