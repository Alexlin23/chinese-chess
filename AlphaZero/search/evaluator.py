"""局面评估器 — 接口定义"""
from typing import Protocol
import numpy as np
from ..engine.state import GameState
from ..engine.constants import POLICY_SIZE


class Evaluator(Protocol):
    """评估器接口协议"""

    def evaluate(self, state: GameState) -> tuple[np.ndarray, float]:
        """评估一个局面。返回 (policy_probs, value)"""
        ...

    def evaluate_batch(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        """批量评估多个局面。"""
        policies = []
        values = []
        for s in states:
            p, v = self.evaluate(s)
            policies.append(p)
            values.append(v)
        return np.stack(policies), np.array(values, dtype=np.float32)
