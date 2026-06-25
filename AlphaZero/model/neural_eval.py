"""神经网络评估器 — 用于 MCTS 搜索"""
import numpy as np
import torch
import torch.nn.functional as F

from ..engine.state import GameState
from ..engine.move import ActionEncoder
from ..engine.constants import POLICY_SIZE
from .network import PolicyWDLEncoder


class NeuralEvaluator:
    """神经网络评估器，实现 Evaluator 协议

    使用 PolicyWDLEncoder 评估局面，返回 masked policy 和 WDL value。
    """

    def __init__(self, model: PolicyWDLEncoder, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.eval()

    def evaluate(self, state: GameState) -> tuple[np.ndarray, float]:
        """评估单个局面

        Returns:
            policy_probs: (8100,) float32 合法动作概率
            value: float = P(win) - P(loss)
        """
        # 编码局面
        state_enc = state.encode()
        state_tensor = torch.from_numpy(state_enc).unsqueeze(0).to(self.device)

        # 获取合法动作掩码
        legal_mask = state.legal_mask()
        legal_tensor = torch.from_numpy(legal_mask).to(self.device)

        # 推理
        policy_probs, wdl_probs, value = self.model.evaluate_state(
            state_tensor, legal_tensor)

        return policy_probs, value

    def evaluate_batch(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        """批量评估局面

        Returns:
            policies: (B, 8100) float32
            values: (B,) float32
        """
        # 编码局面
        state_encs = np.stack([s.encode() for s in states])
        state_tensor = torch.from_numpy(state_encs).to(self.device)

        # 获取合法动作掩码
        legal_masks = np.stack([s.legal_mask() for s in states])
        legal_tensor = torch.from_numpy(legal_masks).to(self.device)

        # 推理
        policy_probs, wdl_probs, values = self.model.evaluate_batch(
            state_tensor, legal_tensor)

        return policy_probs, values
