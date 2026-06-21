"""NeuralEvaluator — 神经网络评估器，对接 MCTS 搜索。

实现 Evaluator 协议接口，支持单局面和批量评估。
批量评估是 CPU 上性能的关键：一次 forward 处理整个 MCTS 节点的所有子节点。
"""
import numpy as np
import torch
import torch.nn.functional as F

from ..engine.move import ActionEncoder
from ..engine.state import GameState
from .network import AlphaZeroNet


class NeuralEvaluator:
    """神经网络局面评估器。

    封装 PyTorch 模型，将 GameState 编码为网络输入，
    返回策略概率分布和价值评分。

    MCTS 调用流程:
      1. search() 扩展节点时调用 evaluate_batch(states) → (policies, values)
      2. 或单个 evaluate(state) → (policy, value)
    """

    def __init__(self, model: AlphaZeroNet, device: str = 'cpu',
                 use_amp: bool = False):
        """
        Args:
            model:   AlphaZeroNet 实例
            device:  'cpu' 或 'cuda'
            use_amp: CPU 上建议 False（AMP 对 CPU 无效）
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()  # 评估器始终 eval 模式
        self.use_amp = use_amp

        # 预分配 batch tensor，避免重复分配
        self._batch_size = 0
        self._batch_tensor = None

    # ── Evaluator 协议 ──

    def evaluate(self, state: GameState) -> tuple[np.ndarray, float]:
        """评估单个局面。

        Args:
            state: GameState 快照

        Returns:
            (policy_probs, value):
              - policy_probs: (POLICY_SIZE,) float32 — 含非法走法掩码
              - value: float — [-1, +1]
        """
        encoded = state.encode()  # (18, 10, 9) float32
        tensor = torch.from_numpy(encoded).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits, value = self.model(tensor)

        policy = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        return policy.astype(np.float32), float(value.item())

    def evaluate_batch(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        """批量评估多个局面（MCTS 节点扩展用）。

        一次 forward pass 处理所有子节点，利用 MKL 并行加速。
        这是 CPU 上 MCTS 性能的关键优化点。

        Args:
            states: GameState 列表

        Returns:
            (policies, values):
              - policies: (B, POLICY_SIZE) float32
              - values:   (B,) float32
        """
        if not states:
            return (np.empty((0, ActionEncoder.POLICY_SIZE), dtype=np.float32),
                    np.empty((0,), dtype=np.float32))

        # 批量编码
        batch_enc = np.stack([s.encode() for s in states])
        tensor = torch.from_numpy(batch_enc).to(self.device)

        with torch.inference_mode():
            logits, values = self.model(tensor)

        policies = F.softmax(logits, dim=1).cpu().numpy()
        return policies.astype(np.float32), values.cpu().numpy().astype(np.float32)

    def _ensure_batch_buffer(self, size: int):
        """预分配 batch tensor 以复用内存。"""
        if self._batch_tensor is None or self._batch_size < size:
            self._batch_size = size
            self._batch_tensor = torch.empty(
                size, 18, 10, 9, dtype=torch.float32, device=self.device)

    # ── 权重管理 ──

    def sync_weights_from(self, model: AlphaZeroNet) -> None:
        """从另一个模型同步权重（训练后更新评估用模型）。"""
        self.model.load_state_dict(model.state_dict())
        self.model.eval()

    def save_checkpoint(self, path: str, **extra) -> None:
        """保存评估器模型权重。"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            **extra,
        }, path)

    def load_checkpoint(self, path: str) -> dict:
        """加载模型权重。返回额外信息字典。"""
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state['model_state_dict'])
        self.model.eval()
        return {k: v for k, v in state.items() if k != 'model_state_dict'}

    @property
    def num_parameters(self) -> int:
        """模型参数量"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


# ── 兼容旧接口：工厂函数 ──

def create_evaluator(num_blocks: int = 20, num_filters: int = 128,
                     checkpoint: str = None, device: str = 'cpu',
                     seed: int = 42) -> NeuralEvaluator:
    """创建 NeuralEvaluator 实例。

    Args:
        num_blocks:  残差块数
        num_filters: 通道数
        checkpoint:  预训练权重路径
        device:      'cpu'
        seed:        随机种子

    Returns:
        NeuralEvaluator
    """
    torch.manual_seed(seed)
    model = AlphaZeroNet(num_blocks=num_blocks, num_filters=num_filters)
    if checkpoint:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state['model_state_dict'])
    return NeuralEvaluator(model, device=device)
