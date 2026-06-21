"""AlphaZero 残差网络 — 中国象棋版

架构 (AlphaZero 论文标准):
  输入: (B, 18, 10, 9) — GameState.encode() 输出
  主体: 20 个 ResBlock（3×3 卷积 + 跳跃连接）
  Policy Head: Conv1×1 → FC → 走法概率分布
  Value Head:  Conv1×1 → FC → tanh 评分

CPU 优化:
  - 默认 filters=128 (原版 256)，参数量降 4 倍
  - Policy Head 中间层压缩 (32×10×9 → 256 → 2086)
  - 全卷积 BatchNorm 融合（推理时可用 torch.jit.script）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..engine.move import ActionEncoder

POLICY_SIZE = ActionEncoder.POLICY_SIZE  # 2086


class ResBlock(nn.Module):
    """残差块：Conv3×3 → BN → ReLU → Conv3×3 → BN → SkipAdd → ReLU"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class AlphaZeroNet(nn.Module):
    """AlphaZero 风格的双头残差网络。

    Args:
        num_blocks:  残差块数量（原版 20，调试可用 5）
        num_filters: 卷积层通道数（原版 256，CPU 版默认 128）
        input_channels: 输入通道数（固定 18）
    """

    def __init__(self,
                 num_blocks: int = 20,
                 num_filters: int = 128,
                 input_channels: int = 18):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_filters = num_filters

        # ── 输入卷积 ──
        self.conv_input = nn.Conv2d(
            input_channels, num_filters, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_filters)

        # ── 残差塔 ──
        self.resblocks = nn.Sequential(*[
            ResBlock(num_filters) for _ in range(num_blocks)
        ])

        # ── Policy Head ──
        # Conv1×1 降维 → FC(256) → FC(POLICY_SIZE)
        self.policy_conv = nn.Conv2d(num_filters, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        policy_flat_size = 32 * 10 * 9  # 2880
        self.policy_fc1 = nn.Linear(policy_flat_size, 256)
        self.policy_fc2 = nn.Linear(256, POLICY_SIZE)

        # ── Value Head ──
        # Conv1×1 → FC(128) → FC(1) → tanh
        self.value_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(10 * 9, 128)
        self.value_fc2 = nn.Linear(128, 1)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """Kaiming 初始化 + BatchNorm 初始化为 1/0"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 18, 10, 9) float32 — 棋盘编码

        Returns:
            (policy_logits, value):
              - policy_logits: (B, POLICY_SIZE) — 走法 logits（softmax 在外部）
              - value:         (B,) float32 — 局面评分 [-1, +1]
        """
        # ── 输入卷积 ──
        x = F.relu(self.bn_input(self.conv_input(x)))

        # ── 残差塔 ──
        x = self.resblocks(x)

        # ── Policy Head ──
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(1)               # (B, 2880)
        p = F.relu(self.policy_fc1(p))  # (B, 256)
        p_logits = self.policy_fc2(p)   # (B, 2086) — logits

        # ── Value Head ──
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(1)               # (B, 90)
        v = F.relu(self.value_fc1(v))  # (B, 128)
        v = torch.tanh(self.value_fc2(v))  # (B, 1) → [-1, +1]

        return p_logits, v.squeeze(-1)

    def count_parameters(self) -> int:
        """可训练参数总数"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def evaluate_state(self, state_encoding: torch.Tensor) -> tuple:
        """评估单个局面，返回 (policy_probs, value)。

        Args:
            state_encoding: (1, 18, 10, 9) 或 (18, 10, 9)

        Returns:
            policy_probs: (POLICY_SIZE,) numpy float32
            value: float
        """
        was_training = self.training
        self.eval()
        if state_encoding.dim() == 3:
            state_encoding = state_encoding.unsqueeze(0)
        with torch.inference_mode():
            logits, value = self.forward(state_encoding)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        if was_training:
            self.train()
        return probs, float(value.item())

    @torch.no_grad()
    def evaluate_batch(self, state_batch: torch.Tensor) -> tuple:
        """批量评估局面。

        Args:
            state_batch: (B, 18, 10, 9)

        Returns:
            policy_probs: (B, POLICY_SIZE) numpy float32
            values: (B,) numpy float32
        """
        was_training = self.training
        self.eval()
        with torch.inference_mode():
            logits, values = self.forward(state_batch)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        if was_training:
            self.train()
        return probs, values.cpu().numpy()

    def to_torchscript(self):
        """导出为 TorchScript（推理加速 20-30%）。"""
        example = torch.randn(1, 18, 10, 9)
        return torch.jit.trace(self.eval(), example)


def create_model(num_blocks: int = 20, num_filters: int = 128,
                 checkpoint: str = None) -> AlphaZeroNet:
    """工厂函数：创建模型并可选加载权重。

    Args:
        num_blocks:  残差块数量
        num_filters: 通道数
        checkpoint:  权重文件路径（可选）

    Returns:
        AlphaZeroNet 实例
    """
    model = AlphaZeroNet(num_blocks=num_blocks, num_filters=num_filters)
    if checkpoint:
        state = torch.load(checkpoint, map_location='cpu',
                          weights_only=True)
        model.load_state_dict(state['model_state_dict'])
        print(f"加载模型: {checkpoint} "
              f"(epoch {state.get('epoch', '?')}, "
              f"loss {state.get('loss', '?'):.4f})")
    return model


# ── 自检 ──
if __name__ == "__main__":
    m = AlphaZeroNet(num_blocks=20, num_filters=128)
    print(f"参数数量: {m.count_parameters():,}")
    x = torch.randn(4, 18, 10, 9)
    p, v = m(x)
    print(f"输入: {x.shape} → policy: {p.shape}, value: {v.shape}")
    print(f"Value range: [{v.min().item():.3f}, {v.max().item():.3f}]")

    # Benchmark
    import time
    m.eval()
    # warmup
    for _ in range(30):
        with torch.inference_mode():
            m(torch.randn(1, 18, 10, 9))
    t0 = time.perf_counter()
    for _ in range(100):
        with torch.inference_mode():
            m(torch.randn(1, 18, 10, 9))
    t1 = time.perf_counter()
    print(f"\n单次推理 (inference_mode): {(t1-t0)/100*1000:.1f}ms")
    # batch=16
    t0 = time.perf_counter()
    for _ in range(50):
        with torch.inference_mode():
            m(torch.randn(16, 18, 10, 9))
    t1 = time.perf_counter()
    print(f"Batch16 推理: {(t1-t0)/50*1000:.1f}ms ({((t1-t0)/50/16*1000):.1f}ms/样本)")
