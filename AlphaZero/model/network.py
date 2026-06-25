"""AlphaZero Policy-WDL 残差网络 — 中国象棋版

架构:
  输入: (B, 18, 10, 9) — GameState.encode() 输出
  主体: ResBlock × N
  Policy Head: Conv1×1 → FC → 8100 logits
  WDL Head: Conv1×1 → FC → 3 logits (win/draw/loss)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..engine.constants import POLICY_SIZE, INPUT_CHANNELS, WDL_SIZE


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


class PolicyWDLEncoder(nn.Module):
    """Policy-WDL 双头残差网络

    Args:
        num_blocks:  残差块数量
        num_filters: 卷积层通道数
        input_channels: 输入通道数（固定 18）
    """

    def __init__(self,
                 num_blocks: int = 8,
                 num_filters: int = 128,
                 input_channels: int = INPUT_CHANNELS):
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
        # Conv1×1 降维 → FC → 8100 logits
        self.policy_conv = nn.Conv2d(num_filters, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        policy_flat_size = 32 * 10 * 9  # 2880
        self.policy_fc1 = nn.Linear(policy_flat_size, 512)
        self.policy_fc2 = nn.Linear(512, POLICY_SIZE)

        # ── WDL Head ──
        # Conv1×1 → FC → 3 logits
        self.wdl_conv = nn.Conv2d(num_filters, 32, 1, bias=False)
        self.wdl_bn = nn.BatchNorm2d(32)
        wdl_flat_size = 32 * 10 * 9  # 2880
        self.wdl_fc1 = nn.Linear(wdl_flat_size, 256)
        self.wdl_fc2 = nn.Linear(256, WDL_SIZE)

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
            (policy_logits, wdl_logits):
              - policy_logits: (B, 8100) — 走法 logits
              - wdl_logits:    (B, 3)    — win/draw/logits
        """
        # ── 输入卷积 ──
        x = F.relu(self.bn_input(self.conv_input(x)))

        # ── 残差塔 ──
        x = self.resblocks(x)

        # ── Policy Head ──
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(1)               # (B, 2880)
        p = F.relu(self.policy_fc1(p))  # (B, 512)
        p_logits = self.policy_fc2(p)   # (B, 8100)

        # ── WDL Head ──
        w = F.relu(self.wdl_bn(self.wdl_conv(x)))
        w = w.flatten(1)               # (B, 2880)
        w = F.relu(self.wdl_fc1(w))    # (B, 256)
        w_logits = self.wdl_fc2(w)     # (B, 3)

        return p_logits, w_logits

    def count_parameters(self) -> int:
        """可训练参数总数"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def evaluate_state(self, state_encoding: torch.Tensor,
                       legal_mask: torch.Tensor = None) -> tuple:
        """评估单个局面，返回 (policy_probs, wdl_probs, value)。

        Args:
            state_encoding: (1, 18, 10, 9) 或 (18, 10, 9)
            legal_mask: (8100,) bool 可选，合法动作掩码

        Returns:
            policy_probs: (8100,) numpy float32
            wdl_probs: (3,) numpy float32
            value: float = P(win) - P(loss)
        """
        was_training = self.training
        self.eval()
        if state_encoding.dim() == 3:
            state_encoding = state_encoding.unsqueeze(0)
        p_logits, w_logits = self.forward(state_encoding)

        # Policy: 应用合法动作掩码后 softmax (clone 避免 inplace 问题)
        p_logits = p_logits.squeeze(0).clone()  # (8100,)
        if legal_mask is not None:
            p_logits[~legal_mask] = float('-inf')
        policy_probs = F.softmax(p_logits, dim=0).cpu().numpy()

        # WDL: softmax
        wdl_probs = F.softmax(w_logits.squeeze(0), dim=0).cpu().numpy()

        # Value = P(win) - P(loss)
        value = float(wdl_probs[0] - wdl_probs[2])

        if was_training:
            self.train()
        return policy_probs, wdl_probs, value

    @torch.no_grad()
    def evaluate_batch(self, state_batch: torch.Tensor,
                       legal_masks: torch.Tensor = None) -> tuple:
        """批量评估局面。

        Args:
            state_batch: (B, 18, 10, 9)
            legal_masks: (B, 8100) bool 可选

        Returns:
            policy_probs: (B, 8100) numpy float32
            wdl_probs: (B, 3) numpy float32
            values: (B,) numpy float32
        """
        was_training = self.training
        self.eval()
        p_logits, w_logits = self.forward(state_batch)

        # Policy: 应用掩码后 softmax (clone 避免 inplace 问题)
        p_logits = p_logits.clone()
        if legal_masks is not None:
            p_logits[~legal_masks] = float('-inf')
        policy_probs = F.softmax(p_logits, dim=1).cpu().numpy()

        # WDL: softmax
        wdl_probs = F.softmax(w_logits, dim=1).cpu().numpy()

        # Value = P(win) - P(loss)
        values = wdl_probs[:, 0] - wdl_probs[:, 2]

        if was_training:
            self.train()
        return policy_probs, wdl_probs, values
