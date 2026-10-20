"""AlphaZero residual tower with policy logits and scalar tanh value."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..engine.constants import POLICY_PLANES, INPUT_CHANNELS

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        return F.relu(x + self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))

class PolicyValueNetwork(nn.Module):
    def __init__(self, num_blocks=19, num_filters=128, input_channels=INPUT_CHANNELS):
        super().__init__()
        self.num_blocks, self.num_filters = num_blocks, num_filters
        self.conv_input = nn.Conv2d(input_channels, num_filters, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_filters)
        self.resblocks = nn.Sequential(*(ResBlock(num_filters) for _ in range(num_blocks)))
        # AlphaZero chess/shogi spatial head, adapted to Xiangqi's 50 moves.
        # Keep the hardware-sized tower width in the first 3x3 convolution.
        # Final outputs are raw logits: no BN, ReLU or softmax after them.
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters), nn.ReLU(),
            nn.Conv2d(num_filters, POLICY_PLANES, 3, padding=1),
        )
        self.value_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(90, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.resblocks(F.relu(self.bn_input(self.conv_input(x))))
        p = self.policy_head(x).flatten(1)  # [N, 50, 10, 9] -> [N, 4500]
        v = F.relu(self.value_bn(self.value_conv(x))).flatten(1)
        v = torch.tanh(self.value_fc2(F.relu(self.value_fc1(v)))).squeeze(-1)
        return p, v

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    @torch.inference_mode()
    def evaluate_batch(self, states, legal_masks=None, forward_model=None):
        if self.training:
            raise RuntimeError("inference requires eval mode")
        logits, values = (forward_model if forward_model is not None else self)(states)
        logits = logits.float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite neural logits")
        if legal_masks is not None:
            if not legal_masks.any(dim=1).all():
                raise ValueError("terminal/empty legal set sent for neural inference")
            logits = logits.masked_fill(~legal_masks, -torch.inf)
        policies = logits.softmax(1).cpu().numpy()
        values = values.float().cpu().numpy()
        if not np.isfinite(policies).all() or not np.isfinite(values).all():
            raise FloatingPointError("non-finite neural output")
        # Middle return retained for old callers; it is NOT a learned WDL head.
        return policies, None, values

    def evaluate_state(self, state, legal_mask=None):
        if state.ndim == 3:
            state = state.unsqueeze(0)
        if legal_mask is not None and legal_mask.ndim == 1:
            legal_mask = legal_mask.unsqueeze(0)
        policies, _, values = self.evaluate_batch(state, legal_mask)
        return policies[0], None, float(values[0])

# Import compatibility only. Old WDL checkpoints are intentionally incompatible.
PolicyWDLEncoder = PolicyValueNetwork
