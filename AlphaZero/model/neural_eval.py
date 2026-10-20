"""Current-player coordinates at the network boundary; absolute actions in the tree."""
from contextlib import nullcontext
import numpy as np
import torch

def autocast_context(device, precision):
    if str(device).startswith("cuda") and precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("requested BF16 is unsupported on this device")
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()

class NeuralEvaluator:
    def __init__(self, model, device="cpu", precision="fp32", channels_last=False, forward_model=None):
        self.model, self.device = model, device
        self.precision, self.channels_last = precision, channels_last
        self.forward_model = forward_model
        self.model.eval()

    def evaluate(self, state):
        policies, values = self.evaluate_batch([state])
        return policies[0], float(values[0])

    def evaluate_batch(self, states):
        if not states:
            raise ValueError("empty inference batch")
        tensor = torch.from_numpy(np.stack([state.encode() for state in states])).to(self.device)
        masks = torch.from_numpy(np.stack([state.canonical_mask() for state in states])).to(self.device)
        if self.channels_last:
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        with autocast_context(self.device, self.precision):
            policies, values = self.model.evaluate_batch(tensor, masks, forward_model=self.forward_model)
        for i, state in enumerate(states):
            if not state.turn:
                policies[i] = policies[i, ::-1].copy()
        return policies, values
