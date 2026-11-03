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
            policies[i] = state.absolute_policy(policies[i])
        return policies, values


class CudaGraphEvaluator(NeuralEvaluator):
    """Local play uses the same complete policy graph and staging buffers as self-play.

    Calls are serialized by the owner (PlaySession.model_lock). Returned arrays
    own their storage, including across later graph replays or model reloads.
    """
    def __init__(self, model, device="cuda", precision="bf16", channels_last=True):
        super().__init__(model, device, precision, channels_last)
        from .cuda_graph import CudaGraphPolicy
        self.graph = CudaGraphPolicy(model, precision, channels_last)

    def evaluate_batch(self, states):
        if not states:
            raise ValueError("empty inference batch")
        if self.model.training:
            raise RuntimeError("inference requires eval mode")
        arrays = [state.encode()[None] for state in states]
        masks = [state.canonical_mask()[None] for state in states]
        if any(not mask.any() for mask in masks):
            raise ValueError("terminal/empty legal set sent for neural inference")
        host = self.graph.run_host(self.graph.prepare_host(arrays, masks))
        policies, values = host[:, :-2], host[:, -2]
        if (not (host[:, -1] == 1).all() or not np.isfinite(policies).all()
                or not np.isfinite(values).all()):
            raise FloatingPointError("non-finite neural output")
        for i, state in enumerate(states):
            policies[i] = state.absolute_policy(policies[i])
        return policies, values
