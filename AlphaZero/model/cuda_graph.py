"""Reusable CUDA inference graphs for immutable evaluation snapshots."""
import torch
import numpy as np


class CudaGraphForward:
    def __init__(self, model, precision="bf16", channels_last=True, max_batch=128):
        if model.training or next(model.parameters()).device.type != "cuda":
            raise ValueError("CUDA graphs require an immutable CUDA eval snapshot")
        self.model, self.precision = model, precision
        self.channels_last, self.max_batch = channels_last, max_batch
        self.cache = {}

    @torch.inference_mode()
    def __call__(self, x):
        count = len(x)
        if not 0 < count <= self.max_batch:
            raise ValueError("invalid inference batch")
        size = min(1 << (count - 1).bit_length(), self.max_batch)
        if size not in self.cache:
            static = torch.zeros((size, *x.shape[1:]), device=x.device, dtype=x.dtype)
            if self.channels_last:
                static = static.contiguous(memory_format=torch.channels_last)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.precision == "bf16", cache_enabled=False):
                        self.model(static)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), torch.autocast("cuda", dtype=torch.bfloat16,
                    enabled=self.precision == "bf16", cache_enabled=False):
                outputs = self.model(static)
            self.cache[size] = static, graph, outputs
        static, graph, outputs = self.cache[size]
        static[:count].copy_(x)
        if count < size:
            static[count:].zero_()
        graph.replay()
        return tuple(output[:count] for output in outputs)


class CudaGraphPolicy:
    """Capture masked policy/value postprocessing and use one host transfer.

    The last output column carries raw-logit validity, so masking an invalid
    illegal action cannot conceal a broken network. Returned storage is reused
    on the next call; callers must copy it to the host before replaying again.
    """
    def __init__(self, model, precision="bf16", channels_last=True, max_batch=128):
        if model.training or next(model.parameters()).device.type != "cuda":
            raise ValueError("CUDA graphs require an immutable CUDA eval snapshot")
        self.model, self.precision = model, precision
        self.channels_last, self.max_batch = channels_last, max_batch
        self.cache = {}
        self.host_cache = {}

    @torch.inference_mode()
    def prepare_host(self, state_batches, mask_batches):
        """Stage directly into persistent graph inputs, without intermediate GPU tensors."""
        count = sum(len(states) for states in state_batches)
        if not 0 < count <= self.max_batch:
            raise ValueError("invalid inference batch")
        size = min(1 << (count - 1).bit_length(), self.max_batch)
        if size not in self.host_cache:
            device = next(self.model.parameters()).device
            shape = (size, *state_batches[0].shape[1:])
            policy_size = mask_batches[0].shape[1]
            # Capture uses the same graph implementation as the device-input API.
            self(torch.zeros(shape, device=device),
                 torch.ones((size, policy_size), dtype=torch.bool, device=device))
            states = torch.empty(shape, pin_memory=True,
                memory_format=torch.channels_last if self.channels_last else torch.contiguous_format)
            masks = torch.empty((size, policy_size), dtype=torch.bool, pin_memory=True)
            output = torch.empty((size, policy_size + 2), pin_memory=True)
            self.host_cache[size] = states, masks, output, states.numpy(), masks.numpy()
        host_states, host_masks, host_output, states_view, masks_view = self.host_cache[size]
        offset = 0
        for states, masks in zip(state_batches, mask_batches):
            end = offset + len(states)
            np.copyto(states_view[offset:end], states)
            np.copyto(masks_view[offset:end], masks)
            offset = end
        if count < size:
            states_view[count:].fill(0)
            masks_view[count:].fill(True)
        static, static_masks, graph, output = self.cache[size]
        static.copy_(host_states, non_blocking=True)
        static_masks.copy_(host_masks, non_blocking=True)
        return count, graph, output, host_output

    @torch.inference_mode()
    def run_host(self, prepared, gpu_sample=None):
        count, graph, output, host_output = prepared
        graph.replay()
        if gpu_sample is not None:
            gpu_sample.mark()
        host_output.copy_(output, non_blocking=True)
        torch.cuda.current_stream(output.device).synchronize()
        # Queue feeders may serialize after the next replay; give them owned data.
        return host_output.numpy()[:count].copy()

    def _forward(self, states, masks):
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=self.precision == "bf16", cache_enabled=False):
            logits, values = self.model(states)
        finite = torch.isfinite(logits).all(dim=1, keepdim=True)
        policies = logits.float().masked_fill(~masks, -torch.inf).softmax(1)
        return torch.cat((policies, values.float().reshape(-1, 1), finite.float()), dim=1)

    @torch.inference_mode()
    def __call__(self, states, masks):
        count = len(states)
        if not 0 < count <= self.max_batch or len(masks) != count:
            raise ValueError("invalid inference batch")
        size = min(1 << (count - 1).bit_length(), self.max_batch)
        if size not in self.cache:
            static = torch.zeros((size, *states.shape[1:]), device=states.device, dtype=states.dtype)
            if self.channels_last:
                static = static.contiguous(memory_format=torch.channels_last)
            # Padding has a nonempty mask; it is sliced off before validation.
            static_masks = torch.ones((size, masks.shape[1]), device=masks.device, dtype=torch.bool)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._forward(static, static_masks)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self._forward(static, static_masks)
            self.cache[size] = static, static_masks, graph, output
        static, static_masks, graph, output = self.cache[size]
        static[:count].copy_(states)
        static_masks[:count].copy_(masks)
        if count < size:
            static[count:].zero_()
            static_masks[count:].fill_(True)
        graph.replay()
        return output[:count]
