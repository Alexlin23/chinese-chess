"""Reusable CUDA inference graphs for immutable evaluation snapshots."""
import torch


class CudaGraphForward:
    def __init__(self, model, precision="bf16", channels_last=True, max_batch=128,
                 stage_callback=None):
        if model.training or next(model.parameters()).device.type != "cuda":
            raise ValueError("CUDA graphs require an immutable CUDA eval snapshot")
        self.model, self.precision = model, precision
        self.channels_last, self.max_batch = channels_last, max_batch
        self.cache = {}
        self.stage_callback = stage_callback

    @torch.inference_mode()
    def __call__(self, x):
        count = len(x)
        if not 0 < count <= self.max_batch:
            raise ValueError("invalid inference batch")
        size = min(1 << (count - 1).bit_length(), self.max_batch)
        if size not in self.cache:
            if self.stage_callback: self.stage_callback("graph_allocate", size)
            static = torch.zeros((size, *x.shape[1:]), device=x.device, dtype=x.dtype)
            if self.channels_last:
                static = static.contiguous(memory_format=torch.channels_last)
            stream = torch.cuda.Stream()
            if self.stage_callback: self.stage_callback("graph_warmup", size)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.precision == "bf16", cache_enabled=False):
                        self.model(static)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            if self.stage_callback: self.stage_callback("graph_capture", size)
            with torch.cuda.graph(graph), torch.autocast("cuda", dtype=torch.bfloat16,
                    enabled=self.precision == "bf16", cache_enabled=False):
                outputs = self.model(static)
            self.cache[size] = static, graph, outputs
            if self.stage_callback: self.stage_callback("forward", size)
        static, graph, outputs = self.cache[size]
        static[:count].copy_(x)
        if count < size:
            static[count:].zero_()
        graph.replay()
        return tuple(output[:count] for output in outputs)
