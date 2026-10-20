"""One CUDA inference owner, cross-game batching and atomic actor refresh."""
from dataclasses import dataclass
from queue import Empty, Full
import multiprocessing as mp
import time
import numpy as np
import torch
from .config import AlphaZeroConfig
from .trainer import create_model
from ..model.neural_eval import autocast_context
from ..model.cuda_graph import CudaGraphForward
from ..engine.constants import POLICY_SIZE

@dataclass
class BatchRequest:
    request_id: int
    states: np.ndarray
    masks: np.ndarray
    worker_id: int
    model_step: object = None

@dataclass
class BatchResponse:
    request_id: int
    policies: object
    values: object
    worker_id: int
    model_step: int = 0
    error: object = None

class InferenceServer:
    def __init__(self, model_path, config, context=None, run_id=None):
        self.config = config
        self.model_path = str(model_path)
        self.run_id = run_id
        self.context = context or mp.get_context("spawn")
        self.request_queue = self.context.Queue(maxsize=config.num_workers * 2)
        self.response_queues = {i:self.context.Queue(maxsize=2) for i in range(config.num_workers)}
        self.ready = self.context.Queue(maxsize=1)
        self.refresh_queue = self.context.Queue(maxsize=1)
        self.stop_event = self.context.Event()
        self._process = None

    def start(self):
        self._process = self.context.Process(target=_serve, args=(
            self.model_path, self.config, self.request_queue, self.response_queues,
            self.ready, self.refresh_queue, self.stop_event, self.run_id), daemon=True)
        self._process.start()
        try:
            status = self.ready.get(timeout=90)
        except Empty as exc:
            self.stop()
            raise RuntimeError("inference service did not become ready") from exc
        if status.get("error"):
            self.stop()
            raise RuntimeError(status["error"])
        return status

    def refresh(self):
        try:
            self.refresh_queue.put_nowait(self.model_path)
        except Full:
            pass  # Existing request loads the newest atomically published file.

    def stop(self):
        self.stop_event.set()
        if self._process:
            self._process.join(5)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(5)
        for queue in [self.request_queue, self.ready, self.refresh_queue, *self.response_queues.values()]:
            queue.cancel_join_thread()
            queue.close()

def _serve(path, config, requests, responses, ready, refresh, stop, run_id):
    torch.set_num_threads(config.torch_threads)
    pending = None
    announced = False
    try:
        models, active_versions = {}, {}
        def load():
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint.get("kind") != "actor" or checkpoint.get("fingerprint") != config.fingerprint:
                raise ValueError("inference actor specification mismatch")
            if run_id is not None and checkpoint.get("run_id") != run_id:
                raise ValueError("inference actor belongs to another run")
            version = int(checkpoint["step"])
            if version not in models:
                model = create_model(config, config.device)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                if config.device == "cuda" and config.inference_cuda_graph:
                    models[version] = CudaGraphForward(model, config.precision,
                        config.channels_last, config.inference_batch_size)
                else:
                    models[version] = torch.compile(model) if config.compile_model else model
            return version
        step = load()
        ready.put(dict(step=step, device=config.device, precision=config.precision))
        announced = True
        while not stop.is_set():
            try:
                refresh.get_nowait()
                step = load()
            except Empty:
                pass
            try:
                request = pending if pending is not None else requests.get(timeout=0.1)
                pending = None
            except Empty:
                continue
            batch = [request]
            count = len(request.states)
            deadline = time.monotonic() + config.inference_wait_ms / 1000
            while count < config.inference_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = requests.get(timeout=remaining)
                except Empty:
                    break
                if count + len(item.states) > config.inference_batch_size:
                    pending = item
                    break
                batch.append(item)
                count += len(item.states)
            try:
                for item in batch:
                    n = len(item.states)
                    if (not 0 < n <= config.inference_batch_size
                            or item.worker_id not in responses
                            or item.states.shape != (n, config.input_channels, 10, 9)
                            or item.masks.shape != (n, POLICY_SIZE)
                            or item.states.dtype != np.float32 or item.masks.dtype != np.bool_
                            or not np.isfinite(item.states).all()
                            or not item.masks.any(axis=1).all()):
                        raise ValueError("invalid inference request")
                groups = {}
                for item in batch:
                    version = step if item.model_step is None else item.model_step
                    if version not in models:
                        raise ValueError("requested actor snapshot has expired")
                    active_versions[item.worker_id] = version
                    groups.setdefault(version, []).append(item)
                outgoing = []
                for version, group in groups.items():
                    array = torch.from_numpy(np.concatenate([x.states for x in group]))
                    masks = torch.from_numpy(np.concatenate([x.masks for x in group])).to(config.device)
                    if config.device == "cuda":
                        array = array.pin_memory()
                    array = array.to(config.device, non_blocking=True)
                    if config.channels_last and config.device == "cuda":
                        array = array.contiguous(memory_format=torch.channels_last)
                    with torch.inference_mode(), autocast_context(config.device, config.precision):
                        logits, values = models[version](array)
                        raw_finite = torch.isfinite(logits).all()
                        policies = logits.float().masked_fill(~masks, -torch.inf).softmax(1).cpu().numpy()
                        values = values.float().cpu().numpy()
                    if (not bool(raw_finite) or not np.isfinite(policies).all() or not np.isfinite(values).all()
                            or np.any(np.abs(values) > 1.00001)
                            or not np.allclose(policies.sum(1), 1, atol=1e-5)):
                        raise FloatingPointError("non-finite or unnormalized neural output")
                    offset = 0
                    for item in group:
                        n = len(item.states)
                        outgoing.append(BatchResponse(item.request_id, policies[offset:offset+n],
                            values[offset:offset+n], item.worker_id, version))
                        offset += n
                for response in outgoing:
                    responses[response.worker_id].put(response, timeout=5)
                keep = set(active_versions.values()) | {step}
                for expired in list(models):
                    if expired not in keep:
                        del models[expired]
            except Exception as exc:
                for item in batch:
                    try:
                        responses[item.worker_id].put(BatchResponse(
                            item.request_id, None, None, item.worker_id, step, repr(exc)), timeout=1)
                    except (Full, KeyError):
                        pass
                raise
    except BaseException as exc:
        if not announced:
            ready.put(dict(error=repr(exc)))
        stop.set()
        raise
