"""One CUDA inference owner, cross-game batching and atomic actor refresh."""
from dataclasses import dataclass
from queue import Empty, Full
import multiprocessing as mp
import time
import os
import traceback
import numpy as np
import torch
from .trainer import create_model
from ..model.neural_eval import autocast_context
from ..model.cuda_graph import CudaGraphForward
from ..engine.constants import POLICY_SIZE
from .performance import Recorder


def report_failure(queue, source, exc, **details):
    """Deliver an actual child-process exception to its supervising learner."""
    if queue is None:
        return
    message = dict(source=source, pid=os.getpid(), error=repr(exc),
                   traceback=traceback.format_exc(), **details)
    try:
        queue.put(message, timeout=1)
    except Full:
        print("Worker failure: " + str(message), flush=True)

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
    def __init__(self, model_path, config, context=None, run_id=None,
                 performance_queue=None, performance_options=None):
        self.config = config
        self.model_path = str(model_path)
        self.run_id = run_id
        self.context = context or mp.get_context("spawn")
        self.request_queue = self.context.Queue(maxsize=config.num_workers * 2)
        self.response_queues = {i:self.context.Queue(maxsize=2) for i in range(config.num_workers)}
        self.ready = self.context.Queue(maxsize=1)
        self.refresh_queue = self.context.Queue(maxsize=1)
        self.stop_event = self.context.Event()
        self.failures = self.context.Queue(maxsize=config.num_workers + 4)
        self._process = None
        self.performance_queue, self.performance_options = performance_queue, performance_options

    def start(self):
        self._process = self.context.Process(target=_serve, args=(
            self.model_path, self.config, self.request_queue, self.response_queues,
            self.ready, self.refresh_queue, self.stop_event, self.run_id,
            self.performance_queue, self.performance_options, self.failures), daemon=True)
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
        for queue in [self.request_queue, self.ready, self.refresh_queue, self.failures, *self.response_queues.values()]:
            queue.cancel_join_thread()
            queue.close()

def _serve(path, config, requests, responses, ready, refresh, stop, run_id,
           performance_queue=None, performance_options=None, failure_queue=None):
    torch.set_num_threads(config.torch_threads)
    performance = (Recorder("inference", performance_queue, **(performance_options or {}))
                   if performance_queue is not None else None)
    pending = None
    announced = False
    try:
        models, active_versions = {}, {}
        def load():
            if performance is not None:
                load_started = time.perf_counter()
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
            if performance is not None:
                performance.observe("data.actor_load_wall", time.perf_counter() - load_started)
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
                if performance is not None:
                    idle_started = time.perf_counter()
                request = pending if pending is not None else requests.get(timeout=0.1)
                pending = None
            except Empty:
                if performance is not None:
                    performance.observe("inference.idle_wait_wall", time.perf_counter() - idle_started)
                    performance.flush()
                continue
            if performance is not None:
                performance.observe("inference.idle_wait_wall", time.perf_counter() - idle_started)
                batch_started = time.perf_counter()
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
            if performance is not None:
                performance.observe("inference.batch_wait_wall", time.perf_counter() - batch_started)
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
                    graph_cache = getattr(models[version], "cache", None)
                    real_batch = sum(len(item.states) for item in group)
                    executed_batch = (min(1 << (real_batch - 1).bit_length(), config.inference_batch_size)
                                      if graph_cache is not None else real_batch)
                    cold_graph = graph_cache is not None and executed_batch not in graph_cache
                    if performance is not None:
                        group_started = time.perf_counter()
                        gpu_sample = performance.cuda_sample(config.device,
                            ("inference.prepare_cuda", "inference.compute_cuda_cold" if cold_graph
                             else "inference.compute_cuda"))
                    array = torch.from_numpy(np.concatenate([x.states for x in group]))
                    masks = torch.from_numpy(np.concatenate([x.masks for x in group])).to(config.device)
                    if config.device == "cuda":
                        array = array.pin_memory()
                    array = array.to(config.device, non_blocking=True)
                    if config.channels_last and config.device == "cuda":
                        array = array.contiguous(memory_format=torch.channels_last)
                    if performance is not None:
                        performance.observe("inference.prepare_host_wall", time.perf_counter() - group_started)
                        if gpu_sample is not None:
                            gpu_sample.mark()
                        execution_started = time.perf_counter()
                    with torch.inference_mode(), autocast_context(config.device, config.precision):
                        logits, values = models[version](array)
                        raw_finite = torch.isfinite(logits).all()
                        policy_tensor = logits.float().masked_fill(~masks, -torch.inf).softmax(1)
                        if performance is not None and gpu_sample is not None:
                            gpu_sample.mark()  # Existing .cpu() below completes events.
                        policies = policy_tensor.cpu().numpy()
                        del policy_tensor
                        values = values.float().cpu().numpy()
                    if (not bool(raw_finite) or not np.isfinite(policies).all() or not np.isfinite(values).all()
                            or np.any(np.abs(values) > 1.00001)
                            or not np.allclose(policies.sum(1), 1, atol=1e-5)):
                        raise FloatingPointError("non-finite or unnormalized neural output")
                    if performance is not None:
                        performance.observe("inference.execution_wall", time.perf_counter() - execution_started)
                        performance.observe("inference.service_wall", time.perf_counter() - group_started)
                        performance.observe("inference.batch_size", len(array))
                        performance.observe("inference.executed_batch_size", executed_batch)
                        if cold_graph:
                            performance.observe("inference.graph_warmup_wall", time.perf_counter() - execution_started)
                        performance.increment("inference.positions", len(array))
                        if gpu_sample is not None:
                            gpu_sample.finish()
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
                if performance is not None:
                    performance.flush()
            except Exception as exc:
                for item in batch:
                    try:
                        responses[item.worker_id].put(BatchResponse(
                            item.request_id, None, None, item.worker_id, step, repr(exc)), timeout=1)
                    except (Full, KeyError):
                        pass
                raise
    except BaseException as exc:
        report_failure(failure_queue, "inference", exc)
        if not announced:
            ready.put(dict(error=repr(exc)))
        stop.set()
        raise
    finally:
        if performance is not None:
            performance.flush(force=True)
