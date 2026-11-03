"""Bounded per-actor shared arrays; queues carry only readiness and version IDs.

Each actor has one outstanding request. A request slot remains immutable until
its response arrives; response arrays are copied before the actor can reuse it.
RawArray ownership follows multiprocessing's normal process lifetime (no named
shared-memory segments to unlink after a failed worker).
"""
from dataclasses import dataclass
from queue import Empty, Full
import numpy as np


class SignalQueue:
    """Small bounded-by-protocol notifications, without a queue feeder thread.

    One reader, potentially many writers. Only one notification per outstanding
    actor request is allowed; the shared arrays hold the actual large payloads.
    """
    def __init__(self, context):
        self.reader, self.writer = context.Pipe(duplex=False)
        self.write_lock = context.Lock()

    def put(self, item, block=True, timeout=None):
        if not self.write_lock.acquire(block, timeout):
            raise Full
        try:
            self.writer.send(item)
        finally:
            self.write_lock.release()

    def get(self, block=True, timeout=None):
        if (not block or timeout is not None) and not self.reader.poll(timeout if block else 0):
            raise Empty
        return self.reader.recv()

    def close(self):
        self.reader.close()
        self.writer.close()

    def cancel_join_thread(self):
        pass


@dataclass
class _Request:
    request_id: int
    worker_id: int
    count: int
    model_step: object


@dataclass
class _Response:
    request_id: int
    worker_id: int
    count: int
    model_step: int


class SharedInferenceBuffers:
    def __init__(self, context, workers, batch, channels, policy_size):
        self.shape = workers, batch, channels, policy_size
        self.states = context.RawArray("f", workers * batch * channels * 90)
        self.masks = context.RawArray("b", workers * batch * policy_size)
        self.results = context.RawArray("f", workers * batch * (policy_size + 1))
        self._views = None

    def __getstate__(self):
        return {**self.__dict__, "_views": None}

    def views(self):
        if self._views is None:
            workers, batch, channels, policy = self.shape
            self._views = (
                np.frombuffer(self.states, dtype=np.float32).reshape(workers, batch, channels, 10, 9),
                np.frombuffer(self.masks, dtype=np.bool_).reshape(workers, batch, policy),
                np.frombuffer(self.results, dtype=np.float32).reshape(workers, batch, policy + 1))
        return self._views


class _Queue:
    def __init__(self, queue, buffers):
        self.queue, self.buffers = queue, buffers

    def cancel_join_thread(self):
        self.queue.cancel_join_thread()

    def close(self):
        self.queue.close()


class SharedRequestQueue(_Queue):
    def put(self, request, *args, **kwargs):
        from .inference_server import BatchRequest
        workers, batch, channels, policy = self.buffers.shape
        if (not isinstance(request, BatchRequest) or not 0 <= request.worker_id < workers
                or not isinstance(request.states, np.ndarray) or not isinstance(request.masks, np.ndarray)
                or request.states.dtype != np.float32 or request.masks.dtype != np.bool_
                or not 0 < len(request.states) <= batch
                or request.states.shape != (len(request.states), channels, 10, 9)
                or request.masks.shape != (len(request.states), policy)):
            # Preserve server-side error reporting for malformed requests.
            return self.queue.put(request, *args, **kwargs)
        count = len(request.states)
        states, masks, _ = self.buffers.views()
        np.copyto(states[request.worker_id, :count], request.states)
        np.copyto(masks[request.worker_id, :count], request.masks)
        return self.queue.put(_Request(request.request_id, request.worker_id, count,
                                       request.model_step), *args, **kwargs)

    def get(self, *args, **kwargs):
        from .inference_server import BatchRequest
        request = self.queue.get(*args, **kwargs)
        if not isinstance(request, _Request):
            return request
        states, masks, _ = self.buffers.views()
        return BatchRequest(request.request_id, states[request.worker_id, :request.count],
                            masks[request.worker_id, :request.count], request.worker_id, request.model_step)

    def get_nowait(self):
        return self.get(block=False)


class SharedResponseQueue(_Queue):
    def put(self, response, *args, **kwargs):
        if response.error:
            return self.queue.put(response, *args, **kwargs)
        count = len(response.policies)
        _, _, results = self.buffers.views()
        target = results[response.worker_id, :count]
        np.copyto(target[:, :-1], response.policies)
        np.copyto(target[:, -1], np.asarray(response.values).reshape(-1))
        return self.queue.put(_Response(response.request_id, response.worker_id, count,
                                        response.model_step), *args, **kwargs)

    def get(self, *args, **kwargs):
        from .inference_server import BatchResponse
        response = self.queue.get(*args, **kwargs)
        if not isinstance(response, _Response):
            return response
        _, _, results = self.buffers.views()
        # The next request may reuse this worker's slot; consumers own this copy.
        host = results[response.worker_id, :response.count].copy()
        return BatchResponse(response.request_id, host[:, :-1], host[:, -1],
                             response.worker_id, response.model_step)

    def get_nowait(self):
        return self.get(block=False)
