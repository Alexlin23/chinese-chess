"""Bounded request failure reporting and an out-of-process view of inference stalls."""
import json
import os
from queue import Full
import threading
import time
import traceback


STAGES = ("starting", "actor_read", "model_create", "weights_load", "idle",
          "prepare", "forward", "graph_allocate", "graph_warmup", "graph_capture",
          "result_copy", "respond", "failed", "stopped")


def report_failure(queue, source, exc, **details):
    if queue is None:
        return
    message = dict(source=source, pid=os.getpid(), error=repr(exc),
                   traceback=traceback.format_exc(), time=time.time(), **details)
    try:
        queue.put(message, timeout=1)
    except Full:
        # Preserve evidence even if the reader has already gone away.
        print("Runtime failure: " + json.dumps(message), flush=True)


class InferenceStatus:
    """Small shared record: the parent can read it while CUDA blocks its owner.

    No CUDA synchronization or hardware queries. GPU memory is a cached allocator
    reading refreshed only during loads/cold calls. Disabled runs create no record.
    """
    def __init__(self, context):
        self.values = context.Array("d", 10)
        self.mark("starting")

    def mark(self, stage, version=None, batch_size=None, versions=None, memory=None):
        now = time.perf_counter()
        with self.values.get_lock():
            data = self.values.get_obj()
            old_stage, elapsed = STAGES[int(data[0])], now - data[1]
            data[0], data[1] = STAGES.index(stage), now
            data[9] = os.getpid()
            if version is not None: data[3] = version
            if batch_size is not None: data[4] = batch_size
            if versions is not None: data[5] = versions
            if memory is not None: data[6], data[7] = memory
        if elapsed >= 5 and old_stage not in ("idle", "starting", "stopped", "failed"):
            print("Slow inference stage: " + json.dumps(dict(stage=old_stage,
                  seconds=elapsed, next_stage=stage, pid=os.getpid(), time=time.time())), flush=True)
        return old_stage, elapsed

    def complete(self):
        with self.values.get_lock():
            data = self.values.get_obj()
            data[2] += 1
            data[8] = time.perf_counter()
        self.mark("idle")

    def snapshot(self):
        with self.values.get_lock():
            data = list(self.values.get_obj())
        return dict(pid=int(data[9]), stage=STAGES[int(data[0])], stage_seconds=max(0, time.perf_counter()-data[1]),
                    completed_batches=int(data[2]), model_step=int(data[3]), batch_size=int(data[4]),
                    resident_versions=int(data[5]), gpu_allocated_gb=data[6]/1024**3,
                    gpu_reserved_gb=data[7]/1024**3,
                    last_completion_age=None if not data[8] else max(0, time.perf_counter()-data[8]))


class InferenceWatch:
    """Parent-only, low-frequency logging independent of the inference thread."""
    def __init__(self, status, directory, live=None, warning_seconds=15, timeout=300, session_id=None):
        self.status, self.directory, self.live = status, directory, live
        self.warning_seconds, self.timeout = warning_seconds, timeout
        self.session_id = session_id
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        previous_stall = False
        with (self.directory / "inference-diagnostics.jsonl").open("a", encoding="utf-8") as log:
            while True:
                snapshot = self.status.snapshot()
                stalled = (snapshot["stage"] not in ("idle", "starting", "failed", "stopped")
                           and snapshot["stage_seconds"] >= self.warning_seconds)
                snapshot.update(time=time.time(), session_id=self.session_id, observer_pid=os.getpid(), request_timeout_seconds=self.timeout,
                                warning_seconds=self.warning_seconds, stalled=stalled)
                log.write(json.dumps(snapshot, allow_nan=False) + "\n")
                log.flush()
                if self.live:
                    self.live.update(inference_diagnostics=snapshot)
                if stalled and not previous_stall:
                    print("Inference stall warning: " + json.dumps(snapshot), flush=True)
                previous_stall = stalled
                if self.stop_event.wait(5):
                    break

    def close(self):
        self.stop_event.set()
        self.thread.join(3)
