"""Optional coarse wall-clock accounting; never consume training RNG or sync CUDA.

Child processes send cumulative snapshots, so a dropped intermediate message
does not lose counts after the next successful report. Histograms use fixed
logarithmic buckets: p95 is an approximate upper bound, not an exact quantile.
Times overlap across actors/inference/learner and MUST NOT be added together.
"""
import math
from queue import Full
import threading
import time


class Metric:
    def __init__(self):
        self.count = 0
        self.total = self.maximum = 0.0
        self.buckets = [0] * 128

    def add(self, value):
        if not math.isfinite(value) or value < 0:
            raise ValueError("performance observation must be finite and nonnegative")
        self.count += 1
        self.total += value
        self.maximum = max(self.maximum, value)
        bucket = 0 if value == 0 else max(0, min(127, int((math.log2(value) + 20) * 4)))
        self.buckets[bucket] += 1

    def merge(self, state):
        self.count += state["count"]
        self.total += state["total"]
        self.maximum = max(self.maximum, state["max"])
        self.buckets = [a + b for a, b in zip(self.buckets, state["histogram"])]

    def raw(self):
        return dict(count=self.count, total=self.total, max=self.maximum, histogram=list(self.buckets))

    def summary(self):
        p95 = None
        if self.count:
            accumulated = 0
            for i, count in enumerate(self.buckets):
                accumulated += count
                if accumulated >= math.ceil(self.count * .95):
                    p95 = self.maximum if i == 127 else min(self.maximum, 2 ** ((i + 1) / 4 - 20))
                    break
        return dict(count=self.count, total=self.total,
                    mean=self.total / self.count if self.count else None,
                    max=self.maximum, p95_upper_approx=p95)


class Recorder:
    def __init__(self, source, queue=None, interval=5.0, gpu_every=64):
        if not math.isfinite(interval) or interval <= 0 or type(gpu_every) is not int or gpu_every < 1:
            raise ValueError("positive report interval and integer GPU sampling period required")
        self.source, self.queue = source, queue
        self.interval, self.gpu_every = interval, gpu_every
        self.lock = threading.Lock()
        self.metrics, self.counters = {}, {}
        self.last_report = time.perf_counter()
        self.cuda_calls = 0
        self.neural_sync_total = 0.0

    def observe(self, name, value):
        with self.lock:
            metric = self.metrics.get(name)
            if metric is None:
                metric = self.metrics[name] = Metric()
            metric.add(value)

    def increment(self, name, amount=1):
        with self.lock:
            self.counters[name] = self.counters.get(name, 0) + amount

    def raw(self):
        with self.lock:
            return dict(source=self.source, reported_at=time.perf_counter(),
                        metrics={name: metric.raw() for name, metric in self.metrics.items()},
                        counters=dict(self.counters))

    def flush(self, force=False):
        if self.queue is None:
            return
        now = time.perf_counter()
        if not force and now - self.last_report < self.interval:
            return
        self.last_report = now
        try:
            self.queue.put_nowait(self.raw())
        except Full:
            self.increment("monitor.dropped_reports")

    def cuda_sample(self, device, names):
        if not str(device).startswith("cuda"):
            return None
        self.cuda_calls += 1
        if (self.cuda_calls - 1) % self.gpu_every:
            return None
        return CudaSample(self, names)


class CudaSample:
    """Read event pairs only if existing D2H/finite checks already completed them.

    These are elapsed stream intervals, including contention and host launch
    gaps, rather than summed kernel-active time. No synchronize() is called.
    """
    def __init__(self, recorder, names):
        import torch
        self.recorder, self.names = recorder, names
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(len(names) + 1)]
        self.index = 0
        self.mark()

    def mark(self):
        self.events[self.index].record()
        self.index += 1

    def finish(self):
        if self.index != len(self.events):
            raise RuntimeError("incomplete CUDA performance sample")
        if not self.events[-1].query():
            self.recorder.increment("monitor.cuda_sample_not_ready")
            return
        for name, first, last in zip(self.names, self.events, self.events[1:]):
            self.recorder.observe(name, first.elapsed_time(last) / 1000)


class Hub:
    def __init__(self, local, clock=time.perf_counter):
        self.local, self.clock = local, clock
        self.started = clock()
        self.lock = threading.Lock()
        self.sources = {}

    def receive(self, message):
        with self.lock:
            previous = self.sources.get(message["source"])
            if previous is None or message["reported_at"] >= previous["reported_at"]:
                self.sources[message["source"]] = message

    def snapshot(self):
        self.receive(self.local.raw())
        with self.lock:
            sources = list(self.sources.values())
        metrics, counters = {}, {}
        for source in sources:
            for name, state in source["metrics"].items():
                metrics.setdefault(name, Metric()).merge(state)
            for name, value in source["counters"].items():
                counters[name] = counters.get(name, 0) + value
        elapsed = max(0.0, self.clock() - self.started)
        def rate(name, scale=1):
            return counters.get(name, 0) * scale / elapsed if elapsed else 0.0
        return dict(enabled=True, elapsed_seconds=elapsed, sources=len(sources),
                    metrics={name: value.summary() for name, value in metrics.items()}, counters=counters,
                    rates=dict(mcts_simulations_per_second=rate("mcts.simulations"),
                               inference_positions_per_second=rate("inference.positions"),
                               new_positions_per_hour=rate("data.new_positions", 3600),
                               completed_games_per_hour=rate("data.completed_games", 3600),
                               sgd_updates_per_second=rate("sgd.updates")),
                    notes=["Cumulative current-session rates; startup and unfinished searches are included in wall time",
                           "MCTS local CPU is a wall-time estimate, not process CPU time",
                           "Request RTT includes inference and queueing; it is not pure IPC overhead",
                           "Concurrent module times overlap and must not be added",
                           "P95 is a fixed-histogram approximate upper bound",
                           "CUDA events are sampled and read without additional synchronization"])
