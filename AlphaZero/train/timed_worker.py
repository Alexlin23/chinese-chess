"""Compatibility import; timing observes the one canonical episode runner."""
from .self_play_worker import BatchRemoteEvaluator, self_play_worker_fn

TimedRemoteEvaluator = BatchRemoteEvaluator
timed_self_play_worker_fn = self_play_worker_fn
