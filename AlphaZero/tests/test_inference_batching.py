"""Batch collection bounds both positions and outstanding independent actors."""
from queue import Empty
from types import SimpleNamespace

from AlphaZero.train.inference_server import _gather_batch


class Requests:
    def __init__(self, *items):
        self.items = list(items)
        self.calls = 0

    def get(self, timeout):
        self.calls += 1
        assert timeout > 0
        if not self.items:
            raise Empty
        return self.items.pop(0)


def request(worker, count=1):
    return SimpleNamespace(worker_id=worker, states=[None] * count)


def test_no_timeout_wait_after_every_actor_has_sent():
    first, second = request(0), request(1, 3)
    queue = Requests(second)
    batch, pending = _gather_batch(first, queue, 128, 2, 1000)
    assert batch == [first, second] and pending is None
    assert queue.calls == 1
    queue = Requests()
    assert _gather_batch(first, queue, 128, 1, 1000) == ([first], None)
    assert queue.calls == 0


def test_overflow_request_is_preserved_for_next_batch():
    first, second = request(0, 3), request(1, 2)
    queue = Requests(second)
    assert _gather_batch(first, queue, 4, 2, 1000) == ([first], second)


def test_partial_batch_deadline_and_disabled_wait():
    first = request(0)
    queue = Requests()
    assert _gather_batch(first, queue, 128, 8, 1000) == ([first], None)
    assert queue.calls == 1
    queue = Requests(request(1))
    assert _gather_batch(first, queue, 128, 8, 0) == ([first], None)
    assert queue.calls == 0


def test_position_limit_stops_before_actor_limit():
    first, second = request(0, 2), request(1, 2)
    queue = Requests(second, request(2))
    assert _gather_batch(first, queue, 4, 8, 1000) == ([first, second], None)
    assert queue.calls == 1
