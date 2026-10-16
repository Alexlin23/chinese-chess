import threading
import pytest
from AlphaZero.train import self_play_worker as workers


def test_group_keeps_slot_identity_and_sequence(monkeypatch):
    seen = {}
    lock, stop = threading.Lock(), threading.Event()
    def worker(worker_id, config, request, response, result, events, stopped, sequence):
        with lock:
            seen[worker_id] = sequence, response
            if len(seen) == 4:
                stopped.set()
        assert stopped.wait(3)
    monkeypatch.setattr(workers, "self_play_worker_fn", worker)
    workers.self_play_group_fn([1,3,5,7], None, None, {i:i*10 for i in [1,3,5,7]},
                              None, None, stop, {7:12})
    assert seen == {1:(0,10),3:(0,30),5:(0,50),7:(12,70)}


def test_slot_failure_stops_the_group(monkeypatch):
    stop = threading.Event()
    def worker(worker_id, config, request, response, result, events, stopped, sequence):
        if worker_id == 0:
            raise RuntimeError("slot failed")
        assert stopped.wait(3)
    monkeypatch.setattr(workers, "self_play_worker_fn", worker)
    with pytest.raises(RuntimeError, match="slot failed"):
        workers.self_play_group_fn([0,1], None, None, {0:None,1:None}, None, None, stop, {})
    assert stop.is_set()
