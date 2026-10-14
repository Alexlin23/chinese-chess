"""Byte-bounded replay storage with compact boards and sparse policies."""
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import os
import numpy as np
from ..engine.constants import POLICY_SIZE
from ..engine.state import GameState


@dataclass
class PackedSample:
    board: bytes
    turn: bool
    move_count: int
    no_capture_count: int
    max_ply: int
    actions: np.ndarray
    probabilities: np.ndarray
    wdl_index: int
    weight: float

    @property
    def nbytes(self):
        # Include a conservative allowance for Python objects and deque refs.
        return len(self.board) + self.actions.nbytes + self.probabilities.nbytes + 256


def pack_sample(state: GameState, policy: np.ndarray,
                wdl: np.ndarray, weight: float = 1.0) -> PackedSample:
    policy = np.asarray(policy)
    if policy.shape != (POLICY_SIZE,) or not np.all(np.isfinite(policy)):
        raise ValueError("policy must be a finite (8100,) vector")
    actions = np.flatnonzero(policy > 0).astype(np.uint16)
    if len(actions) == 0:
        raise ValueError("policy has no positive action")
    probabilities = policy[actions].astype(np.float16)
    if probabilities.sum(dtype=np.float32) <= 0:
        raise ValueError("policy underflows float16 storage")
    return PackedSample(
        np.asarray(state.board, dtype=np.int8).tobytes(), bool(state.turn),
        int(state.move_count), int(state.no_capture_count), int(state._max_ply),
        actions, probabilities, int(np.argmax(wdl)), float(weight))


class ReplayBuffer:
    """FIFO replay bounded by sample count and stored bytes."""

    def __init__(self, max_size: int = 100_000,
                 max_bytes: int = 512 * 1024 * 1024):
        if max_size < 1 or max_bytes < 1:
            raise ValueError("replay limits must be positive")
        self.max_size, self.max_bytes = max_size, max_bytes
        self.samples = deque()
        self.byte_size = 0

    def add_packed(self, sample: PackedSample):
        if sample.nbytes > self.max_bytes:
            raise ValueError("single sample exceeds replay byte budget")
        while self.samples and (len(self.samples) >= self.max_size or
                                self.byte_size + sample.nbytes > self.max_bytes):
            self.byte_size -= self.samples.popleft().nbytes
        self.samples.append(sample)
        self.byte_size += sample.nbytes

    def add(self, state: GameState, policy: np.ndarray, wdl: np.ndarray,
            weight: float = 1.0):
        self.add_packed(pack_sample(state, policy, wdl, weight))

    def sample(self, n: int):
        if not self.samples:
            raise ValueError("Buffer is empty")
        items = list(self.samples)
        weights = np.asarray([s.weight for s in items], dtype=np.float64)
        if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("invalid replay weights")
        indices = np.random.choice(len(items), size=n, replace=True,
                                   p=weights / weights.sum())
        states = np.empty((n, 18, 10, 9), dtype=np.float32)
        policies = np.zeros((n, POLICY_SIZE), dtype=np.float32)
        wdls = np.zeros((n, 3), dtype=np.float32)
        batch_weights = np.empty(n, dtype=np.float32)
        for i, index in enumerate(indices):
            s = items[index]
            board = np.frombuffer(s.board, dtype=np.int8).reshape(10, 9)
            state = GameState(board, s.turn, s.move_count,
                              s.no_capture_count, max_ply=s.max_ply)
            states[i] = state.encode()
            probs = s.probabilities.astype(np.float32)
            policies[i, s.actions] = probs / probs.sum()
            wdls[i, s.wdl_index] = 1.0
            batch_weights[i] = s.weight
        return states, policies, wdls, batch_weights

    def __len__(self):
        return len(self.samples)

    def save(self, path: str):
        """Write numeric arrays atomically; loading never requires pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        items = list(self.samples)
        offsets = np.zeros(len(items) + 1, dtype=np.int64)
        for i, sample in enumerate(items):
            offsets[i + 1] = offsets[i] + len(sample.actions)
        temporary = path.with_name(path.name + ".tmp.npz")
        with open(temporary, "wb") as out:
            np.savez_compressed(
                out,
                boards=np.frombuffer(b"".join(s.board for s in items), dtype=np.int8).reshape(-1, 10, 9),
                turns=np.asarray([s.turn for s in items], dtype=np.bool_),
                move_counts=np.asarray([s.move_count for s in items], dtype=np.int32),
                no_capture_counts=np.asarray([s.no_capture_count for s in items], dtype=np.int32),
                max_plies=np.asarray([s.max_ply for s in items], dtype=np.int32),
                offsets=offsets,
                actions=np.concatenate([s.actions for s in items]) if items else np.array([], dtype=np.uint16),
                probabilities=np.concatenate([s.probabilities for s in items]) if items else np.array([], dtype=np.float16),
                wdl_indices=np.asarray([s.wdl_index for s in items], dtype=np.uint8),
                weights=np.asarray([s.weight for s in items], dtype=np.float32),
            )
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: str, max_size: int = 100_000,
             max_bytes: int = 512 * 1024 * 1024):
        buffer = cls(max_size, max_bytes)
        with np.load(path, allow_pickle=False) as data:
            offsets = data['offsets']
            for i in range(len(data['turns'])):
                lo, hi = int(offsets[i]), int(offsets[i + 1])
                buffer.add_packed(PackedSample(
                    data['boards'][i].tobytes(), bool(data['turns'][i]),
                    int(data['move_counts'][i]), int(data['no_capture_counts'][i]),
                    int(data['max_plies'][i]), data['actions'][lo:hi].copy(),
                    data['probabilities'][lo:hi].copy(),
                    int(data['wdl_indices'][i]), float(data['weights'][i])))
        return buffer
