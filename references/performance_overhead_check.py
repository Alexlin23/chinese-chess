"""Small CPU learner A/B; NOT a GPU or whole-system throughput benchmark."""
import hashlib
import json
import statistics
import time
import numpy as np
import torch
from AlphaZero.engine import GameState
from AlphaZero.train.config import AlphaZeroConfig
from AlphaZero.train.performance import Recorder
from AlphaZero.train.trainer import Trainer


def main():
    torch.set_num_threads(1)
    state = GameState.new_game()
    policy = state.legal_mask().astype(np.float32)
    policy /= policy.sum()
    class Batch:
        def __len__(self): return 2
        def sample(self, n):
            return np.repeat(state.encode()[None], n, 0), np.repeat(policy[None], n, 0), np.zeros(n, np.float32), np.ones(n)
    results = []
    for enabled in (False, True, True, False):
        torch.manual_seed(81)
        trainer = Trainer(AlphaZeroConfig(device="cpu", precision="fp32", batch_size=2))
        for _ in range(3):
            trainer.step(Batch())
        trainer.performance = Recorder("learner") if enabled else None
        started = time.perf_counter()
        for _ in range(12):
            trainer.step(Batch())
        seconds = time.perf_counter() - started
        digest = hashlib.sha256()
        for value in trainer.model.state_dict().values():
            digest.update(value.detach().numpy().tobytes())
        results.append(dict(enabled=enabled, seconds_per_update=seconds / 12,
                            parameter_sha256=digest.hexdigest()))
    off = statistics.median(r["seconds_per_update"] for r in results if not r["enabled"])
    on = statistics.median(r["seconds_per_update"] for r in results if r["enabled"])
    print(json.dumps(dict(scope="CPU learner only, 19x128, batch2, ABBA, 12 updates per trial",
                          results=results, off_seconds=off, on_seconds=on,
                          relative_change_percent=(on / off - 1) * 100,
                          same_parameters=len({r["parameter_sha256"] for r in results}) == 1,
                          limitation="Does not measure CUDA events, IPC, summary I/O or GPU throughput overhead"), indent=2))


if __name__ == "__main__":
    main()
