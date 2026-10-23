"""Rebase a stopped run's SGD schedule while preserving model versions and all tensors."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from AlphaZero.scripts.paths import DEFAULT_CONFIG, run_paths
from AlphaZero.scripts.pipeline import run_lock
from AlphaZero.train.config import AlphaZeroConfig
from AlphaZero.train.trainer import atomic_torch_save


def reset_optimizer_step(directory, target):
    if type(target) is not int or target < 1:
        raise ValueError("optimizer step must be a positive integer")
    directory = Path(directory)
    with run_lock(directory, write_pid=False):
        path = directory / "learner.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("kind") != "learner" or payload.get("optimizer_type") != "SGD":
            raise ValueError("a versioned SGD learner checkpoint is required")
        config = AlphaZeroConfig(**payload["config"])
        if payload.get("fingerprint") != config.fingerprint:
            raise ValueError("checkpoint specification mismatch")
        old = payload["step"]
        offset = payload.get("optimizer_step_offset", 0)
        if type(old) is not int or type(offset) is not int or offset < 0 or not 1 <= target <= old:
            raise ValueError("target must not exceed the current SGD step")
        lr = config.learning_rate * config.lr_decay_rate ** sum(
            target >= milestone for milestone in config.lr_milestones)
        record = dict(event="optimizer_step_reset", created_utc=datetime.now(timezone.utc).isoformat(),
                      run_id=payload.get("run_id"), old_step=old, new_step=target,
                      model_step=old + offset, learning_rate=lr)
        if target == old:
            return record
        payload["step"] = target
        payload["optimizer_step_offset"] = old + offset - target
        payload["current_iteration"] = target // config.checkpoint_interval
        for group in payload["optimizer_state_dict"]["param_groups"]:
            group["lr"] = lr
        extra = dict(payload.get("extra", {}))
        extra["optimizer_step_changes"] = [*extra.get("optimizer_step_changes", []), record]
        payload["extra"] = extra
        atomic_torch_save(payload, path)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir")
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    _, directory = run_paths(args.config, args.run_dir)
    torch.set_num_threads(1)
    print(json.dumps(reset_optimizer_step(directory, args.step)), flush=True)


if __name__ == "__main__":
    main()
