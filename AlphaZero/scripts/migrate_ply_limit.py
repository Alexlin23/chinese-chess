"""Decrease the game horizon in a new run, preserving SGD state and safe replay.

Only completed games ending by the new limit are retained. Historical visit
targets are kept with provenance; their searches are not regenerated.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import tempfile

import numpy as np
import torch

from AlphaZero.scripts.migrate_natural_limit import stopped_source, validate_game
from AlphaZero.train.config import AlphaZeroConfig
from AlphaZero.train.replay import PackedSample, ReplayBuffer, VERSION
from AlphaZero.train.trainer import atomic_torch_save


def migrate(source, destination, max_game_ply):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or source == destination or source in destination.parents:
        raise ValueError("destination must be a new sibling run, outside the source run")
    with stopped_source(source):
        payload = torch.load(source / "learner.pt", map_location="cpu", weights_only=False)
        original = AlphaZeroConfig(**payload["config"])
        config = AlphaZeroConfig(**dict(original.to_dict(), max_game_ply=max_game_ply,
                                       checkpoint_dir=str(destination)))
        if max_game_ply >= original.max_game_ply:
            raise ValueError("this migration only supports decreasing max_game_ply")
        if (payload.get("kind") != "learner" or payload.get("optimizer_type") != "SGD"
                or payload.get("fingerprint") != original.fingerprint
                or payload.get("spec") != original.spec):
            raise ValueError("source checkpoint specification mismatch")
        tensors = list(payload["model_state_dict"].values()) + [
            value for state in payload["optimizer_state_dict"]["state"].values()
            for value in state.values() if isinstance(value, torch.Tensor)]
        if not all(torch.isfinite(value).all() for value in tensors):
            raise ValueError("source has non-finite model or optimizer state")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect((source / "replay.sqlite").as_uri() + "?mode=ro", uri=True) as db:
            db.execute("BEGIN")
            metadata = dict(db.execute("SELECT key,value FROM metadata"))
            generation = db.execute("SELECT COALESCE(MAX(id),0) FROM games").fetchone()[0]
            recovery = payload.get("extra", {})
            if (metadata.get("version") != VERSION
                    or metadata.get("fingerprint") != original.fingerprint
                    or metadata.get("run_id") != payload.get("run_id")
                    or recovery.get("replay_run_id") != metadata.get("run_id")
                    or recovery.get("replay_generation", -1) > generation
                    or int(metadata.get("max_actor_step", 0)) > payload["step"]):
                raise ValueError("source learner/replay identity or generation mismatch")
            provenance = dict(source_run_id=metadata["run_id"],
                              source_max_game_ply=original.max_game_ply,
                              target_max_game_ply=max_game_ply,
                              policy_search_max_game_ply=original.max_game_ply,
                              policy_targets="historical MCTS visits; not regenerated")
            report = dict(source=str(source), destination=str(destination),
                          source_fingerprint=original.fingerprint, target_fingerprint=config.fingerprint,
                          step=payload["step"], preserved="model, SGD momentum, step, LR schedule and RNG states",
                          kept_games=0, kept_positions=0, rejected=[], **provenance)
            with tempfile.TemporaryDirectory(prefix=destination.name + "-migration-", dir=destination.parent) as temp:
                staging = Path(temp) / "run"
                staging.mkdir()
                replay = ReplayBuffer(staging / "replay.sqlite", config.replay_window_games,
                                      config.replay_cache_mb, window_positions=config.replay_window_positions)
                try:
                    replay.bind_spec(config.fingerprint)
                    report["target_run_id"] = replay.run_id
                    for game, episode, worker, sequence, reason, start, end, raw_trace in db.execute(
                            "SELECT id,episode,worker,sequence,reason,actor_start,actor_end,trace FROM games ORDER BY id"):
                        if max(start, end) > payload["step"]:
                            raise ValueError("game uses actor newer than checkpoint")
                        trace = json.loads(raw_trace)
                        samples = [PackedSample(*row) for row in db.execute(
                            "SELECT planes,globals,actions,probabilities,value FROM positions WHERE game=? ORDER BY id",
                            (game,))]
                        if not samples or len(trace.get("moves", [])) != len(samples):
                            raise ValueError("missing or incomplete recorded moves")
                        if len(samples) > max_game_ply:
                            report["rejected"].append(dict(episode=episode, positions=len(samples),
                                                           reason="trajectory exceeds new horizon"))
                            continue
                        rejection = validate_game(trace, samples, reason, original.max_game_ply)
                        if rejection:
                            raise ValueError("invalid source trajectory: " + rejection)
                        converted = []
                        for ply, sample in enumerate(samples):
                            globals_ = np.frombuffer(sample.globals, dtype="<f4").copy()
                            globals_[1] = ply / max_game_ply
                            converted.append(PackedSample(sample.planes, globals_.tobytes(), sample.actions,
                                                          sample.probabilities, sample.value))
                        rejection = validate_game(trace, converted, reason, max_game_ply)
                        if rejection:
                            raise ValueError("target trajectory changed: " + rejection)
                        trace["ply_limit_migration"] = provenance
                        replay.add_game(episode, converted, worker=worker, sequence=sequence, reason=reason,
                                        actor_start=start, actor_end=end, trace=json.dumps(trace))
                        report["kept_games"] += 1
                        report["kept_positions"] += len(converted)
                    with replay.db:
                        replay.db.executemany("INSERT OR REPLACE INTO metadata VALUES(?,?)",
                                              ((k, v) for k, v in metadata.items() if k.startswith("worker:")))
                        replay.db.execute("INSERT INTO metadata VALUES('ply_limit_migration',?)", (json.dumps(report),))
                    payload.update(spec=config.spec, fingerprint=config.fingerprint, config=config.to_dict(),
                                   run_id=replay.run_id, extra=replay.recovery_marker(), ply_limit_migration=report)
                    atomic_torch_save(payload, staging / "learner.pt")
                    actor = {key: payload[key] for key in ("spec", "fingerprint", "config", "run_id", "step", "model_state_dict")}
                    atomic_torch_save(dict(actor, kind="actor", ply_limit_migration=report), staging / "latest.pt")
                    (staging / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf8")
                    (staging / "ply-limit-migration.json").write_text(json.dumps(report, indent=2), encoding="utf8")
                    replay.flush()
                finally:
                    replay.close()
                staging.rename(destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--max-game-ply", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(migrate(args.source, args.destination, args.max_game_ply), indent=2), flush=True)


if __name__ == "__main__":
    main()
