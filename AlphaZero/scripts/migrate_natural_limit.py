"""Explicit v2 -> natural-limit migration into a NEW run; source stays recoverable.

Replays every recorded move and checks observations, legal policy support and
terminal value labels. Historical policy targets remain searches under v2 and
are marked as such; replaying moves does not regenerate their MCTS visits.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile

import numpy as np
import torch

from AlphaZero.engine import GameState, Move
from AlphaZero.train.config import AlphaZeroConfig, RULE_VERSION
from AlphaZero.train.replay import PackedSample, ReplayBuffer, VERSION
from AlphaZero.train.trainer import atomic_torch_save

OLD_RULE = "xiangqi-development-v2"


def fingerprint(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


@contextmanager
def stopped_source(source):
    # Share the learner's lock, without rewriting its process.pid or checkpoint.
    import fcntl
    with (source / ".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("stop the source learner before migration") from exc
        yield


def validate_game(trace, samples, reason, max_ply):
    moves = trace.get("moves")
    if not isinstance(moves, list) or len(moves) != len(samples) or not moves:
        raise ValueError("missing or incomplete recorded moves")
    state = GameState.new_game(max_ply)
    players = []
    for move, sample in zip(moves, samples):
        if state.is_terminal():
            return "new rule ends the trajectory earlier"
        sample.validate()
        encoding = state.encode()
        if (np.packbits(encoding[:128].astype(np.uint8)).tobytes() != sample.planes
                or encoding[128:, 0, 0].astype("<f4").tobytes() != sample.globals):
            raise ValueError("stored observation differs from replayed position")
        actions = np.frombuffer(sample.actions, dtype="<u2")
        if not state.canonical_mask()[actions].all():
            raise ValueError("stored policy contains illegal actions")
        players.append(state.current_player)
        state = state.apply(Move(*move))
    result = state.game_result()
    if result is None or result.reason != reason:
        return "terminal adjudication changed"
    if any(sample.value != result.to_value(player) for sample, player in zip(samples, players)):
        raise ValueError("stored outcome labels differ from adjudication")
    return None


def migrate(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or source == destination or source in destination.parents:
        raise ValueError("destination must be a new sibling run, outside the source run")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with stopped_source(source):
        payload = torch.load(source / "learner.pt", map_location="cpu", weights_only=False)
        config = AlphaZeroConfig(**payload["config"])
        old_spec = dict(config.spec)
        old_spec["rule"] = OLD_RULE
        for key in ("natural_limit_rounds", "natural_limit_check_cap", "natural_limit_claim"):
            del old_spec[key]
        if (payload.get("kind") != "learner" or payload.get("optimizer_type") != "SGD"
                or payload.get("spec") != old_spec or payload.get("fingerprint") != fingerprint(old_spec)):
            raise ValueError("only the known v2 learner with unchanged architecture/algorithm can migrate")
        for value in list(payload["model_state_dict"].values()) + [
                value for state in payload["optimizer_state_dict"]["state"].values()
                for value in state.values() if isinstance(value, torch.Tensor)]:
            if not torch.isfinite(value).all():
                raise ValueError("source has non-finite model or optimizer state")
        with sqlite3.connect((source / "replay.sqlite").as_uri() + "?mode=ro", uri=True) as db:
            db.execute("BEGIN")
            metadata = dict(db.execute("SELECT key,value FROM metadata"))
            generation = db.execute("SELECT COALESCE(MAX(id),0) FROM games").fetchone()[0]
            recovery = payload.get("extra", {})
            if (metadata.get("version") != VERSION.replace(RULE_VERSION, OLD_RULE)
                    or metadata.get("fingerprint") != payload["fingerprint"]
                    or metadata.get("run_id") != payload.get("run_id")
                    or recovery.get("replay_run_id") != metadata["run_id"]
                    or recovery.get("replay_generation", -1) > generation
                    or int(metadata.get("max_actor_step", 0)) > payload["step"]):
                raise ValueError("source learner/replay identity or generation mismatch")
            report = dict(source=str(source), destination=str(destination), source_run_id=metadata["run_id"],
                          source_fingerprint=payload["fingerprint"], target_fingerprint=config.fingerprint,
                          source_rule=OLD_RULE, target_rule=RULE_VERSION, step=payload["step"],
                          preserved="model, SGD momentum, step, learning-rate schedule and RNG states",
                          policy_targets="historical v2 MCTS visits, not regenerated under new rules",
                          kept_games=0, kept_positions=0, rejected=[])
            with tempfile.TemporaryDirectory(prefix=destination.name + "-migration-", dir=destination.parent) as temp:
                staging = Path(temp) / "run"
                staging.mkdir()
                replay = ReplayBuffer(staging / "replay.sqlite", config.replay_window_games, config.replay_cache_mb)
                try:
                    report["target_run_id"] = replay.run_id
                    replay.bind_spec(config.fingerprint)
                    for game_id, episode, worker, sequence, reason, start, end, raw_trace in db.execute(
                            "SELECT id,episode,worker,sequence,reason,actor_start,actor_end,trace FROM games ORDER BY id"):
                        if max(start, end) > payload["step"]:
                            raise ValueError("game uses actor newer than checkpoint")
                        trace = json.loads(raw_trace)
                        samples = [PackedSample(*row) for row in db.execute(
                            "SELECT planes,globals,actions,probabilities,value FROM positions WHERE game=? ORDER BY id",
                            (game_id,))]
                        rejection = validate_game(trace, samples, reason, config.max_game_ply)
                        if rejection:
                            report["rejected"].append(dict(episode=episode, reason=rejection))
                            continue
                        trace["rule_migration"] = dict(source_rule=OLD_RULE, validated_rule=RULE_VERSION,
                                                       policy_search_rule=OLD_RULE, source_run_id=metadata["run_id"])
                        replay.add_game(episode, samples, worker=worker, sequence=sequence, reason=reason,
                                        actor_start=start, actor_end=end, trace=json.dumps(trace))
                        report["kept_games"] += 1
                        report["kept_positions"] += len(samples)
                    # Keep discarded episodes' seeds from being reused as new games.
                    with replay.db:
                        replay.db.executemany("INSERT OR REPLACE INTO metadata VALUES(?,?)",
                                              ((k, v) for k, v in metadata.items() if k.startswith("worker:")))
                        replay.db.execute("INSERT INTO metadata VALUES('rule_migration',?)", (json.dumps(report),))
                    config.checkpoint_dir = str(destination)
                    payload.update(spec=config.spec, fingerprint=config.fingerprint, config=config.to_dict(),
                                   run_id=replay.run_id, extra=replay.recovery_marker(), rule_migration=report)
                    atomic_torch_save(payload, staging / "learner.pt")
                    actor = {key: payload[key] for key in ("spec", "fingerprint", "config", "run_id", "step", "model_state_dict")}
                    atomic_torch_save(dict(actor, kind="actor", rule_migration=report), staging / "latest.pt")
                    (staging / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
                    (staging / "rule-migration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                    replay.flush()
                finally:
                    replay.close()
                staging.rename(destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    print(json.dumps(migrate(args.source, args.destination), indent=2), flush=True)


if __name__ == "__main__":
    main()
