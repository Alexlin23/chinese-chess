"""Adjust the game horizon in the same stopped run, with recoverable file replacement."""
import json
from contextlib import closing
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import uuid

import numpy as np
import torch

from AlphaZero.scripts.migrate_natural_limit import validate_game
from .config import AlphaZeroConfig, RESUME_FIXED_FIELDS
from .replay import PackedSample, ReplayBuffer, VERSION
from .trainer import atomic_torch_save

_FILES = ("replay.sqlite", "learner.pt", "latest.pt")


def _json_save(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf8") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _sidecars(directory):
    for suffix in ("-wal", "-shm"):
        (directory / ("replay.sqlite" + suffix)).unlink(missing_ok=True)


def _cleanup_adjustment(directory, record):
    # Keep the journal until cleanup finishes so an interrupted cleanup can resume.
    for name in _FILES:
        (directory / (".resume-backup-" + record["id"] + "-" + name)).unlink(missing_ok=True)
    (directory / ".resume-adjustment.json").unlink()


def _finish(directory, record):
    audit = directory / "parameter-changes.jsonl"
    known = {json.loads(line)["id"] for line in audit.read_text().splitlines()} if audit.exists() else set()
    if record["id"] not in known:
        with audit.open("a", encoding="utf8") as stream:
            stream.write(json.dumps(dict(record["report"], id=record["id"])) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    _cleanup_adjustment(directory, record)


def recover_adjustment(directory):
    """Caller holds the run lock; rollback any incomplete multi-file install."""
    directory = Path(directory)
    journal = directory / ".resume-adjustment.json"
    if not journal.exists():
        return
    record = json.loads(journal.read_text())
    if record.get("version") != 1 or not re.fullmatch(r"[0-9a-f]{32}", record.get("id", "")):
        raise ValueError("invalid resume adjustment journal")
    if record["phase"] == "committed":
        _finish(directory, record)
        return
    if record["phase"] == "rolled_back":
        _cleanup_adjustment(directory, record)
        return
    if record["phase"] != "prepared":
        raise ValueError("invalid resume adjustment phase")
    _sidecars(directory)
    for name in _FILES:
        target = directory / name
        if record["present"][name]:
            backup = directory / (".resume-backup-" + record["id"] + "-" + name)
            temporary = directory / (".resume-restore-" + name)
            shutil.copy2(backup, temporary)
            os.replace(temporary, target)
        else:
            target.unlink(missing_ok=True)
    record["phase"] = "rolled_back"
    _json_save(journal, record)
    _cleanup_adjustment(directory, record)


def _convert(directory, staging, payload, config, report):
    old = AlphaZeroConfig(**payload["config"])
    source = sqlite3.connect((directory / "replay.sqlite").as_uri() + "?mode=ro", uri=True)
    replay = ReplayBuffer(staging / "replay.sqlite", config.replay_window_games, config.replay_cache_mb,
                          window_positions=config.replay_window_positions)
    try:
        source.execute("BEGIN")
        metadata = dict(source.execute("SELECT key,value FROM metadata"))
        row = source.execute("SELECT seq FROM sqlite_sequence WHERE name='games'").fetchone()
        generation = int(row[0]) if row else 0
        recovery = payload.get("extra", {})
        model_step = payload["step"] + payload.get("optimizer_step_offset", 0)
        if (metadata.get("version") != VERSION or metadata.get("fingerprint") != old.fingerprint
                or metadata.get("run_id") != payload.get("run_id")
                or recovery.get("replay_run_id") != metadata.get("run_id")
                or recovery.get("replay_generation", -1) > generation
                or int(metadata.get("max_actor_step", 0)) > model_step):
            raise ValueError("source learner/replay identity or generation mismatch")
        for game, episode, worker, sequence, reason, start, end, raw_trace in source.execute(
                "SELECT id,episode,worker,sequence,reason,actor_start,actor_end,trace FROM games ORDER BY id"):
            trace = json.loads(raw_trace)
            samples = [PackedSample(*row) for row in source.execute(
                "SELECT planes,globals,actions,probabilities,value FROM positions WHERE game=? ORDER BY id", (game,))]
            if not samples or len(trace.get("moves", [])) != len(samples):
                raise ValueError("missing or incomplete recorded moves")
            if max(start, end) > model_step:
                raise ValueError("game uses actor newer than checkpoint")
            if (len(samples) > config.max_game_ply or
                    (config.max_game_ply > old.max_game_ply and reason == "max_ply")):
                report["removed_games"] += 1
                report["removed_positions"] += len(samples)
                continue
            error = validate_game(trace, samples, reason, old.max_game_ply)
            if error:
                raise ValueError("invalid source trajectory: " + error)
            converted = []
            for ply, sample in enumerate(samples):
                globals_ = np.frombuffer(sample.globals, dtype="<f4").copy()
                globals_[1] = ply / config.max_game_ply
                converted.append(PackedSample(sample.planes, globals_.tobytes(), sample.actions,
                                              sample.probabilities, sample.value))
            error = validate_game(trace, converted, reason, config.max_game_ply)
            if error:
                raise ValueError("target trajectory changed: " + error)
            trace.setdefault("parameter_adjustments", []).append(dict(
                source_max_game_ply=old.max_game_ply, target_max_game_ply=config.max_game_ply,
                policy_targets="historical visits; not regenerated"))
            replay.db.execute("INSERT INTO games VALUES(?,?,?,?,?,?,?,?)",
                              (game, episode, worker, sequence, reason, start, end, json.dumps(trace)))
            replay.db.executemany("INSERT INTO positions(game,planes,globals,actions,probabilities,value) VALUES(?,?,?,?,?,?)",
                                  ((game, s.planes, s.globals, s.actions, s.probabilities, s.value) for s in converted))
        metadata["fingerprint"] = config.fingerprint
        replay.db.executemany("INSERT OR REPLACE INTO metadata VALUES(?,?)", metadata.items())
        replay.db.execute("DELETE FROM sqlite_sequence WHERE name='games'")
        replay.db.execute("INSERT INTO sqlite_sequence(name,seq) VALUES('games',?)", (generation,))
        replay.db.commit()
        replay.run_id = metadata["run_id"]
        replay._refresh()
        replay.enforce_limits()
        report.update(retained_games=replay.game_count, retained_positions=len(replay), run_id=replay.run_id)
        payload.update(config=config.to_dict(), spec=config.spec, fingerprint=config.fingerprint,
                       extra=dict(payload.get("extra", {}), **replay.recovery_marker()))
        atomic_torch_save(payload, staging / "learner.pt")
        actor = {key: payload[key] for key in ("spec", "fingerprint", "config", "run_id", "step", "model_state_dict")}
        actor["step"] = model_step
        actor["config"] = dict(actor["config"])
        actor["config"].pop("replay_window_positions", None)
        actor["config"].pop("mirror_augmentation", None)
        actor["config"].pop("search_noise_depth", None)
        actor["config"].pop("new_positions_per_update", None)
        actor["config"].pop("training_microbatch_size", None)
        atomic_torch_save(dict(actor, kind="actor"), staging / "latest.pt")
        replay.flush()
    finally:
        replay.close()
        source.close()


def prepare_resume(directory, config, resume):
    """Called under the pipeline lock, before loading a model onto CUDA."""
    directory = Path(directory).resolve()
    recover_adjustment(directory)
    payload = torch.load(resume, map_location="cpu", weights_only=False)
    if payload.get("fingerprint") == config.fingerprint:
        return {}
    old = AlphaZeroConfig(**payload["config"])
    if (payload.get("kind") != "learner" or payload.get("optimizer_type") != "SGD"
            or payload.get("spec") != old.spec or payload.get("fingerprint") != old.fingerprint):
        raise ValueError("checkpoint specification mismatch")
    differences = {k for k in old.spec.keys() | config.spec.keys() if old.spec.get(k) != config.spec.get(k)}
    if differences != {"max_game_ply"}:
        raise ValueError("checkpoint model/rule/observation/target specification mismatch")
    if Path(resume).resolve() != directory / "learner.pt":
        raise ValueError("changing max_game_ply requires this run's learner.pt")
    current = json.loads(json.dumps(config.to_dict()))
    previous = json.loads(json.dumps(old.to_dict()))
    previous.setdefault("search_noise_depth", 1)
    for key in RESUME_FIXED_FIELDS:
        if previous.get(key) != current.get(key):
            raise ValueError("resume changes training semantics: " + key)
    tensors = list(payload["model_state_dict"].values()) + [v for state in payload["optimizer_state_dict"]["state"].values()
                                                           for v in state.values() if isinstance(v, torch.Tensor)]
    if not all(torch.isfinite(t).all() for t in tensors):
        raise ValueError("source has non-finite model or optimizer state")
    changes = {k: dict(old=previous.get(k), new=v) for k, v in current.items() if previous.get(k) != v}
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), step=payload["step"],
                  source_fingerprint=old.fingerprint, target_fingerprint=config.fingerprint,
                  changes=changes, removed_games=0, removed_positions=0,
                  policy_targets="historical visits; not regenerated")
    token = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix=".resume-preparing-", dir=directory) as temp:
        staging = Path(temp)
        _convert(directory, staging, payload, config, report)
        present = {name: (directory / name).exists() for name in _FILES}
        with closing(sqlite3.connect(directory / "replay.sqlite")) as source:
            if source.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]:
                raise RuntimeError("replay still busy; cannot apply parameters")
            with closing(sqlite3.connect(directory / (".resume-backup-" + token + "-replay.sqlite"))) as backup:
                source.backup(backup)
        for name in ("learner.pt", "latest.pt"):
            if present[name]:
                shutil.copy2(directory / name, directory / (".resume-backup-" + token + "-" + name))
        record = dict(version=1, id=token, phase="prepared", present=present, report=report)
        journal = directory / ".resume-adjustment.json"
        _json_save(journal, record)
        try:
            _sidecars(directory)
            for name in _FILES:
                os.replace(staging / name, directory / name)
            record["phase"] = "committed"
            _json_save(journal, record)
        except Exception:
            recover_adjustment(directory)
            raise
        _finish(directory, record)
    print("Resume parameter adjustment: " + json.dumps(report), flush=True)
    return changes
