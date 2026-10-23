"""Disk-backed FIFO of complete games. Uniform position sampling, bounded SQLite cache."""
from dataclasses import dataclass
from pathlib import Path
import os
import sqlite3
import threading
import uuid
import numpy as np
from .config import OBSERVATION_VERSION, ACTION_VERSION, RULE_VERSION, TARGET_VERSION, ALGORITHM_VERSION
from ..engine.constants import ROWS, COLS, POLICY_SIZE
from ..engine.move import ActionEncoder

VERSION = "|".join((ALGORITHM_VERSION, OBSERVATION_VERSION, ACTION_VERSION, RULE_VERSION, TARGET_VERSION))

# Reflect both the displacement plane and origin; padding maps to padding.
_MIRROR_ACTION_IDS = np.array([ActionEncoder.mirror_action(a) for a in range(POLICY_SIZE)],
                            dtype=np.uint16)
_MIRROR_ACTION_IDS.flags.writeable = False

@dataclass
class PackedSample:
    planes: bytes
    globals: bytes
    actions: bytes
    probabilities: bytes
    value: float

    @property
    def nbytes(self):
        return len(self.planes) + len(self.globals) + len(self.actions) + len(self.probabilities) + 128

    def validate(self):
        actions = np.frombuffer(self.actions, dtype="<u2")
        probabilities = np.frombuffer(self.probabilities, dtype="<f4")
        if len(self.planes) != 1440 or len(self.globals) != 12:
            raise ValueError("wrong observation format")
        if (not len(actions) or len(actions) != len(probabilities)
                or np.any(actions >= POLICY_SIZE) or len(np.unique(actions)) != len(actions)
                or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
                or not np.isclose(probabilities.sum(), 1, atol=1e-5)):
            raise ValueError("invalid sparse policy")
        if not np.isfinite(self.value) or abs(self.value) > 1:
            raise ValueError("invalid scalar target")
        if not np.isfinite(np.frombuffer(self.globals, dtype="<f4")).all():
            raise ValueError("invalid global planes")

def pack_sample(state, policy, value, weight=1.0):
    policy = np.asarray(policy, dtype=np.float32)
    if (policy.shape != (POLICY_SIZE,) or not np.isfinite(policy).all()
            or np.any(policy < 0) or not np.isclose(policy.sum(), 1, atol=1e-5)):
        raise ValueError("policy must be a finite normalized nonnegative vector")
    if weight != 1:
        raise ValueError("paper baseline uses uniform sampling without sample weights")
    if np.any(policy[~state.legal_mask()] > 0):
        raise ValueError("policy target contains illegal or off-board actions")
    value = np.asarray(value)
    if value.shape == (3,):  # Import helper, never a learned WDL training objective.
        value = value[0] - value[2]
    canonical = state.canonical_policy(policy)
    actions = np.flatnonzero(canonical)
    encoding = state.encode()
    packed = PackedSample(np.packbits(encoding[:128].astype(np.uint8)).tobytes(),
                          encoding[128:, 0, 0].astype("<f4").tobytes(),
                          actions.astype("<u2").tobytes(),
                          canonical[actions].astype("<f4").tobytes(), float(value))
    packed.validate()
    return packed

class ReplayBuffer:
    def __init__(self, path=":memory:", window_games=1000000, cache_mb=256, *, window_positions=0):
        self.path, self.window_games = str(path), window_games
        self.window_positions = window_positions
        if (window_games < 1 or cache_mb < 1 or type(window_positions) is not int
                or window_positions < 0):
            raise ValueError("replay limits must be positive")
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute(f"PRAGMA cache_size={-cache_mb * 1024}")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS games(
                id INTEGER PRIMARY KEY AUTOINCREMENT, episode TEXT UNIQUE NOT NULL,
                worker INTEGER NOT NULL, sequence INTEGER NOT NULL, reason TEXT NOT NULL,
                actor_start INTEGER NOT NULL, actor_end INTEGER NOT NULL, trace TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS positions(
                id INTEGER PRIMARY KEY AUTOINCREMENT, game INTEGER NOT NULL REFERENCES games(id) ON DELETE CASCADE,
                planes BLOB NOT NULL, globals BLOB NOT NULL, actions BLOB NOT NULL,
                probabilities BLOB NOT NULL, value REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS positions_game ON positions(game);
        """)
        row = self.db.execute("SELECT value FROM metadata WHERE key='version'").fetchone()
        if row and row[0] != VERSION:
            raise ValueError("incompatible replay schema/rules/targets")
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES('version',?)", (VERSION,))
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES('run_id',?)", (str(uuid.uuid4()),))
        self.db.commit()
        self.run_id = self.db.execute("SELECT value FROM metadata WHERE key='run_id'").fetchone()[0]
        self._refresh()

    def _refresh(self):
        self.count, self.low, self.high = self.db.execute(
            "SELECT COUNT(*),MIN(id),MAX(id) FROM positions").fetchone()
        self.game_count = self.db.execute("SELECT COUNT(*) FROM games").fetchone()[0]
        if self.count and self.high - self.low + 1 != self.count:
            raise ValueError("replay position IDs are not contiguous")
        row = self.db.execute("SELECT seq FROM sqlite_sequence WHERE name='games'").fetchone()
        self.generation = int(row[0]) if row else 0
        row = self.db.execute("SELECT value FROM metadata WHERE key='max_actor_step'").fetchone()
        self.max_actor_step = int(row[0]) if row else 0
        row = self.db.execute("SELECT value FROM metadata WHERE key='total_positions'").fetchone()
        self.total_positions = int(row[0]) if row else self.count

    def bind_spec(self, fingerprint):
        with self.lock, self.db:
            row = self.db.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()
            if row and row[0] != fingerprint:
                raise ValueError("replay run specification mismatch")
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES('fingerprint',?)", (fingerprint,))

    def recovery_marker(self):
        with self.lock:
            return dict(replay_generation=self.generation, replay_run_id=self.run_id,
                        recovery_policy="keep committed replay; resume learner step")

    def next_sequences(self):
        with self.lock:
            return {int(key.split(':')[1]): int(value) for key, value in self.db.execute(
                "SELECT key,value FROM metadata WHERE key LIKE 'worker:%'")}

    def _prune_games(self, newest_game, high):
        """Called inside the write transaction; retain a contiguous suffix of complete games."""
        removed = self.db.execute("DELETE FROM games WHERE id<=?",
                                  (newest_game - self.window_games,)).rowcount
        if self.window_positions and high is not None:
            boundary = self.db.execute(
                "SELECT game FROM positions WHERE id<=? ORDER BY id DESC LIMIT 1",
                (high - self.window_positions,)).fetchone()
            if boundary:
                removed += self.db.execute("DELETE FROM games WHERE id<=?", (boundary[0],)).rowcount
        return removed

    def enforce_limits(self):
        """Apply retention after checkpoint/run validation, before sampling."""
        with self.lock, self.db:
            before_games, before_positions = self.game_count, self.count
            if self.window_positions and self.count:
                newest_size = self.db.execute("SELECT COUNT(*) FROM positions WHERE game=?",
                                              (self.generation,)).fetchone()[0]
                if newest_size > self.window_positions:
                    raise ValueError("newest complete game exceeds replay position cap")
            self._prune_games(self.generation, self.high)
            self._refresh()
            return dict(removed_games=before_games - self.game_count,
                        removed_positions=before_positions - self.count,
                        retained_games=self.game_count, retained_positions=self.count)

    def add_game(self, episode, samples, *, worker=0, sequence=0, reason="unknown",
                 actor_start=0, actor_end=0, trace="[]"):
        if not samples:
            raise ValueError("cannot commit empty episode")
        if self.window_positions and len(samples) > self.window_positions:
            raise ValueError("complete game exceeds replay position cap")
        for sample in samples:
            sample.validate()
        with self.lock, self.db:
            if self.db.execute("SELECT 1 FROM games WHERE episode=?", (episode,)).fetchone():
                return False
            game = self.db.execute("INSERT INTO games(episode,worker,sequence,reason,actor_start,actor_end,trace) "
                                   "VALUES(?,?,?,?,?,?,?)",
                                   (episode, worker, sequence, reason, actor_start, actor_end, trace)).lastrowid
            self.db.execute("INSERT OR REPLACE INTO metadata VALUES(?,?)",
                            (f"worker:{worker}", str(sequence + 1)))
            max_actor_step = max(self.max_actor_step, actor_end)
            self.db.execute("INSERT OR REPLACE INTO metadata VALUES('max_actor_step',?)",
                            (str(max_actor_step),))
            total_positions = self.total_positions + len(samples)
            self.db.execute("INSERT OR REPLACE INTO metadata VALUES('total_positions',?)",
                            (str(total_positions),))
            self.db.executemany("INSERT INTO positions(game,planes,globals,actions,probabilities,value) "
                                "VALUES(?,?,?,?,?,?)",
                                ((game, s.planes, s.globals, s.actions, s.probabilities, s.value)
                                 for s in samples))
            self.high = self.db.execute("SELECT id FROM positions ORDER BY id DESC LIMIT 1").fetchone()[0]
            removed = self._prune_games(game, self.high)
            # SQLite COUNT(*) would scan the full replay every game.
            self.low = self.db.execute("SELECT id FROM positions ORDER BY id LIMIT 1").fetchone()[0]
            self.count = self.high - self.low + 1 if self.high else 0
            self.game_count += 1 - removed
            self.generation = game
            self.max_actor_step = max_actor_step
            self.total_positions = total_positions
        return True

    def sample(self, n, rng=None, *, mirror_augmentation=False):
        if n < 1:
            raise ValueError("batch size must be positive")
        if type(mirror_augmentation) is not bool:
            raise ValueError("mirror_augmentation must be boolean")
        rng = rng if rng is not None else np.random
        with self.lock:
            if not self.count:
                raise ValueError("replay is empty")
            ids = (rng.integers(self.low, self.high + 1, size=n) if hasattr(rng, "integers")
                   else rng.randint(self.low, self.high + 1, size=n))
            # Uniformly sample the virtual originals-plus-mirrors pool.
            # Disabled augmentation consumes no additional random numbers.
            reflected = ((rng.integers(0, 2, size=n) if hasattr(rng, "integers")
                          else rng.randint(0, 2, size=n)).astype(bool)
                         if mirror_augmentation else None)
            unique = np.unique(ids)
            rows = {}
            for start in range(0, len(unique), 500):
                chunk = [int(i) for i in unique[start:start+500]]
                sql = "SELECT id,planes,globals,actions,probabilities,value FROM positions WHERE id IN (" + ",".join("?" * len(chunk)) + ")"
                rows.update({row[0]: row[1:] for row in self.db.execute(sql, chunk)})
        states = np.empty((n, 131, 10, 9), dtype=np.float32)
        policies = np.zeros((n, POLICY_SIZE), dtype=np.float32)
        values = np.empty(n, dtype=np.float32)
        # Decode each sampled position once; repeated samples still occupy their
        # original batch slots and keep exactly the same uniform sampling law.
        order = np.argsort(ids)
        boundaries = np.flatnonzero(np.diff(ids[order])) + 1
        for destinations in np.split(order, boundaries):
            planes, globals_, actions, probabilities, value = rows[int(ids[destinations[0]])]
            PackedSample(planes, globals_, actions, probabilities, value).validate()
            decoded = np.unpackbits(np.frombuffer(planes, dtype=np.uint8)).reshape(128,10,9)
            actions = np.frombuffer(actions, dtype="<u2")
            probabilities = np.frombuffer(probabilities, dtype="<f4")
            if reflected is None:
                states[destinations, :128] = decoded
                policies[destinations[:,None], actions[None,:]] = probabilities
            else:
                normal_slots = destinations[~reflected[destinations]]
                mirror_slots = destinations[reflected[destinations]]
                states[normal_slots, :128] = decoded
                states[mirror_slots, :128] = decoded[:, :, ::-1]
                policies[normal_slots[:,None], actions[None,:]] = probabilities
                policies[mirror_slots[:,None], _MIRROR_ACTION_IDS[actions][None,:]] = probabilities
            states[destinations, 128:] = np.frombuffer(globals_, dtype="<f4")[:,None,None]
            values[destinations] = value
        return states, policies, values, np.ones(n, dtype=np.float32)

    def __len__(self):
        with self.lock:
            return self.count

    def flush(self):
        with self.lock:
            self.db.execute("PRAGMA wal_checkpoint(PASSIVE)")

    def close(self):
        with self.lock:
            self.db.close()
