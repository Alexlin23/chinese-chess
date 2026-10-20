"""Versioned AlphaZero specification. Hardware changes belong to model/runtime."""
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import hashlib
import json
import math
import yaml
from ..engine.constants import NATURAL_LIMIT_ROUNDS, NATURAL_LIMIT_CHECK_CAP, POLICY_PLANES

SCHEMA_VERSION = 3
ALGORITHM_VERSION = "alphazero-datas1-aligned-v3"
OBSERVATION_VERSION = "xiangqi-history8-131-v2"
ACTION_VERSION = "absolute-displacement-50x10x9-v1"
POLICY_HEAD_VERSION = "conv3x3-bn-relu-conv3x3-v1"
RULE_VERSION = "xiangqi-development-v3-natural-limit"  # CNXA 2020 check cap; chase remains incomplete.
TARGET_VERSION = "scalar-mse-visits-v1"

@dataclass
class AlphaZeroConfig:
    num_blocks: int = 19
    num_filters: int = 128
    input_channels: int = 131
    num_simulations: int = 800
    c_puct: float = 1.25
    pb_c_base: float = 19652.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    num_sampling_moves: int = 30
    reuse_tree: bool = False
    mcts_batch_size: int = 1
    virtual_loss: float = 0.0
    min_replay_games: int = 1
    max_game_ply: int = 512
    batch_size: int = 4096
    loss_reduction: str = "mean"  # Paper per-position objective; Data S1 literal sum is an audit mode.
    learning_rate: float = 0.2
    lr_momentum: float = 0.9
    lr_milestones: tuple = (100000, 300000, 500000)
    lr_decay_rate: float = 0.1
    weight_decay: float = 0.0001  # Data S1 c * tf.nn.l2_loss(theta) = c/2 * sum(theta**2).
    checkpoint_interval: int = 1000
    recovery_interval: int = 100  # Learner recovery only; does not publish actor weights.
    max_optimizer_steps: int = 700000
    replay_window_games: int = 1000000
    replay_cache_mb: int = 256
    num_workers: int = 8
    actor_processes: int = 8
    inference_batch_size: int = 128
    inference_wait_ms: float = 2.0
    checkpoint_dir: str = "AlphaZero/runs/policy-planes-v1"
    max_iterations: int = 700
    seed: int = 20261015
    precision: str = "bf16"
    channels_last: bool = True
    compile_model: bool = False
    inference_cuda_graph: bool = True
    device: str = "cuda"
    torch_threads: int = 1
    telemetry_host: str = "127.0.0.1"
    telemetry_port: int = 8768
    telemetry_interval: float = 0.25

    def __post_init__(self):
        for key in ("learning_rate", "lr_momentum", "weight_decay", "c_puct", "pb_c_base",
                    "virtual_loss", "dirichlet_alpha", "dirichlet_epsilon", "lr_decay_rate",
                    "inference_wait_ms", "telemetry_interval"):
            if not isinstance(getattr(self, key), (int, float)) or not math.isfinite(getattr(self, key)):
                raise ValueError(f"{key} must be finite")
        for key in ("num_filters", "input_channels", "num_simulations", "mcts_batch_size",
                    "min_replay_games", "max_game_ply", "batch_size", "checkpoint_interval", "recovery_interval",
                    "max_optimizer_steps", "replay_window_games", "replay_cache_mb",
                    "num_workers", "actor_processes", "inference_batch_size",
                    "max_iterations", "torch_threads"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.num_blocks < 1 or self.input_channels != 131:
            raise ValueError("this observation version requires 131 channels and positive blocks")
        if type(self.num_sampling_moves) is not int or self.num_sampling_moves < 0:
            raise ValueError("num_sampling_moves must be a nonnegative integer")
        if type(self.reuse_tree) is not bool:
            raise ValueError("reuse_tree must be boolean")
        if self.mcts_batch_size > self.inference_batch_size:
            raise ValueError("mcts batch cannot exceed inference batch")
        if self.loss_reduction not in ("mean", "sum"):
            raise ValueError("loss_reduction must be mean or sum")
        if self.compile_model and self.inference_cuda_graph:
            raise ValueError("choose CUDA graphs or torch.compile, not both")
        if self.precision not in ("fp32", "bf16") or self.device not in ("cpu", "cuda"):
            raise ValueError("unsupported precision/device")
        if self.learning_rate <= 0 or self.weight_decay < 0 or not 0 <= self.lr_momentum < 1:
            raise ValueError("invalid SGD settings")
        if self.pb_c_base <= 0 or self.c_puct < 0 or self.virtual_loss < 0:
            raise ValueError("invalid search settings")
        if self.dirichlet_alpha <= 0 or not 0 <= self.dirichlet_epsilon <= 1:
            raise ValueError("invalid root noise")
        self.lr_milestones = tuple(self.lr_milestones)
        if any(type(x) is not int or x < 1 for x in self.lr_milestones):
            raise ValueError("invalid learning-rate milestones")
        if tuple(sorted(set(self.lr_milestones))) != self.lr_milestones:
            raise ValueError("milestones must increase strictly")
        if not 0 < self.lr_decay_rate <= 1 or self.inference_wait_ms < 0:
            raise ValueError("invalid decay/wait")
        if not 0 <= self.telemetry_port <= 65535 or self.telemetry_interval <= 0:
            raise ValueError("invalid telemetry settings")

    @property
    def checkpoint_path(self):
        return Path(self.checkpoint_dir)

    def to_dict(self):
        return asdict(self)

    @property
    def spec(self):
        return dict(schema=SCHEMA_VERSION, algorithm=ALGORITHM_VERSION,
                    loss_reduction=self.loss_reduction, observation=OBSERVATION_VERSION,
                    action=ACTION_VERSION, rule=RULE_VERSION, target=TARGET_VERSION,
                    policy_head=POLICY_HEAD_VERSION, policy_planes=POLICY_PLANES,
                    blocks=self.num_blocks, filters=self.num_filters,
                    channels=self.input_channels, max_game_ply=self.max_game_ply,
                    no_capture_plies=2 * NATURAL_LIMIT_ROUNDS,
                    natural_limit_rounds=NATURAL_LIMIT_ROUNDS,
                    natural_limit_check_cap=NATURAL_LIMIT_CHECK_CAP,
                    natural_limit_claim="side-to-move-auto", repetition_limit=3)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.spec, sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_yaml(cls, path):
        with open(path, encoding="utf-8-sig") as stream:
            values = yaml.safe_load(stream)
        if not isinstance(values, dict):
            raise ValueError("configuration must be a mapping")
        allowed = {f.name for f in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown or obsolete configuration fields: {sorted(unknown)}")
        return cls(**values)
