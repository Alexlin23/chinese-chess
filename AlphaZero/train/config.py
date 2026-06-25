"""AlphaZero 训练超参数 — 支持 YAML 配置文件"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AlphaZeroConfig:
    """AlphaZero 训练配置"""

    # ── 模型架构 ──
    num_blocks: int = 20
    num_filters: int = 256
    input_channels: int = 18
    policy_size: int = 8100
    wdl_size: int = 3

    # ── MCTS 搜索 ──
    num_simulations: int = 800
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    mcts_batch_size: int = 256

    # ── 自我对弈 ──
    temperature_ply: int = 30
    games_per_iteration: int = 25000
    max_game_ply: int = 512
    repetition_limit: int = 3

    # ── 训练 ──
    batch_size: int = 512
    learning_rate: float = 0.2
    lr_momentum: float = 0.9
    lr_decay_step: int = 100000
    lr_decay_rate: float = 0.1
    weight_decay: float = 1e-4
    epochs_per_iteration: int = 10
    replay_buffer_size: int = 1_000_000
    samples_per_epoch: int = 100000

    # ── 奖励权重 ──
    capture_reward: float = 0.3        # 吃子奖励（远小于赢棋的 1.0）
    non_terminal_base_weight: float = 0.001  # 非终局位置基础权重
    terminal_weight: float = 2.0       # 终局位置权重

    # ── Arena ──
    arena_games: int = 400
    promotion_score_rate: float = 0.55

    # ── 并行 ──
    num_workers: Optional[int] = None  # None = os.cpu_count()
    inference_batch_size: int = 256
    inference_wait_ms: float = 10.0

    # ── 系统 ──
    checkpoint_dir: str = "AlphaZero/checkpoints"
    log_interval: int = 100
    save_interval: int = 1
    monitor_interval: float = 5.0

    # ── 训练循环 ──
    max_iterations: int = 100

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.checkpoint_dir)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}

    @classmethod
    def from_yaml(cls, path: str) -> 'AlphaZeroConfig':
        """从 YAML 文件加载配置"""
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        config = cls()

        # 模型
        if 'model' in data:
            config.num_blocks = data['model'].get('blocks', config.num_blocks)
            config.num_filters = data['model'].get('filters', config.num_filters)
            config.input_channels = data['model'].get('input_channels', config.input_channels)

        # MCTS
        if 'mcts' in data:
            config.num_simulations = data['mcts'].get('simulations', config.num_simulations)
            config.c_puct = data['mcts'].get('c_puct', config.c_puct)
            config.dirichlet_alpha = data['mcts'].get('dirichlet_alpha', config.dirichlet_alpha)
            config.dirichlet_epsilon = data['mcts'].get('dirichlet_epsilon', config.dirichlet_epsilon)
            config.mcts_batch_size = data['mcts'].get('batch_size', config.mcts_batch_size)

        # 自我对弈
        if 'self_play' in data:
            config.games_per_iteration = data['self_play'].get('games', config.games_per_iteration)
            config.max_game_ply = data['self_play'].get('max_ply', config.max_game_ply)
            config.temperature_ply = data['self_play'].get('temperature_ply', config.temperature_ply)

        # 训练
        if 'training' in data:
            config.batch_size = data['training'].get('batch_size', config.batch_size)
            config.learning_rate = data['training'].get('lr', config.learning_rate)
            config.lr_momentum = data['training'].get('lr_momentum', config.lr_momentum)
            config.weight_decay = data['training'].get('weight_decay', config.weight_decay)
            config.epochs_per_iteration = data['training'].get('epochs', config.epochs_per_iteration)
            config.replay_buffer_size = data['training'].get('replay_buffer', config.replay_buffer_size)
            config.samples_per_epoch = data['training'].get('samples_per_epoch', config.samples_per_epoch)
            config.capture_reward = data['training'].get('capture_reward', config.capture_reward)
            config.non_terminal_base_weight = data['training'].get('non_terminal_base_weight', config.non_terminal_base_weight)
            config.terminal_weight = data['training'].get('terminal_weight', config.terminal_weight)

        # Arena
        if 'arena' in data:
            config.arena_games = data['arena'].get('games', config.arena_games)
            config.promotion_score_rate = data['arena'].get('threshold', config.promotion_score_rate)

        # 并行
        if 'parallel' in data:
            config.num_workers = data['parallel'].get('workers', config.num_workers)
            config.inference_batch_size = data['parallel'].get('inference_batch', config.inference_batch_size)
            config.inference_wait_ms = data['parallel'].get('inference_wait_ms', config.inference_wait_ms)

        # 系统
        if 'system' in data:
            config.checkpoint_dir = data['system'].get('checkpoint_dir', config.checkpoint_dir)
            config.log_interval = data['system'].get('log_interval', config.log_interval)
            config.save_interval = data['system'].get('save_interval', config.save_interval)
            config.monitor_interval = data['system'].get('monitor_interval', config.monitor_interval)

        # 训练循环
        if 'pipeline' in data:
            config.max_iterations = data['pipeline'].get('max_iterations', config.max_iterations)

        return config
