"""AlphaZero 训练超参数 — 针对 CPU 优化"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AlphaZeroConfig:
    """AlphaZero 训练配置。

    所有参数均可覆盖，适合逐步调优。
    """

    # ── 模型架构 ──
    num_blocks: int = 20          # 残差块数量（AlphaZero 原版 20）
    num_filters: int = 128        # 卷积通道数（原版 256，CPU 版降为 128）

    # ── MCTS 搜索 ──
    num_simulations: int = 200    # 每次搜索的模拟次数（GPU 用 800，CPU 用 200）
    c_puct: float = 1.5           # PUCT 探索系数
    dirichlet_alpha: float = 0.3  # Dirichlet 噪声浓度（中国象棋合法走法多，略低）
    dirichlet_epsilon: float = 0.25  # 噪声混合比例

    # ── 自我对弈 ──
    temperature_threshold: int = 30   # 前 N 步用 τ=1（探索），之后 τ→0（贪心）
    games_per_iteration: int = 50     # 每轮迭代生成对局数
    max_game_length: int = 200        # 单局最大步数（防止无限循环）

    # ── 训练 ──
    batch_size: int = 128             # 训练批次大小（CPU 优化）
    learning_rate: float = 0.001      # 初始学习率（Adam）
    lr_decay_step: int = 5            # 每 N 轮迭代学习率衰减
    lr_decay_rate: float = 0.5        # 衰减倍率
    weight_decay: float = 1e-4        # L2 正则化系数
    epochs_per_iteration: int = 10    # 每轮训练 epoch 数
    replay_buffer_size: int = 100_000  # 经验回放缓冲区最大容量

    # ── 系统 ──
    num_workers: int = 4              # DataLoader 线程数
    checkpoint_dir: str = "AlphaZero/checkpoints"
    data_dir: str = "AlphaZero/data"  # self-play 数据存放
    eval_games: int = 10              # 评估新模型时对弈局数
    win_rate_threshold: float = 0.55  # 替换旧模型的胜率阈值

    # ── 日志 ──
    log_interval: int = 10            # 训练每 N batch 打印一次
    save_interval: int = 5            # 每 N 轮保存一次 checkpoint

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.checkpoint_dir)

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    def to_dict(self) -> dict:
        """转为普通 dict，便于序列化到 checkpoint。"""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}


# ── 预设配置 ──

def quick_config() -> AlphaZeroConfig:
    """快速测试配置（用于验证管道通畅）。"""
    return AlphaZeroConfig(
        num_blocks=5,
        num_filters=64,
        num_simulations=50,
        games_per_iteration=5,
        epochs_per_iteration=2,
        batch_size=64,
        replay_buffer_size=10_000,
    )


def cpu_config() -> AlphaZeroConfig:
    """CPU 优化配置（完整训练用）。"""
    return AlphaZeroConfig(
        num_blocks=20,
        num_filters=128,
        num_simulations=200,
        games_per_iteration=50,
        epochs_per_iteration=10,
        batch_size=128,
    )


def full_config() -> AlphaZeroConfig:
    """完整配置（需要 GPU）。"""
    return AlphaZeroConfig(
        num_blocks=20,
        num_filters=256,
        num_simulations=800,
        games_per_iteration=100,
        epochs_per_iteration=10,
        batch_size=512,
    )
