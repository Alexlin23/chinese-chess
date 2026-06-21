"""训练器 — 从 ReplayBuffer 采样训练神经网络。

损失函数（AlphaZero 标准）:
  L = (z - v)² - πᵀ log(p) + c·||θ||²
   - value_loss:   MSE(预测价值, 真实结果)
   - policy_loss:  CrossEntropy(MCTS策略, 预测策略)
   - l2_loss:      L2 权重衰减
"""
import os
import time
import copy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .config import AlphaZeroConfig
from .replay import ReplayBuffer
from ..model import AlphaZeroNet, NeuralEvaluator
from ..engine.move import ActionEncoder


class Trainer:
    """AlphaZero 训练器。

    职责:
      - 从 ReplayBuffer 加载训练数据
      - 执行多 epoch 训练
      - 保存检查点
      - 评估新模型 vs 旧模型
    """

    def __init__(self, config: AlphaZeroConfig, device: str = 'cpu'):
        self.config = config
        self.device = device

        # 创建模型
        self.model = AlphaZeroNet(
            num_blocks=config.num_blocks,
            num_filters=config.num_filters,
        ).to(device)

        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.lr_decay_step,
            gamma=config.lr_decay_rate,
        )

        # 训练状态
        self.current_iteration = 0
        self.train_losses: list[float] = []
        self.policy_losses: list[float] = []
        self.value_losses: list[float] = []

    def train(self, replay: ReplayBuffer,
              epochs: Optional[int] = None) -> dict:
        """执行一轮训练。

        Args:
            replay: 经验回放缓冲区
            epochs: 覆盖配置中的 epochs（可选）

        Returns:
            dict: 训练统计信息
        """
        if len(replay) == 0:
            print("ReplayBuffer 为空，跳过训练")
            return {}

        epochs = epochs or self.config.epochs_per_iteration

        # 准备数据
        states, policies, results = replay.sample_all()
        dataset = TensorDataset(
            torch.from_numpy(states),
            torch.from_numpy(policies),
            torch.from_numpy(results),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=False,  # CPU 上不需要
            drop_last=True,     # 最后不完整 batch 丢弃
        )

        self.model.train()
        t_start = time.perf_counter()
        total_batches = 0

        for epoch in range(epochs):
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_total_loss = 0.0
            n_batches = 0

            for batch_states, batch_policies, batch_results in dataloader:
                batch_states = batch_states.to(self.device)
                batch_policies = batch_policies.to(self.device)
                batch_results = batch_results.to(self.device)

                # 前向传播
                logits, values = self.model(batch_states)

                # Policy loss: cross-entropy
                # batch_policies 是目标概率分布
                log_probs = F.log_softmax(logits, dim=1)
                policy_loss = -(batch_policies * log_probs).sum(dim=1).mean()

                # Value loss: MSE
                value_loss = F.mse_loss(values, batch_results)

                # 总损失（L2 正则化内置于 optimizer weight_decay）
                loss = value_loss + policy_loss

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪（防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

                epoch_total_loss += loss.item()
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                n_batches += 1
                total_batches += 1

                # 定期打印
                if total_batches % self.config.log_interval == 0:
                    print(f"\r  epoch {epoch+1}/{epochs} "
                          f"| batch {n_batches} "
                          f"| loss {loss.item():.4f} "
                          f"(p:{policy_loss.item():.4f} v:{value_loss.item():.4f})",
                          end="")

            # Epoch 统计
            avg_total = epoch_total_loss / max(n_batches, 1)
            avg_policy = epoch_policy_loss / max(n_batches, 1)
            avg_value = epoch_value_loss / max(n_batches, 1)
            self.train_losses.append(avg_total)
            self.policy_losses.append(avg_policy)
            self.value_losses.append(avg_value)

            print(f"\r  epoch {epoch+1}/{epochs} "
                  f"| loss {avg_total:.4f} "
                  f"(p:{avg_policy:.4f} v:{avg_value:.4f})")

        # 更新学习率
        self.scheduler.step()

        elapsed = time.perf_counter() - t_start
        stats = {
            'epochs': epochs,
            'batches': total_batches,
            'avg_loss': np.mean(self.train_losses[-epochs:]),
            'avg_policy_loss': np.mean(self.policy_losses[-epochs:]),
            'avg_value_loss': np.mean(self.value_losses[-epochs:]),
            'lr': self.scheduler.get_last_lr()[0],
            'time': elapsed,
        }

        print(f"训练完成: {elapsed:.0f}s | "
              f"loss {stats['avg_loss']:.4f} | "
              f"lr {stats['lr']:.6f}")

        self.current_iteration += 1
        return stats

    def save_checkpoint(self, path: Optional[str] = None,
                        extra: Optional[dict] = None) -> str:
        """保存模型检查点。

        Args:
            path:  保存路径（None = 自动生成）
            extra: 额外保存的元数据

        Returns:
            实际保存路径
        """
        if path is None:
            path = self.config.checkpoint_path / \
                   f"model_iter{self.current_iteration:03d}.pt"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'iteration': self.current_iteration,
            'config': self.config.to_dict(),
            'train_losses': self.train_losses,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
        }
        if extra:
            checkpoint.update(extra)

        torch.save(checkpoint, str(path))
        print(f"Checkpoint 已保存: {path}")
        return str(path)

    def load_checkpoint(self, path: str) -> dict:
        """加载模型检查点。

        Returns:
            检查点中的额外元数据
        """
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        if 'scheduler_state_dict' in state:
            self.scheduler.load_state_dict(state['scheduler_state_dict'])
        self.current_iteration = state.get('iteration', 0)
        self.train_losses = state.get('train_losses', [])
        self.policy_losses = state.get('policy_losses', [])
        self.value_losses = state.get('value_losses', [])

        print(f"加载 checkpoint: {path} (iteration {self.current_iteration})")
        return state

    def evaluate_against(self, old_model_path: str,
                         evaluator: NeuralEvaluator,
                         num_games: int = 10) -> dict:
        """新模型 vs 旧模型对战评估。

        Args:
            old_model_path: 旧模型权重路径
            evaluator:      当前最佳评估器
            num_games:      对战局数

        Returns:
            {'wins': int, 'losses': int, 'draws': int, 'win_rate': float}
        """
        # TODO: 实现 Arena 对战评估
        return {'wins': 0, 'losses': 0, 'draws': 0, 'win_rate': 0.0}


# ── CLI 入口 ──

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AlphaZero 训练")
    parser.add_argument("--data", type=str, required=True,
                        help="ReplayBuffer .npz 文件路径")
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--filters", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="预训练权重加载")
    parser.add_argument("--output", type=str,
                        default="AlphaZero/checkpoints/model.pt")
    args = parser.parse_args()

    config = AlphaZeroConfig(
        num_blocks=args.blocks,
        num_filters=args.filters,
        epochs_per_iteration=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # 加载数据
    print(f"加载数据: {args.data}")
    replay = ReplayBuffer.from_file(args.data)

    # 训练
    trainer = Trainer(config)
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    stats = trainer.train(replay)
    trainer.save_checkpoint(args.output, extra={'train_stats': stats})
