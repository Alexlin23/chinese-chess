"""训练器 — 从 ReplayBuffer 采样训练神经网络。

损失函数:
  L = policy_loss + wdl_loss + weight_decay
  - policy_loss: CrossEntropy(MCTS_policy, predicted_policy)
  - wdl_loss: CrossEntropy(wdl_target, wdl_logits)
"""
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .config import AlphaZeroConfig
from .replay import ReplayBuffer
from ..model import PolicyWDLEncoder


class Trainer:
    """AlphaZero 训练器。

    职责:
      - 从 ReplayBuffer 加载训练数据
      - 执行多 epoch 训练
      - 保存检查点
    """

    def __init__(self, config: AlphaZeroConfig, device: str = 'cpu'):
        self.config = config
        self.device = device

        # 创建模型
        self.model = PolicyWDLEncoder(
            num_blocks=config.num_blocks,
            num_filters=config.num_filters,
        ).to(device)

        # 优化器 — SGD + Momentum（对齐 AlphaZero 论文）
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=config.lr_momentum,
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
        self.wdl_losses: list[float] = []

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
        samples_per_epoch = self.config.samples_per_epoch

        self.model.train()
        t_start = time.perf_counter()
        total_batches = 0

        for epoch in range(epochs):
            epoch_policy_loss = 0.0
            epoch_wdl_loss = 0.0
            epoch_total_loss = 0.0
            n_batches = 0

            # 随机采样
            states, policies, wdls, weights = replay.sample(
                min(samples_per_epoch, len(replay))
            )
            dataset = TensorDataset(
                torch.from_numpy(states),
                torch.from_numpy(policies),
                torch.from_numpy(wdls),
                torch.from_numpy(weights),
            )
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=min(4, os.cpu_count() or 1),
                pin_memory=True if self.device == 'cuda' else False,
                drop_last=True,
            )

            for batch_states, batch_policies, batch_wdls, batch_weights in dataloader:
                batch_states = batch_states.to(self.device)
                batch_policies = batch_policies.to(self.device)
                batch_wdls = batch_wdls.to(self.device)
                batch_weights = batch_weights.to(self.device)

                # 前向传播
                p_logits, w_logits = self.model(batch_states)

                # Policy loss: cross-entropy (加权)
                log_probs = F.log_softmax(p_logits, dim=1)
                policy_loss = -(batch_policies * log_probs).sum(dim=1)
                policy_loss = (policy_loss * batch_weights).mean()

                # WDL loss: cross-entropy (加权)
                wdl_loss = F.cross_entropy(w_logits, batch_wdls, reduction='none')
                wdl_loss = (wdl_loss * batch_weights).mean()

                # 总损失
                loss = policy_loss + wdl_loss

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

                epoch_total_loss += loss.item()
                epoch_policy_loss += policy_loss.item()
                epoch_wdl_loss += wdl_loss.item()
                n_batches += 1
                total_batches += 1

                # 定期打印
                if total_batches % self.config.log_interval == 0:
                    print(f"\r  epoch {epoch+1}/{epochs} "
                          f"| batch {n_batches} "
                          f"| loss {loss.item():.4f} "
                          f"(p:{policy_loss.item():.4f} w:{wdl_loss.item():.4f})",
                          end="")

            # Epoch 统计
            avg_total = epoch_total_loss / max(n_batches, 1)
            avg_policy = epoch_policy_loss / max(n_batches, 1)
            avg_wdl = epoch_wdl_loss / max(n_batches, 1)
            self.train_losses.append(avg_total)
            self.policy_losses.append(avg_policy)
            self.wdl_losses.append(avg_wdl)

            print(f"\r  epoch {epoch+1}/{epochs} "
                  f"| loss {avg_total:.4f} "
                  f"(p:{avg_policy:.4f} w:{avg_wdl:.4f})")

        # 更新学习率
        self.scheduler.step()

        elapsed = time.perf_counter() - t_start
        stats = {
            'epochs': epochs,
            'batches': total_batches,
            'avg_loss': np.mean(self.train_losses[-epochs:]),
            'avg_policy_loss': np.mean(self.policy_losses[-epochs:]),
            'avg_wdl_loss': np.mean(self.wdl_losses[-epochs:]),
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
        """保存模型检查点。"""
        if path is None:
            path = self.config.checkpoint_path / \
                   f"model_iter{self.current_iteration:03d}.pt"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.to_dict(),
            'train_losses': self.train_losses,
            'policy_losses': self.policy_losses,
            'wdl_losses': self.wdl_losses,
        }
        if extra:
            checkpoint.update(extra)

        torch.save(checkpoint, str(path))
        print(f"Checkpoint 已保存: {path}")
        return str(path)

    def load_checkpoint(self, path: str) -> dict:
        """加载模型检查点（兼容旧 Adam checkpoint 的优化器状态）。"""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state['model_state_dict'])

        # 优化器状态：兼容旧版 Adam → 新版 SGD 迁移
        try:
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
        except (KeyError, ValueError, RuntimeError):
            print("  优化器状态不兼容（旧版 Adam checkpoint），使用全新优化器从头开始")

        if 'scheduler_state_dict' in state:
            try:
                self.scheduler.load_state_dict(state['scheduler_state_dict'])
            except (KeyError, ValueError, RuntimeError):
                pass

        self.train_losses = state.get('train_losses', [])
        self.policy_losses = state.get('policy_losses', [])
        self.wdl_losses = state.get('wdl_losses', [])

        print(f"加载 checkpoint: {path}")
        return state
