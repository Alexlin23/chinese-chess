"""经验回放缓冲区 — 存储 (state, policy, wdl, weight) 训练样本"""
import numpy as np
from pathlib import Path
from typing import Optional


class ReplayBuffer:
    """经验回放缓冲区

    存储格式:
      - states: (N, 18, 10, 9) float32
      - policies: (N, 8100) float32
      - wdls: (N, 3) float32  [win, draw, loss]
      - weights: (N,) float32  训练权重
    """

    def __init__(self, max_size: int = 100_000):
        self.max_size = max_size
        self.states = None
        self.policies = None
        self.wdls = None
        self.weights = None
        self.size = 0
        self.position = 0

    def add(self, state: np.ndarray, policy: np.ndarray, wdl: np.ndarray,
            weight: float = 1.0):
        """添加一个样本

        Args:
            state: (18, 10, 9) float32
            policy: (8100,) float32
            wdl: (3,) float32 [win, draw, loss]
            weight: float 训练权重
        """
        if self.states is None:
            # 延迟初始化
            self.states = np.zeros((self.max_size, *state.shape), dtype=np.float32)
            self.policies = np.zeros((self.max_size, *policy.shape), dtype=np.float32)
            self.wdls = np.zeros((self.max_size, *wdl.shape), dtype=np.float32)
            self.weights = np.ones(self.max_size, dtype=np.float32)

        self.states[self.position] = state
        self.policies[self.position] = policy
        self.wdls[self.position] = wdl
        self.weights[self.position] = weight

        self.position = (self.position + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """随机采样 n 个样本（按权重加权采样）

        Returns:
            states: (n, 18, 10, 9) float32
            policies: (n, 8100) float32
            wdls: (n, 3) float32
            weights: (n,) float32
        """
        if self.size == 0:
            raise ValueError("Buffer is empty")

        # 按权重采样
        probs = self.weights[:self.size]
        probs = probs / probs.sum()
        indices = np.random.choice(self.size, size=n, replace=True, p=probs)

        return (self.states[indices],
                self.policies[indices],
                self.wdls[indices],
                self.weights[indices])

    def __len__(self) -> int:
        return self.size

    def save(self, path: str):
        """保存到文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            states=self.states[:self.size],
            policies=self.policies[:self.size],
            wdls=self.wdls[:self.size],
            weights=self.weights[:self.size],
        )
        print(f"ReplayBuffer 已保存: {path} ({self.size} 样本)")

    @classmethod
    def load(cls, path: str, max_size: Optional[int] = None) -> 'ReplayBuffer':
        """从文件加载"""
        data = np.load(path)
        size = len(data['states'])
        if max_size is None:
            max_size = size

        buffer = cls(max_size=max_size)
        load_size = min(size, max_size)
        buffer.states[:load_size] = data['states'][:load_size]
        buffer.policies[:load_size] = data['policies'][:load_size]
        buffer.wdls[:load_size] = data['wdls'][:load_size]
        if 'weights' in data:
            buffer.weights[:load_size] = data['weights'][:load_size]
        buffer.size = load_size
        buffer.position = load_size % max_size

        print(f"ReplayBuffer 已加载: {path} ({load_size} 样本)")
        return buffer
