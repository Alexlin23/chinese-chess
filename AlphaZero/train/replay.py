"""经验回放缓冲区 — 存储 (state, policy, wdl) 训练样本"""
import numpy as np
from pathlib import Path
from typing import Optional


class ReplayBuffer:
    """经验回放缓冲区

    存储格式:
      - states: (N, 18, 10, 9) float32
      - policies: (N, 8100) float32
      - wdls: (N, 3) float32  [win, draw, loss]
    """

    def __init__(self, max_size: int = 100_000):
        self.max_size = max_size
        self.states = None
        self.policies = None
        self.wdls = None
        self.size = 0
        self.position = 0

    def add(self, state: np.ndarray, policy: np.ndarray, wdl: np.ndarray):
        """添加一个样本

        Args:
            state: (18, 10, 9) float32
            policy: (8100,) float32
            wdl: (3,) float32 [win, draw, loss]
        """
        if self.states is None:
            # 延迟初始化
            self.states = np.zeros((self.max_size, *state.shape), dtype=np.float32)
            self.policies = np.zeros((self.max_size, *policy.shape), dtype=np.float32)
            self.wdls = np.zeros((self.max_size, *wdl.shape), dtype=np.float32)

        self.states[self.position] = state
        self.policies[self.position] = policy
        self.wdls[self.position] = wdl

        self.position = (self.position + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """随机采样 n 个样本

        Returns:
            states: (n, 18, 10, 9) float32
            policies: (n, 8100) float32
            wdls: (n, 3) float32
        """
        if self.size == 0:
            raise ValueError("ReplayBuffer 为空")

        n = min(n, self.size)
        indices = np.random.choice(self.size, n, replace=False)
        return (
            self.states[indices],
            self.policies[indices],
            self.wdls[indices],
        )

    def __len__(self) -> int:
        return self.size

    def save(self, path: str):
        """保存到文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            states=self.states[:self.size],
            policies=self.policies[:self.size],
            wdls=self.wdls[:self.size],
            size=self.size,
            position=self.position,
        )

    @classmethod
    def from_file(cls, path: str, max_size: Optional[int] = None) -> 'ReplayBuffer':
        """从文件加载"""
        data = np.load(path)
        size = int(data['size'])
        if max_size is None:
            max_size = max(size, int(data.get('max_size', size)))

        buffer = cls(max_size=max_size)
        buffer.states = np.zeros((max_size, *data['states'].shape[1:]), dtype=np.float32)
        buffer.policies = np.zeros((max_size, *data['policies'].shape[1:]), dtype=np.float32)
        buffer.wdls = np.zeros((max_size, *data['wdls'].shape[1:]), dtype=np.float32)

        load_size = min(size, max_size)
        buffer.states[:load_size] = data['states'][:load_size]
        buffer.policies[:load_size] = data['policies'][:load_size]
        buffer.wdls[:load_size] = data['wdls'][:load_size]
        buffer.size = load_size
        buffer.position = load_size % max_size

        return buffer
