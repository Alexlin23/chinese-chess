"""经验回放缓冲区

存储自我对弈生成的训练数据：(state_encoding, mcts_policy, game_result)。

设计要点:
  - 定长循环缓冲区：超出容量时淘汰最旧数据
  - 支持保存/加载到磁盘（.npz 格式）
  - 支持批量随机采样
"""
import numpy as np
from pathlib import Path
from typing import Optional
from collections import deque
import random


class ReplayBuffer:
    """定长经验回放缓冲区。

    训练样本格式:
      state:  (18, 10, 9) float32 — GameState 编码
      policy: (POLICY_SIZE,) float32 — MCTS 搜索的访问概率分布
      result: float32 — 终局结果（+1=红胜, -1=黑胜, 0=和棋）
    """

    def __init__(self, max_size: int = 100_000, seed: int = 42):
        """
        Args:
            max_size: 最大样本容量
            seed:     随机种子
        """
        self.max_size = max_size
        self.rng = random.Random(seed)

        # 使用 list + 指针实现循环缓冲区（比 deque 更高效切片）
        self._states = []
        self._policies = []
        self._results = []
        self._ptr = 0  # 写入指针

    # ── 添加数据 ──

    def add_game(self, positions: list[tuple[np.ndarray, np.ndarray]],
                 winner: float) -> None:
        """添加一局完整对弈数据。

        Args:
            positions: [(state_encoding, mcts_policy), ...] 每步的数据
            winner:    终局结果
        """
        for state, policy in positions:
            self.add(state, policy, winner)

    def add(self, state: np.ndarray, policy: np.ndarray,
            result: float) -> None:
        """添加单个训练样本。

        Args:
            state:  (18, 10, 9) float32
            policy: (POLICY_SIZE,) float32
            result: float32
        """
        if len(self._states) < self.max_size:
            self._states.append(state)
            self._policies.append(policy)
            self._results.append(result)
        else:
            # 循环覆盖
            idx = self._ptr % self.max_size
            self._states[idx] = state
            self._policies[idx] = policy
            self._results[idx] = result
        self._ptr = (self._ptr + 1) % self.max_size

    # ── 采样 ──

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """随机采样一个批次。

        Returns:
            (states, policies, results):
              - states:   (batch_size, 18, 10, 9) float32
              - policies: (batch_size, POLICY_SIZE) float32
              - results:  (batch_size,) float32
        """
        n = len(self)
        if n == 0:
            raise ValueError("ReplayBuffer is empty")
        indices = [self.rng.randint(0, n - 1) for _ in range(batch_size)]

        states = np.stack([self._states[i] for i in indices])
        policies = np.stack([self._policies[i] for i in indices])
        results = np.array([self._results[i] for i in indices], dtype=np.float32)

        return states, policies, results

    def sample_all(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回所有数据（用于全量训练）。"""
        if len(self) == 0:
            raise ValueError("ReplayBuffer is empty")
        states = np.stack(self._states)
        policies = np.stack(self._policies)
        results = np.array(self._results, dtype=np.float32)
        return states, policies, results

    # ── 属性 ──

    def __len__(self) -> int:
        return len(self._states)

    def is_full(self) -> bool:
        return len(self._states) >= self.max_size

    def clear(self) -> None:
        """清空缓冲区。"""
        self._states.clear()
        self._policies.clear()
        self._results.clear()
        self._ptr = 0

    # ── 持久化 ──

    def save(self, path: str) -> None:
        """保存缓冲区到磁盘（高效格式）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        states, policies, results = self.sample_all()
        # 每个数组独立保存，避免 npz 逐行索引问题
        np.savez_compressed(path,
                            states=states,
                            policies=policies,
                            results=results,
                            allow_pickle=False)
        print(f"ReplayBuffer 已保存: {path} ({len(self)} 条样本)")

    def load(self, path: str) -> None:
        """从磁盘加载缓冲区。"""
        data = np.load(path, allow_pickle=False)
        results_raw = data['results']
        n_all = len(results_raw)
        n = min(n_all, self.max_size)
        self.clear()

        # 直接引用底层数组（np.load 返回的内存映射在访问 [i] 时可能触发全量加载）
        # 对 np.savez_compressed 保存的文件，data 是 NpzFile，其内部数组已完全加载
        for i in range(n):
            self._states.append(data['states'][i].copy())
            self._policies.append(data['policies'][i].copy())
            self._results.append(float(results_raw[i]))

        print(f"ReplayBuffer 已加载: {path} ({len(self)} 条样本)")

    @classmethod
    def from_file(cls, path: str, max_size: int = 100_000) -> 'ReplayBuffer':
        """从文件创建 ReplayBuffer。"""
        buf = cls(max_size=max_size)
        buf.load(path)
        return buf
