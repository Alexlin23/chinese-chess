# AlphaZero 中国象棋 — 模块化项目规划

> 基于 AlphaZero 算法从零训练中国象棋 AI。
> **硬约束：本文件夹完全自包含，不修改 `backend/` 和 `frontend/` 的任何代码。**
> 对已有规则引擎仅做**只读导入**，不对其进行任何改动。

---

## 目录

1. [架构设计原则](#1-架构设计原则)
2. [依赖关系图](#2-依赖关系图)
3. [目录结构](#3-目录结构)
4. [模块接口契约](#4-模块接口契约)
5. [数据流设计](#5-数据流设计)
6. [实施路线图](#6-实施路线图)
7. [附录](#7-附录)

---

## 1. 架构设计原则

### 1.1 核心约束

```
┌──────────────────────────────────────────────────┐
│                  项目边界                          │
│                                                   │
│  backend/          frontend/        alphazero/    │
│  ┌─────────┐      ┌─────────┐     ┌───────────┐  │
│  │ 不允许   │      │ 不允许   │     │  ✓ 所有    │  │
│  │ 修改！   │      │ 修改！   │     │ 开发在这里  │  │
│  └─────────┘      └─────────┘     └───────────┘  │
│       │                                 │         │
│       └──── 只读导入 ──────────────────→│         │
│          (chess_rules.py)              │         │
└──────────────────────────────────────────────────┘
```

### 1.2 设计原则

| 原则 | 说明 |
|------|------|
| **接口先行** | 每个模块先定义抽象接口，再写具体实现 |
| **依赖倒置** | 高层模块不依赖低层模块，都依赖抽象接口 |
| **单一职责** | 一个模块只做一件事，但做到极致 |
| **可替换性** | 规则引擎、神经网络、搜索算法均可独立替换 |
| **零侵入** | 不对 `backend/`、`frontend/` 做任何修改 |

### 1.3 规则引擎接入方式

```python
# alphazero 不复制、不修改规则代码，而是只读导入
# 通过 sys.path 将项目根目录加入搜索路径，导入已验证的 chess_rules

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # 项目根目录

from backend.chess_rules import (
    create_initial_board, get_valid_moves, is_valid_move,
    make_move, check_game_result, is_in_check,
    ROWS, COLS
)
```

这等价于把 `backend.chess_rules` 当成**第三方库**来使用——依赖但不修改。

---

## 2. 依赖关系图

```
                    ┌──────────────┐
                    │   config.py  │  ← 全局超参数，被所有模块依赖
                    └──────────────┘

    ┌───────────────────────────────────────────────────┐
    │                alphazero 模块依赖图                  │
    │                                                    │
    │   engine/          network/          search/       │
    │  (游戏引擎)        (神经网络)        (MCTS)         │
    │      │                 │                │          │
    │      │   只读导入       │                │          │
    │      ├─ backend/       │                │          │
    │      │  chess_rules    │                │          │
    │      │                 │                │          │
    │      │                 │                │          │
    │      └─────────────────┼────────────────┘          │
    │                        │                           │
    │                   play/ (自对弈)                     │
    │                   ┌────┴────┐                      │
    │                   │          │                      │
    │              train/       eval/                     │
    │              (训练)       (评估)                     │
    │                   │          │                      │
    │                   └────┬────┘                      │
    │                        │                           │
    │                   scripts/                          │
    │                   (运行脚本)                         │
    └───────────────────────────────────────────────────┘

依赖方向: 上层 → 下层 (无循环依赖)

engine/ ──────────→ backend.chess_rules (只读)
network/ ─────────→ torch (独立，不依赖任何 alphazero 模块)
search/ ──────────→ engine/ + network/
play/ ────────────→ search/ + engine/
train/ ───────────→ network/ + play/replay_buffer
eval/ ────────────→ search/ + engine/
scripts/ ─────────→ train/ + play/ + eval/
```

---

## 3. 目录结构

```
alphazero/
│
├── PLAN.md                          # 本文件
├── README.md                        # 项目说明
│
├── config.py                        # 全局超参数 + 路径配置
│
├── engine/                          # ── 游戏引擎层 ──
│   ├── __init__.py                  # 导出: GameState, Move, ActionEncoder
│   ├── state.py                     # GameState 类：棋盘状态封装
│   ├── move.py                      # Move 数据类 + 走法索引编解码
│   └── constants.py                 # 棋子枚举、方向常量、通道定义
│
├── network/                         # ── 神经网络层 ──
│   ├── __init__.py                  # 导出: create_network, save_checkpoint, load_checkpoint
│   ├── model.py                     # ChineseChessNet (完整网络)
│   ├── blocks.py                    # ConvBlock, ResBlock (可复用构件)
│   └── heads.py                     # PolicyHead, ValueHead
│
├── search/                          # ── MCTS 搜索层 ──
│   ├── __init__.py                  # 导出: MCTS, mcts_search
│   ├── tree.py                      # 搜索树管理 (选择/扩展/回传)
│   ├── node.py                      # MCTSNode 数据结构
│   └── evaluator.py                 # 批量神经网络评估器
│
├── play/                            # ── 自对弈层 ──
│   ├── __init__.py                  # 导出: SelfPlayWorker, ReplayBuffer
│   ├── worker.py                    # 单局对弈生成器
│   ├── manager.py                   # 多进程协调器
│   └── buffer.py                    # 经验回放缓冲区
│
├── train/                           # ── 训练层 ──
│   ├── __init__.py                  # 导出: Trainer, AlphaZeroLoss
│   ├── trainer.py                   # 训练循环 orchestrator
│   ├── loss.py                      # 损失函数 (policy loss + value loss)
│   └── dataset.py                   # torch.utils.data.Dataset
│
├── eval/                            # ── 评估层 ──
│   ├── __init__.py                  # 导出: Arena, EloTracker
│   ├── arena.py                     # 模型对战评估
│   └── elo.py                       # Elo 评分系统
│
├── scripts/                         # ── 入口脚本 ──
│   ├── selfplay.py                  # 启动自对弈
│   ├── train.py                     # 启动训练
│   ├── evaluate.py                  # 启动评估
│   └── pipeline.py                  # 一键运行完整训练管道
│
├── data/                            # ── 数据存储 ──
│   ├── buffers/                     # 自对弈经验数据
│   └── checkpoints/                 # 模型权重文件
│
└── tests/                           # ── 单元测试 ──
    ├── __init__.py
    ├── conftest.py                  # 共享 fixtures
    ├── test_state.py                # GameState 测试
    ├── test_move.py                 # Move 编解码测试
    ├── test_network.py              # 网络 forward/backward 测试
    ├── test_mcts.py                 # MCTS 正确性测试
    ├── test_buffer.py               # ReplayBuffer 测试
    └── test_rules_consistency.py    # 规则一致性测试（对比 chess_rules）
```

---

## 4. 模块接口契约

### 4.1 `engine/` — 游戏引擎层

**职责**: 封装棋盘状态、走法表示、规则查询。是 `backend.chess_rules` 的唯一对接点。

#### `engine/constants.py`

```python
from enum import IntEnum

ROWS = 10
COLS = 9
NUM_CHANNELS = 18       # 神经网络输入通道数
POLICY_SIZE = 2086       # 走法空间维度

class Piece(IntEnum):
    """棋子类型枚举"""
    ROOK = 0      # 車
    KNIGHT = 1    # 馬
    ELEPHANT = 2  # 象/相
    ADVISOR = 3   # 士/仕
    KING = 4      # 將/帥
    CANNON = 5    # 砲/炮
    PAWN = 6      # 卒/兵

class Color(IntEnum):
    RED = 0
    BLACK = 1

# 通道布局:
#  0-6:  当前走棋方棋子位 (ROOK..PAWN)
#  7-13: 对方棋子位
#  14:   己方颜色标记 (全1)
#  15:   步数 / 总步数
#  16:   无吃子步数 (重复局面检测)
#  17:   将军标记
```

#### `engine/move.py`

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)  # 不可变，可哈希
class Move:
    """走法 - 不可变值对象"""
    from_row: int
    from_col: int
    to_row: int
    to_col: int

class ActionEncoder:
    """
    走法 ↔ 索引 双向映射。

    编码方案: 预计算所有合理的 from→to 组合，
    建立双向查找表。无效组合对应 -1。
    """

    @staticmethod
    def encode(move: Move) -> int:
        """走法 → 策略向量索引 (0~2085)"""
        ...

    @staticmethod
    def decode(index: int) -> Move:
        """策略向量索引 → 走法"""
        ...

    @staticmethod
    def legal_mask(state: 'GameState') -> np.ndarray:
        """
        返回 (POLICY_SIZE,) 的布尔掩码，
        标记当前局面下哪些走法合法。
        """
        ...

    @staticmethod
    def build_lookup() -> None:
        """
        预计算映射表。在模块加载时调用一次。
        """
        ...
```

#### `engine/state.py`

```python
from typing import Optional
import numpy as np

class GameState:
    """
    棋盘状态 — 不可变快照。

    内部使用 dict 棋盘（来自 chess_rules），
    对外暴露 NumPy 编码用于神经网络输入。
    """

    # ── 构造 ──

    @classmethod
    def new_game(cls) -> 'GameState':
        """创建初始棋盘"""
        ...

    @classmethod
    def from_dict_board(cls, board: list, turn: str) -> 'GameState':
        """从 dict 棋盘构造（对接 chess_rules）"""
        ...

    # ── 查询 ──

    def legal_moves(self) -> list[Move]:
        """当前局面下所有合法走法"""
        ...

    def is_legal(self, move: Move) -> bool:
        """单步走法合法性校验"""
        ...

    def is_terminal(self) -> bool:
        """是否终局"""
        ...

    def result(self) -> Optional[float]:
        """
        终局结果:
          +1.0 = 红胜
          -1.0 = 黑胜
           0.0 = 和棋
          None = 未结束
        """
        ...

    def is_in_check(self) -> bool:
        """当前走棋方是否被将军"""
        ...

    # ── 状态迁移 ──

    def apply(self, move: Move) -> 'GameState':
        """执行走法，返回新状态（不修改自身）"""
        ...

    # ── 编码 ──

    def encode(self) -> np.ndarray:
        """
        编码为神经网络输入。
        返回 shape (18, 10, 9) 的 float32 数组。
        始终从"当前走棋方"视角编码。
        """
        ...

    def to_dict_board(self) -> tuple[list, str]:
        """导出为 dict 棋盘（用于对接外部 API）"""
        ...
```

#### `engine/__init__.py`

```python
from .constants import ROWS, COLS, NUM_CHANNELS, POLICY_SIZE, Piece, Color
from .move import Move, ActionEncoder
from .state import GameState

__all__ = [
    "GameState", "Move", "ActionEncoder",
    "ROWS", "COLS", "NUM_CHANNELS", "POLICY_SIZE",
    "Piece", "Color",
]
```

---

### 4.2 `network/` — 神经网络层

**职责**: 定义、构建、保存/加载神经网络。纯 PyTorch，不依赖任何 alphazero 其他模块。

#### `network/blocks.py`

```python
import torch.nn as nn

class ConvBlock(nn.Module):
    """Conv2d + BatchNorm + ReLU 组合"""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        ...

class ResBlock(nn.Module):
    """残差块: Conv→BN→ReLU→Conv→BN → +input → ReLU"""
    def __init__(self, channels: int):
        ...
```

#### `network/heads.py`

```python
import torch.nn as nn

class PolicyHead(nn.Module):
    """
    策略头。
    输入:  (B, C, 10, 9)  残差塔输出
    输出:  (B, 2086)       走法 logits
    """
    def __init__(self, in_channels: int, policy_size: int = 2086):
        ...

class ValueHead(nn.Module):
    """
    价值头。
    输入:  (B, C, 10, 9)  残差塔输出
    输出:  (B, 1)          局面评分 [-1, +1]
    """
    def __init__(self, in_channels: int):
        ...
```

#### `network/model.py`

```python
import torch
import torch.nn as nn

class ChineseChessNet(nn.Module):
    """
    AlphaZero 中国象棋网络。

    输入:  (B, 18, 10, 9)  棋盘编码
    输出:  (policy_logits: (B, 2086), value: (B, 1))

    Args:
        num_blocks:  残差块数量 (默认 10)
        num_filters: 卷积通道数 (默认 256)
    """

    def __init__(self, num_blocks: int = 10, num_filters: int = 256):
        ...

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (policy_logits, value)"""
        ...

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        推理模式：返回 (policy_probs, value)。
        policy_probs 已过 softmax。
        """
        ...

    @property
    def device(self) -> torch.device:
        """模型所在设备"""
        ...
```

#### `network/__init__.py`

```python
import torch
from pathlib import Path
from .model import ChineseChessNet

def create_network(num_blocks: int = 10, num_filters: int = 256,
                   device: str = "cpu") -> ChineseChessNet:
    """工厂函数：创建并初始化网络"""
    ...

def save_checkpoint(model: ChineseChessNet, path: Path,
                    iteration: int, optimizer=None) -> None:
    """保存检查点"""
    ...

def load_checkpoint(path: Path, device: str = "cpu") -> tuple[ChineseChessNet, dict]:
    """加载检查点，返回 (model, metadata)"""
    ...

__all__ = ["ChineseChessNet", "create_network", "save_checkpoint", "load_checkpoint"]
```

---

### 4.3 `search/` — MCTS 搜索层

**职责**: 实现蒙特卡洛树搜索。依赖 `GameState`（引擎层）和 `ChineseChessNet`（网络层）。

#### `search/node.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class MCTSNode:
    """MCTS 树节点"""
    prior: float                          # 先验概率 P(s,a)
    visit_count: int = 0                  # N(s,a)
    total_value: float = 0.0              # W(s,a)
    children: dict[int, MCTSNode] = field(default_factory=dict)
    is_expanded: bool = False             # 是否已扩展

    @property
    def q(self) -> float:
        """平均价值 Q = W / N"""
        return self.total_value / max(self.visit_count, 1)
```

#### `search/evaluator.py`

```python
import torch
import numpy as np
from typing import Protocol

class NetworkInterface(Protocol):
    """神经网络接口协议 — evaluator 不依赖具体网络实现"""
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
    @property
    def device(self) -> torch.device: ...

class BatchEvaluator:
    """
    批量神经网络评估器。

    收集多个待评估局面，批量送入 GPU 推理，
    显著提升吞吐量。多 Worker 共享同一个实例。
    """

    def __init__(self, network: NetworkInterface, batch_size: int = 64):
        ...

    def evaluate(self, states: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        批量评估多个局面。
        Args:
            states: list of (18, 10, 9) ndarray
        Returns:
            (policies: (N, 2086), values: (N, 1))
        """
        ...

    def evaluate_single(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        """单局面评估（用于简单场景）"""
        ...
```

#### `search/tree.py`

```python
import numpy as np
from engine import GameState, Move, ActionEncoder, POLICY_SIZE
from search.node import MCTSNode
from search.evaluator import BatchEvaluator

class MCTS:
    """
    蒙特卡洛树搜索。

    典型用法:
        mcts = MCTS(evaluator, c_puct=1.5, num_simulations=800)
        probs = mcts.search(state)
        best_move = ActionEncoder.decode(np.argmax(probs))
    """

    def __init__(self,
                 evaluator: BatchEvaluator,
                 num_simulations: int = 800,
                 c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25):
        ...

    def search(self, root_state: GameState) -> np.ndarray:
        """
        执行完整搜索，返回走法概率分布。

        Returns:
            np.ndarray shape (POLICY_SIZE,)
            概率 ∝ N(s,a)^(1/τ)，τ 为温度参数
        """
        ...

    def select_move(self, root_state: GameState,
                    temperature: float = 1.0) -> tuple[Move, np.ndarray]:
        """
        搜索并采样一步走法。

        Args:
            temperature: τ=0 为贪心选择，τ=1 为按访问次数比例采样
        Returns:
            (selected_move, search_probabilities)
        """
        ...
```

#### `search/__init__.py`

```python
from .node import MCTSNode
from .evaluator import BatchEvaluator, NetworkInterface
from .tree import MCTS

__all__ = ["MCTS", "MCTSNode", "BatchEvaluator", "NetworkInterface"]
```

---

### 4.4 `play/` — 自对弈层

**职责**: 用 MCTS 指导自我对弈，产出训练数据。

#### `play/buffer.py`

```python
from dataclasses import dataclass
import numpy as np
from collections import deque

@dataclass
class TrainingSample:
    """一条训练样本"""
    board: np.ndarray           # (18, 10, 9)
    policy: np.ndarray          # (2086,) MCTS 搜索概率
    result: float               # +1 / -1 / 0

class ReplayBuffer:
    """
    固定容量 FIFO 经验回放池。

    容量到达上限后自动淘汰最旧样本。
    """

    def __init__(self, max_size: int = 500_000):
        ...

    def add_game(self, samples: list[TrainingSample]) -> None:
        """批量写入一局棋的所有样本"""
        ...

    def sample(self, batch_size: int) -> list[TrainingSample]:
        """随机无放回采样"""
        ...

    def __len__(self) -> int:
        ...

    def save(self, path: str) -> None:
        """持久化到磁盘 (HDF5)"""
        ...

    @classmethod
    def load(cls, path: str) -> 'ReplayBuffer':
        """从磁盘加载"""
        ...
```

#### `play/worker.py`

```python
import numpy as np
from engine import GameState, Move, ActionEncoder
from search import MCTS, BatchEvaluator

class SelfPlayWorker:
    """
    单局自对弈生成器。

    负责:
      1. 创建初始棋盘
      2. 循环执行 MCTS 搜索 → 采样走法 → 记录样本
      3. 终局后回溯标注所有样本

    产生样本数 ≈ 对局步数
    每步产生 1 条 (board, mcts_probs, result)
    """

    def __init__(self, mcts: MCTS):
        ...

    def play_one_game(self) -> list[TrainingSample]:
        """
        进行一局完整的自对弈。

        Returns:
            该局所有训练样本。
        """
        ...

    def play_one_game_with_callback(
        self, on_move: callable = None
    ) -> list[TrainingSample]:
        """
        带回调的自对弈（用于实时观察对局过程）。
        on_move(state, move, step_number) -> None
        """
        ...
```

#### `play/manager.py`

```python
import torch
from network import ChineseChessNet
from play.worker import SelfPlayWorker
from play.buffer import ReplayBuffer

class SelfPlayManager:
    """
    多进程自对弈管理器。

    协调多个 Worker 并行对弈，共用同一个神经网络进行 MCTS 搜索。
    使用进程池 + 共享 BatchEvaluator 实现高效并行。
    """

    def __init__(self,
                 network: ChineseChessNet,
                 buffer: ReplayBuffer,
                 num_workers: int = None,     # None = CPU 核心数
                 games_per_worker: int = 10):
        ...

    def generate(self) -> int:
        """
        启动自对弈，返回产生的总样本数。
        阻塞直到所有 Worker 完成。
        """
        ...

    def generate_async(self) -> 'Future[int]':
        """异步启动自对弈（用于管道并行）"""
        ...
```

#### `play/__init__.py`

```python
from .buffer import TrainingSample, ReplayBuffer
from .worker import SelfPlayWorker
from .manager import SelfPlayManager

__all__ = ["TrainingSample", "ReplayBuffer", "SelfPlayWorker", "SelfPlayManager"]
```

---

### 4.5 `train/` — 训练层

**职责**: 神经网络训练循环。从 ReplayBuffer 采样，计算损失，更新权重。

#### `train/dataset.py`

```python
import torch
from torch.utils.data import Dataset
from play.buffer import ReplayBuffer

class SelfPlayDataset(Dataset):
    """
    将 ReplayBuffer 包装为 PyTorch Dataset。
    每次迭代从缓冲区全量加载当前数据。
    """

    def __init__(self, buffer: ReplayBuffer):
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (board, policy, value) — 都是 Tensor"""
        ...
```

#### `train/loss.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaZeroLoss(nn.Module):
    """
    联合损失: L = L_policy + L_value + λ * L_reg

    - L_policy: 交叉熵 (MCTS概率 vs 网络预测)
    - L_value:  MSE    (对局结果 vs 网络预测)
    - L_reg:    L2 正则
    """

    def __init__(self, weight_decay: float = 1e-4):
        ...

    def forward(self,
                policy_logits: torch.Tensor,   # (B, 2086)
                value_pred: torch.Tensor,       # (B, 1)
                policy_target: torch.Tensor,    # (B, 2086)
                value_target: torch.Tensor,     # (B, 1)
                parameters: list[torch.Tensor]  # 用于计算 L2
                ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            (total_loss, {"policy": ..., "value": ..., "l2": ...})
        """
        ...
```

#### `train/trainer.py`

```python
import torch
from torch.utils.data import DataLoader
from network import ChineseChessNet, save_checkpoint

class Trainer:
    """
    训练编排器。

    每次迭代:
      1. 从 ReplayBuffer 构建 Dataset
      2. 分多个 epoch 训练
      3. 返回训练好的模型

    不负责:
      - 自对弈数据生成 (由 SelfPlayManager 负责)
      - 模型评估 (由 Arena 负责)
    """

    def __init__(self,
                 model: ChineseChessNet,
                 optimizer: torch.optim.Optimizer = None,
                 lr_scheduler=None,
                 batch_size: int = 2048,
                 epochs: int = 5):
        ...

    def train(self, buffer: ReplayBuffer) -> dict:
        """
        执行一次训练迭代。

        Args:
            buffer: 包含最新自对弈数据的经验池
        Returns:
            {"policy_loss": ..., "value_loss": ..., "total_loss": ...}
        """
        ...

    def train_step(self, batch) -> dict:
        """单步训练（用于自定义训练循环）"""
        ...
```

#### `train/__init__.py`

```python
from .loss import AlphaZeroLoss
from .dataset import SelfPlayDataset
from .trainer import Trainer

__all__ = ["Trainer", "AlphaZeroLoss", "SelfPlayDataset"]
```

---

### 4.6 `eval/` — 评估层

**职责**: 新老模型对战，判断新模型是否更强。

#### `eval/arena.py`

```python
import torch
from network import ChineseChessNet
from search import MCTS, BatchEvaluator

class Arena:
    """
    模型对战竞技场。

    新模型 vs 旧模型，进行 N 局对战。
    双方各执红/黑各半，消除先手优势。

    晋升条件: 胜率 > 55% (可配置)
    """

    def __init__(self,
                 num_games: int = 400,
                 mcts_simulations: int = 800,
                 promotion_threshold: float = 0.55):
        ...

    def evaluate(self,
                 new_model: ChineseChessNet,
                 old_model: ChineseChessNet) -> ArenaResult:
        """
        进行对战评估。

        Returns:
            ArenaResult(
                new_wins, old_wins, draws,
                new_win_rate, promoted: bool
            )
        """
        ...
```

#### `eval/elo.py`

```python
class EloTracker:
    """
    Elo 评分追踪器。

    记录每代模型的 Elo 评分变化，
    绘制训练过程中的能力增长曲线。
    """

    def __init__(self, initial_elo: float = 1500.0, k_factor: float = 32.0):
        ...

    def update(self, winner: str, loser: str, draw: bool = False) -> None:
        """更新 Elo 分数"""
        ...

    def get_rating(self, model_id: str) -> float:
        ...

    def export_history(self) -> list[dict]:
        """导出评分历史（用于可视化）"""
        ...
```

#### `eval/__init__.py`

```python
from .arena import Arena, ArenaResult
from .elo import EloTracker

__all__ = ["Arena", "ArenaResult", "EloTracker"]
```

---

### 4.7 `config.py` — 全局配置

```python
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# ── 路径 ──
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
BUFFER_DIR = DATA_DIR / "buffers"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
LOG_DIR = DATA_DIR / "logs"

# ── 网络架构 ──
NUM_BLOCKS = 10
NUM_FILTERS = 256
NUM_CHANNELS = 18
POLICY_SIZE = 2086

# ── MCTS 搜索 ──
MCTS_SIMULATIONS = 800
C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25
TEMPERATURE_THRESHOLD = 15       # 前 N 步使用温度采样
TEMPERATURE = 1.0                 # 温度参数 τ

# ── 自对弈 ──
GAMES_PER_ITERATION = 500
NUM_WORKERS = None                # None = CPU 核心数

# ── 训练 ──
BATCH_SIZE = 2048
EPOCHS = 5
LEARNING_RATE = 0.001
LR_DECAY_STEPS = 100_000
LR_DECAY_RATE = 0.1
WEIGHT_DECAY = 1e-4
REPLAY_BUFFER_SIZE = 500_000      # 最多保留样本数

# ── 评估 ──
ARENA_GAMES = 400
PROMOTION_THRESHOLD = 0.55

# ── 硬件 ──
DEVICE = "cuda"                   # "cuda" | "cpu"
USE_AMP = True                    # 混合精度训练

# ── 日志 ──
LOG_INTERVAL = 100                # 每 N 步输出一次日志
CHECKPOINT_INTERVAL = 5           # 每 N 次迭代保存一次
```

---

## 5. 数据流设计

### 5.1 完整训练管道

```
┌─────────────────────────────────────────────────────────────────┐
│                       一次训练迭代 (Iteration)                    │
│                                                                  │
│  Step 1: 自对弈 (SelfPlay)                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ current_model (权重)                                       │   │
│  │      │                                                     │   │
│  │      ▼                                                     │   │
│  │ MCTS(当前模型) × N局                                       │   │
│  │      │                                                     │   │
│  │      ▼                                                     │   │
│  │ TrainingSample[] ──→ ReplayBuffer                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           │                                      │
│  Step 2: 训练 (Train)          │                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ReplayBuffer ──→ DataLoader ──→ Training Loop              │   │
│  │                                    │                        │   │
│  │                              new_model (更新权重)            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           │                                      │
│  Step 3: 评估 (Evaluate)      │                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Arena: new_model vs old_model (400局)                     │   │
│  │      │                                                     │   │
│  │      ▼                                                     │   │
│  │ 胜率 > 55%? ──Yes──→ 保存检查点 + 晋升为当前模型          │   │
│  │      │                                                     │   │
│  │     No                                                     │   │
│  │      │                                                     │   │
│  │      └──→ 丢弃新模型，保持当前模型                          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 数据格式定义

训练数据使用 HDF5 存储，每条记录包含：

```
/game_00001/boards     (S, 18, 10, 9) float32  局面编码
/game_00001/policies   (S, 2086)     float32  MCTS 搜索概率
/game_00001/result      scalar        float32  对局结果 (+1/-1/0)
/game_00001/steps        scalar        int32   对局步数 S
```

---

## 6. 实施路线图

### 阶段 0：基础设施 (预计 5-7 天)

**目标**: 可运行的 GameState + Move + 网络前向推理。

| # | 任务 | 文件 | 验收标准 |
|---|------|------|---------|
| 0.1 | 项目骨架搭建 | `config.py`, 所有 `__init__.py` | import 无错误 |
| 0.2 | 棋子/走法常量 | `engine/constants.py` | 单元测试通过 |
| 0.3 | Move + ActionEncoder | `engine/move.py` | 编解码往返一致性 100% |
| 0.4 | GameState 状态封装 | `engine/state.py` | 与 `chess_rules` 行为一致 |
| 0.5 | 神经网络构件 | `network/blocks.py`, `heads.py` | 前向传播 shape 正确 |
| 0.6 | 完整网络定义 | `network/model.py` | 输入(18,10,9) → (2086,) + (1,) |
| 0.7 | 规则一致性测试 | `tests/test_rules_consistency.py` | 所有走法吻合 `chess_rules` |

**阶段 0 里程碑**: `python -m pytest alphazero/tests/ -v` 全部通过。

### 阶段 1：MCTS 搜索 (预计 5-7 天)

**目标**: MCTS 能正确搜出残局杀棋。

| # | 任务 | 文件 | 验收标准 |
|---|------|------|---------|
| 1.1 | MCTSNode 数据结构 | `search/node.py` | 单元测试 |
| 1.2 | BatchEvaluator | `search/evaluator.py` | 批量推理延迟 < 单次×批量 |
| 1.3 | MCTS 核心搜索 | `search/tree.py` | 树搜索无死循环 |
| 1.4 | 残局测试：一步杀 | `tests/test_mcts.py` | MCTS 100% 找到杀棋 |
| 1.5 | 残局测试：三步杀 | `tests/test_mcts.py` | MCTS > 90% 找到杀棋 |
| 1.6 | 随机网络性能基线 | 基准测试 | 800 模拟/步 > 1 步/秒 |

**阶段 1 里程碑**: MCTS + 随机网络能稳定下完一局（不走非法步）。

### 阶段 2：训练管道 (预计 5-7 天)

**目标**: 完整的自对弈 → 训练 → 评估闭环。

| # | 任务 | 文件 | 验收标准 |
|---|------|------|---------|
| 2.1 | ReplayBuffer | `play/buffer.py` | 写入/采样/持久化测试 |
| 2.2 | SelfPlayWorker | `play/worker.py` | 单局产生有效样本 |
| 2.3 | SelfPlayManager 并行 | `play/manager.py` | 多核利用率 > 80% |
| 2.4 | AlphaZeroLoss | `train/loss.py` | loss 数值合理 |
| 2.5 | SelfPlayDataset | `train/dataset.py` | DataLoader 迭代正常 |
| 2.6 | Trainer 训练循环 | `train/trainer.py` | loss 随训练下降 |
| 2.7 | Arena 对战评估 | `eval/arena.py` | 自动判断胜负 |
| 2.8 | 运行脚本 | `scripts/pipeline.py` | 一键启动全管道 |

**阶段 2 里程碑**: `python scripts/pipeline.py` 跑通完整一次迭代。

### 阶段 3：小规模验证训练 (预计 3-5 天)

**目标**: 确认算法能从随机初始化开始学会基本走法。

| 参数 | 值 |
|------|-----|
| 网络规模 | 5 blocks, 128 filters |
| MCTS 模拟 | 200/步 |
| 自对弈 | 200 局/迭代 |
| 总迭代 | 20-30 轮 |

**验收标准**:
- 模型学会不吃自己的子（> 95% 走法合法）
- 模型学会吃对方无保护棋子
- 训练 loss 收敛
- 模型能击败纯随机走子

### 阶段 4：正式训练 (预计 2-4 周)

**目标**: 训练到业余水平。

| 参数 | 值 |
|------|-----|
| 网络规模 | 10 blocks, 256 filters |
| MCTS 模拟 | 800/步 |
| 自对弈 | 500 局/迭代 |
| 目标迭代 | 100+ 轮 |

**运行方式**: `python scripts/pipeline.py` 持续运行，定期检查 Arena 胜率。

### 阶段 5：优化与增强 (持续)

| 优化方向 | 预期收益 |
|----------|---------|
| MCTS 搜索剪枝 | 减少无效搜索，提速 20% |
| 混合精度训练 (AMP) | 训练提速 2× |
| Dirichlet 噪声调优 | 更好的探索多样性 |
| 循环学习率 | 训练更稳定 |
| 网络蒸馏（大模型→小模型） | 推理提速 |
| TensorBoard 可视化 | 可观测性 |

---

## 7. 附录

### A. 与现有项目的对接方式

alphazero 完全自包含，不需要修改任何现有代码。对接方式：

```python
# alphazero 通过 sys.path 只读导入规则引擎
# 等价于 pip install 了一个叫 chess_rules 的包
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.chess_rules import create_initial_board, get_valid_moves, ...
```

**未来可选集成**（需要修改 backend 时才做，当前不做）：

```python
# 如果将来想在网页上对战 AI，只需在 backend 中新增一个 API 端点，
# 导入 alphazero 的模型即可——不需要改动 alphazero
# 这是"backend 依赖 alphazero"，而不是反过来
```

### B. 棋子编码表

| 棋子 | 中文名 | `Piece` 枚举 | 索引 |
|------|--------|-------------|------|
| 車/车 | Rook | `Piece.ROOK` | 0 |
| 馬/马 | Knight | `Piece.KNIGHT` | 1 |
| 象/相 | Elephant | `Piece.ELEPHANT` | 2 |
| 士/仕 | Advisor | `Piece.ADVISOR` | 3 |
| 將/帥 | King | `Piece.KING` | 4 |
| 砲/炮 | Cannon | `Piece.CANNON` | 5 |
| 卒/兵 | Pawn | `Piece.PAWN` | 6 |

### C. 通道编码详解

```
神经网络输入 shape: (18, 10, 9)

通道  0:  己方車位置      (0/1)
通道  1:  己方馬位置
通道  2:  己方象位置
通道  3:  己方士位置
通道  4:  己方將位置
通道  5:  己方砲位置
通道  6:  己方卒位置
通道  7:  对方車位置      (0/1)
通道  8:  对方馬位置
通道  9:  对方象位置
通道 10:  对方士位置
通道 11:  对方將位置
通道 12:  对方砲位置
通道 13:  对方卒位置
通道 14:  己方颜色        (全 0 或 全 1，用于区分方向)
通道 15:  总步数 / 200    (归一化到 [0, 1])
通道 16:  无吃子步数 / 120 (归一化，重复局面检测)
通道 17:  将军标记        (0 = 未被将, 1 = 被将)
```

### D. 计算资源需求

| 阶段 | GPU | RAM | 磁盘 | 预计时间 |
|------|-----|-----|------|---------|
| 阶段 0-2 (开发验证) | 无要求 | 8GB | 1GB | — |
| 阶段 3 (小规模) | GTX 1060+ | 16GB | 20GB | 3-5 天 |
| 阶段 4 (正式) | RTX 3060+ | 32GB | 200GB | 2-4 周 |
| 阶段 5 (大规模) | RTX 4090 | 64GB | 500GB | 4-8 周 |

---

> **下一步**: 按阶段 0 开始编码实现。首先创建 `engine/constants.py` → `engine/move.py` → `engine/state.py`，完成最底层的基础设施。
