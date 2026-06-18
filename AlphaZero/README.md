# AlphaZero 中国象棋

基于 AlphaZero 算法从零训练中国象棋 AI。完全自包含模块，不修改项目其他代码。

## 架构

```
engine/ ──→ backend.chess_rules (只读导入，不修改)
network/     独立 PyTorch 模块
search/  ──→ engine/ + network/
play/    ──→ search/ + engine/
train/   ──→ network/ + play/
eval/    ──→ search/ + engine/
```

## 快速开始

```bash
# 安装依赖
pip install torch numpy h5py

# 运行测试
python -m pytest AlphaZero/tests/ -v

# 一键启动完整训练管道
python scripts/pipeline.py
```

## 模块说明

| 模块 | 职责 |
|------|------|
| `engine/` | 棋盘状态、走法编解码、规则封装 |
| `network/` | 残差网络 (PyTorch) |
| `search/` | MCTS 蒙特卡洛树搜索 |
| `play/` | 自我对弈生成训练数据 |
| `train/` | 神经网络训练循环 |
| `eval/` | 模型对战评估、Elo 追踪 |
| `scripts/` | 各阶段入口脚本 |

## 配置

所有超参数在 `config.py` 中集中管理。

## 详细规划

见 [PLAN.md](PLAN.md)
