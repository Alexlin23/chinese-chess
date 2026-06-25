# 中国象棋 AlphaZero

基于 AlphaZero 算法的中国象棋 AI，支持全并行自对弈训练和 Web 对战。

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                    训练管道 (pipeline.py)                  │
│                                                          │
│   ┌──────────────┐     ┌──────────────┐     ┌────────┐  │
│   │ InferenceServer│◄───│ 16 Workers   │────►│ Trainer│  │
│   │   (GPU推理)    │    │ (CPU自对弈)   │     │ (GPU)  │  │
│   └──────────────┘     └──────────────┘     └────────┘  │
│          │                                      │        │
│          └──────────► Arena ◄────────────────────┘        │
│                     (新旧模型对战)                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    Web 服务 (FastAPI)                      │
│   /api/move ──► MCTS搜索 ──► best.pt 模型                │
│   WebSocket ──► 实时对战                                  │
└─────────────────────────────────────────────────────────┘
```

## 快速开始

### 1. 环境

```bash
conda activate visaion  # Python 3.10 + PyTorch 2.12 + CUDA 13
```

### 2. 启动 Web 服务

```bash
cd backend
python -c "import uvicorn; from main import app; uvicorn.run(app, host='0.0.0.0', port=8002)"
```

访问 `http://localhost:8002` 开始对战。

### 3. 训练

```bash
# 快速测试（10 sims, ~30秒/2轮）
python -m AlphaZero.scripts.pipeline --config config/quick.yaml

# 正式训练（100 sims, ~75秒/局）
python -m AlphaZero.scripts.pipeline --config config/stage1v2.yaml

# AlphaZero 规模（800 sims, 需要长时间）
python -m AlphaZero.scripts.pipeline --config config/default.yaml
```

## 配置文件

所有训练参数通过 YAML 配置管理，不硬编码：

| 配置文件 | sims | games/轮 | 用途 |
|----------|------|----------|------|
| `config/quick.yaml` | 10 | 16 | 快速验证闭环 |
| `config/stage1v2.yaml` | 100 | 100 | 正式训练 |
| `config/default.yaml` | 800 | 25000 | AlphaZero 规模 |

配置项说明：

```yaml
model:
  blocks: 20           # 残差块数量
  filters: 256         # 卷积通道数

mcts:
  simulations: 100     # 每步 MCTS 模拟次数
  batch_size: 100      # 必须 = simulations（一轮完成）
  c_puct: 1.5          # PUCT 探索系数

self_play:
  games: 100           # 每轮自对弈局数
  max_ply: 300         # 单局最大步数

training:
  batch_size: 512      # 训练 batch 大小
  lr: 0.2              # 学习率

arena:
  games: 20            # 评估局数
  threshold: 0.55      # 晋升胜率阈值

parallel:
  workers: 16          # 自对弈 worker 数（null=CPU核心数）
  inference_batch: 256 # InferenceServer batch 大小
```

## 项目结构

```
chinese-chess/
├── AlphaZero/
│   ├── engine/          # 棋盘引擎
│   │   ├── state.py     # GameState (8100动作空间)
│   │   ├── move.py      # ActionEncoder
│   │   └── repetition.py # 重复局面检测
│   ├── model/           # 神经网络
│   │   └── network.py   # PolicyWDLEncoder (policy+WDL)
│   ├── search/          # MCTS 搜索
│   │   ├── tree.py      # PUCT MCTS
│   │   └── node.py      # 树节点
│   ├── train/           # 训练模块
│   │   ├── config.py    # AlphaZeroConfig (YAML加载)
│   │   ├── inference_server.py  # GPU 批量推理服务
│   │   ├── self_play_worker.py  # 多进程自对弈
│   │   ├── trainer.py   # 训练器
│   │   ├── arena.py     # 新旧模型对战评估
│   │   └── monitor.py   # 实时监控
│   └── scripts/
│       └── pipeline.py  # 训练管道主入口
├── backend/
│   ├── main.py          # FastAPI 后端
│   ├── chess_rules.py   # 象棋规则
│   └── database.py      # SQLite 存储
├── frontend/
│   ├── index.html       # 棋盘界面
│   └── chess.js         # 前端逻辑
└── config/
    ├── quick.yaml       # 快速测试配置
    ├── stage1v2.yaml    # 正式训练配置
    └── default.yaml     # AlphaZero 规模配置
```

## 技术细节

- **动作空间**: 8100 (90×90 起点终点编码)
- **价值头**: WDL (Win/Draw/Loss) 三分类
- **并行架构**: InferenceServer(GPU) + N个SelfPlayWorker(CPU) via Queue
- **批量推理**: BatchRequest/Response 一次发送整个 MCTS batch
- **模型选择**: 前端 AI 使用 best.pt（Arena 晋升的最佳模型）

## 性能

| 配置 | sims | 每局时间 | GPU利用率 |
|------|------|----------|-----------|
| quick | 10 | ~3s | 50% |
| stage1v2 | 100 | ~75s | 70-90% |
| default | 800 | ~10min | 90%+ |

## 许可证

MIT
