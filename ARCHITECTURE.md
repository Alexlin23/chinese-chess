# 中国象棋 AlphaZero 训练系统 — 技术架构图

## 一、系统总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                    training_server.py (FastAPI)                      │
│                    http://localhost:8080                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    training_thread()                         │    │
│  │                                                             │    │
│  │   for it in range(max_iterations):                          │    │
│  │     1. 多进程自我对弈 → ReplayBuffer                        │    │
│  │     2. Trainer.train(replay) → 更新模型权重                 │    │
│  │     3. Arena评估 → 新模型 vs 旧模型                        │    │
│  │     4. 保存 checkpoint                                      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  Phase 1: HEURISTIC (前2轮)        Phase 2: NEURAL (第3轮起)       │
│  ┌─────────────────────┐           ┌─────────────────────┐         │
│  │ HeuristicEvaluator  │           │ NeuralEvaluator     │         │
│  │ (纯CPU, 物质+位置)  │           │ (GPU batch推理)     │         │
│  └─────────────────────┘           └─────────────────────┘         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 二、模块架构

```
AlphaZero/
├── engine/                    # 棋盘引擎
│   ├── constants.py           # 棋子常量定义
│   ├── move.py                # 走法编码 (ActionEncoder)
│   │   ├── POLICY_SIZE = 2550 # 走法空间大小
│   │   ├── encode(Move) → int # 走法→索引
│   │   ├── decode(int) → Move # 索引→走法
│   │   └── legal_indices(state) → list[int]  # 合法走法索引
│   ├── state.py               # 棋盘状态 (GameState)
│   │   ├── board: np.ndarray  # (10, 9) 棋盘矩阵
│   │   ├── turn: bool         # True=红方, False=黑方
│   │   ├── encode() → (18, 10, 9)  # 18通道编码
│   │   ├── apply(Move) → GameState  # 执行走法
│   │   ├── is_terminal() → bool     # 是否终局
│   │   └── result() → float         # 终局结果 (+1红胜, -1黑胜, 0和棋)
│   └── fast_chess.py          # 走法规则 (get_valid_moves, is_in_check)
│
├── model/                     # 神经网络
│   ├── network.py             # AlphaZeroNet (残差网络)
│   │   ├── 输入: (B, 18, 10, 9)   # 18通道棋盘编码
│   │   ├── 20个 ResBlock          # 残差块 (Conv3×3 + BN + ReLU)
│   │   ├── Policy Head            # Conv1×1 → FC → 2550 (走法概率)
│   │   └── Value Head             # Conv1×1 → FC → tanh (局面价值)
│   └── neural_eval.py         # NeuralEvaluator (评估器)
│       ├── evaluate(state) → (policy, value)     # 单局面评估
│       └── evaluate_batch(states) → (policies, values)  # 批量评估
│
├── search/                    # MCTS搜索
│   ├── node.py                # MCTSNode (树节点)
│   │   ├── prior: float       # 先验概率 P(s, a)
│   │   ├── visit_count: int   # 访问次数 N(s, a)
│   │   ├── total_value: float # 累计价值 W(s, a)
│   │   ├── q → float          # 平均价值 Q = W / N
│   │   └── select_child(c_puct) → int  # PUCT选择
│   ├── tree.py                # MCTS (搜索树)
│   │   ├── search(state, temp) → policy  # 完整搜索
│   │   ├── _select_leaf()     # 选择叶节点
│   │   ├── _expand(node, state)  # 扩展节点
│   │   └── _backup(path, value)  # 回传值
│   └── evaluator.py           # 评估器接口
│       ├── Evaluator (Protocol)  # 接口定义
│       ├── RandomEvaluator       # 随机评估器
│       └── HeuristicEvaluator    # 启发式评估器
│           ├── 物质价值 (车9, 马4, 炮4.5, 士2, 象2, 兵1)
│           ├── 位置奖励 (车在开阔线, 马在中心, 炮在对方半场)
│           ├── 将军奖励 (+/-8分)
│           └── 综合评分 clip(total/60, -1, +1)
│
└── train/                     # 训练流程
    ├── config.py              # AlphaZeroConfig (超参数)
    ├── replay.py              # ReplayBuffer (经验回放)
    │   ├── add(state, policy, result)  # 添加样本
    │   ├── sample(batch_size) → (states, policies, results)  # 随机采样
    │   └── save/load(path)    # 持久化
    ├── trainer.py             # Trainer (训练器)
    │   ├── optimizer: Adam    # 优化器
    │   ├── scheduler: StepLR  # 学习率调度
    │   └── train(replay) → stats  # 训练一轮
    └── self_play.py           # SelfPlayGame (自我对弈)
        ├── step()             # 走一步 (MCTS搜索+执行)
        ├── is_terminal()      # 是否终局
        ├── result()           # 终局结果
        └── get_training_data() → [(state, policy, value), ...]
```

## 三、MCTS搜索算法

```
MCTS.search(root_state):
    root = MCTSNode(prior=1.0)
    _expand(root, root_state)           # 初始扩展
    添加 Dirichlet 噪声到根节点先验      # 鼓励探索

    for sim in range(num_simulations):   # 400次模拟
        # 1. 选择: 沿 PUCT 最大路径走到叶节点
        node = root
        state = root_state.copy()
        path = []
        while node.is_expanded() and not state.is_terminal():
            idx = node.select_child(c_puct=1.5)
            # PUCT = Q + 1.5 * P * sqrt(N_parent) / (1 + N_child)
            path.append((node, idx))
            node = node.children[idx]
            state = state.apply(decode(idx))

        # 2. 扩展+评估
        if state.is_terminal():
            value = state.result()  # +1/-1/0
        else:
            policy, value = evaluator.evaluate(state)  # 神经网络推理
            for idx in legal_indices:
                node.children[idx] = MCTSNode(prior=policy[idx])

        # 3. 回传 (视角翻转)
        if state.turn != root_state.turn:
            value = -value
        for node, _ in reversed(path):
            node.visit_count += 1
            node.total_value += value
            value = -value  # 对方视角

    # 4. 返回走法概率 (按访问次数)
    policy = [child.visit_count^(1/τ) for child in root.children]
    return policy / sum(policy)
```

## 四、训练循环

```
训练流程:
    model = AlphaZeroNet(blocks=20, filters=128)  # 7.3M参数
    trainer = Trainer(config, device='cuda')
    replay = ReplayBuffer(max_size=200000)

    for iteration in range(30):
        # Phase 1: HEURISTIC (前2轮)
        if iteration < 2:
            evaluator = HeuristicEvaluator()  # 纯CPU, 物质+位置评估
        # Phase 2: NEURAL (第3轮起)
        else:
            evaluator = NeuralEvaluator(model, device='cuda')  # GPU batch推理

        # 自我对弈 (多进程并行)
        for game in parallel(20 games, 16 workers):
            mcts = MCTS(evaluator, num_simulations=400, batch_size=128)
            while not game.is_terminal():
                game.step()  # MCTS搜索 → 选择走法 → 执行
            replay.add(game.get_training_data())

        # 训练 (GPU)
        stats = trainer.train(replay, epochs=10)
        # Loss = MSE(z, v) - π^T log(p) + L2正则
        # 随机采样 8192 样本, batch=512

        # Arena评估 (第3轮起)
        if iteration >= 2:
            arena_result = arena(model_new, model_old, 8局)
            if arena_result.win_rate < 55%:
                model = model_old  # 回滚
```

## 五、关键问题分析

### 问题1: 终止率0% (所有对局都是和棋)

```
原因:
  - HeuristicEvaluator 棋力太弱，1000步内分不出胜负
  - NeuralEvaluator 从随机初始化开始，搜索找不到杀棋
  - max_game_length=1000 步后强制和棋

AlphaZero原版解决方案:
  - 800次模拟/步 (当前400次)
  - 不限步数 (当前1000步)
  - 多进程并行 (当前16进程)
  - 但核心是: 足够深的搜索能找到杀棋路径
```

### 问题2: Neural阶段比Heuristic慢

```
原因:
  - Heuristic: 纯numpy计算, <1ms/步
  - Neural: 神经网络推理, CPU上约10ms/步 (batch=1)
  - 多进程时每个进程独立推理，无batch优化

解决:
  - 主进程用GPU batch推理 (batch_size=128)
  - 多进程方案需要共享GPU (需要重构)
```

### 问题3: Arena评估全是和棋

```
原因:
  - 新旧模型都很弱，无法分出胜负
  - 8局Arena全是和棋 → win_rate=0% → 模型被回滚

解决:
  - 需要更强的评估器 (Pikafish NNUE)
  - 或者增加Arena局数
  - 或者降低Arena阈值
```

## 六、AlphaZero标准 vs 当前实现

| 组件 | AlphaZero标准 | 当前实现 | 差距 |
|------|-------------|---------|------|
| 模型 | ResNet-20, 256通道 | ResNet-20, 128通道 | 通道数少一半 |
| MCTS模拟 | 800次/步 | 400次/步 | 模拟次数少一半 |
| 步数限制 | 不限 | 1000步 | 限制步数 |
| 并行 | 多进程+GPU batch | 多进程+CPU推理 | GPU未充分利用 |
| 训练数据 | 自我对弈 | 自我对弈 | 一致 |
| 损失函数 | MSE+CrossEntropy+L2 | MSE+CrossEntropy+L2 | 一致 |
| Arena | 新模型vs旧模型 | 新模型vs旧模型 | 一致 |

## 七、当前配置

```python
# training_server.py
config = {
    "num_blocks": 20,           # 残差块数
    "num_filters": 128,         # 卷积通道数
    "num_simulations": 400,     # MCTS模拟次数
    "games_per_iteration": 20,  # 每轮对弈局数
    "epochs_per_iteration": 10, # 每轮训练epoch数
    "batch_size": 512,          # 训练batch大小
    "learning_rate": 0.001,     # 初始学习率
    "replay_buffer_size": 200000,  # 经验回放大小
    "max_game_length": 1000,    # 最大步数
    "max_iterations": 30,       # 总迭代轮数
    "warmup_iterations": 2,     # Heuristic轮数
    "arena_games": 8,           # Arena局数
    "arena_threshold": 0.55,    # Arena胜率阈值
    "num_workers": 16,          # 多进程worker数
}
```
