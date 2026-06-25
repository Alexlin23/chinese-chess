# training_thread() 详细架构图

## 一、整体流程

```
training_thread(config_dict)
│
├── 初始化
│   ├── 解析配置 (max_iter, warmup, arena_games, arena_threshold, num_workers)
│   ├── 创建模型: AlphaZeroNet(blocks=20, filters=128) → GPU
│   ├── 创建ReplayBuffer(max_size=200000)
│   └── 创建Trainer(config, device='cuda')
│
└── 主循环: for it in range(30)
    │
    ├── ─── 阶段判断 ───
    │   it < 2  → phase = "heuristic"
    │   it >= 2 → phase = "neural"
    │
    ├── ─── 步骤1: 多进程自我对弈 ───
    │   │
    │   ├── 保存旧模型 (neural阶段)
    │   │   model_old = copy(model)
    │   │
    │   ├── run_parallel_selfplay(config, model, phase, num_workers=16)
    │   │   │
    │   │   ├── 准备
    │   │   │   ├── config_dict = config.to_dict()
    │   │   │   ├── model_state_dict = model.state_dict() → CPU
    │   │   │   └── result_dir = /tmp/chinese_chess_sp_{timestamp}/
    │   │   │
    │   │   ├── 启动20个worker进程 (最多16个并行)
    │   │   │   │
    │   │   │   └── self_play_worker(worker_id, config_dict, model_state_dict, phase, result_dir)
    │   │   │       │
    │   │   │       ├── 每个进程独立:
    │   │   │       │   ├── 加载模型: AlphaZeroNet() → load_state_dict()
    │   │   │       │   ├── 创建评估器:
    │   │   │       │   │   ├── heuristic → HeuristicEvaluator(seed)
    │   │   │       │   │   └── neural   → NeuralEvaluator(model, device='cpu')
    │   │   │       │   └── 创建MCTS(evaluator, sims=400, batch=128)
    │   │   │       │
    │   │   │       ├── 跑一局自我对弈
    │   │   │       │   game = SelfPlayGame(mcts, config)
    │   │   │       │   while not game.is_terminal():
    │   │   │       │       game.step()
    │   │   │       │           ├── mcts.search(state, temperature) → policy
    │   │   │       │           │   ├── 创建根节点
    │   │   │       │           │   ├── 添加Dirichlet噪声
    │   │   │       │           │   ├── 400次模拟:
    │   │   │       │           │   │   ├── _select_leaf(): PUCT选择路径
    │   │   │       │           │   │   ├── _expand(): 评估器.evaluate_batch() → policy, value
    │   │   │       │           │   │   └── _backup(): 回传值 (视角翻转)
    │   │   │       │           │   └── 返回走法概率 policy (2550维)
    │   │   │       │           ├── 从policy采样走法
    │   │   │       │           └── state.apply(move) → 新状态
    │   │   │       │
    │   │   │       └── 保存结果到文件
    │   │   │           ├── game_{id}.npz  → states, policies, values (numpy数组)
    │   │   │           └── game_{id}.json → result, moves, board, last_move
    │   │   │
    │   │   ├── 等待所有进程完成 (p.join(timeout=600))
    │   │   │
    │   │   └── 从文件收集结果
    │   │       for game_*.json:
    │   │           meta = json.load()
    │   │           meta["training_data"] = np.load(game_*.npz)
    │   │           results.append(meta)
    │   │
    │   └── 处理结果
    │       for r in results:
    │           ├── 统计: red_wins, black_wins, draws
    │           ├── 累计: total_moves
    │           ├── 收集: replay.add(state, policy, value)
    │           └── 更新前端: current_board, last_move
    │
    ├── ─── 步骤2: 训练 (GPU) ───
    │   │
    │   ├── trainer.train(replay, epochs=10)
    │   │   │
    │   │   ├── 随机采样: replay.sample(8192)
    │   │   │   → states (8192, 18, 10, 9)
    │   │   │   → policies (8192, 2550)
    │   │   │   → results (8192,)
    │   │   │
    │   │   ├── DataLoader: batch_size=512, shuffle=True
    │   │   │
    │   │   └── for epoch in range(10):
    │   │       for batch in dataloader:
    │   │           ├── 前向传播
    │   │           │   logits, values = model(states)
    │   │           │
    │   │           ├── 计算损失
    │   │           │   policy_loss = -π^T · log(p)  (CrossEntropy)
    │   │           │   value_loss = (z - v)^2       (MSE)
    │   │           │   loss = policy_loss + value_loss
    │   │           │
    │   │           ├── 反向传播
    │   │           │   optimizer.zero_grad()
    │   │           │   loss.backward()
    │   │           │   clip_grad_norm_(5.0)
    │   │           │   optimizer.step()
    │   │           │
    │   │           └── 统计: epoch_policy_loss, epoch_value_loss
    │   │
    │   ├── 更新模型权重
    │   │   model.load_state_dict(trainer.model.state_dict())
    │   │
    │   └── 更新状态
    │       state["loss"] = avg_loss
    │       state["policy_loss"] = avg_policy_loss
    │       state["value_loss"] = avg_value_loss
    │       state["lr"] = current_lr
    │
    ├── ─── 步骤3: Arena评估 (neural阶段) ───
    │   │
    │   ├── run_parallel_arena(config, model, model_old, num_games=8, num_workers=8)
    │   │   │
    │   │   ├── 准备
    │   │   │   ├── model_new_state = model.state_dict() → CPU
    │   │   │   ├── model_old_state = model_old.state_dict() → CPU
    │   │   │   └── result_dir = /tmp/chinese_chess_arena_{timestamp}/
    │   │   │
    │   │   ├── 启动8个worker进程 (最多8个并行)
    │   │   │   │
    │   │   │   └── arena_worker(game_id, config_dict, model_new_state, model_old_state, result_dir)
    │   │   │       │
    │   │   │       ├── 每个进程独立:
    │   │   │       │   ├── 加载 model_new 和 model_old
    │   │   │       │   ├── 创建评估器:
    │   │   │       │   │   ├── 偶数局: eval_red=new, eval_black=old
    │   │   │       │   │   └── 奇数局: eval_red=old, eval_black=new
    │   │   │       │   └── 创建 mcts_red, mcts_black
    │   │   │       │
    │   │   │       ├── 跑一局对战
    │   │   │       │   game = SelfPlayGame(mcts_red, config)
    │   │   │       │   while not game.is_terminal():
    │   │   │       │       game.mcts = mcts_red if turn else mcts_black
    │   │   │       │       game.step()
    │   │   │       │
    │   │   │       └── 判断结果
    │   │   │           r = game.result()
    │   │   │           if 偶数局: win if r>0.5, loss if r<-0.5
    │   │   │           if 奇数局: win if r<-0.5, loss if r>0.5
    │   │   │           → 保存 arena_{id}.json
    │   │   │
    │   │   └── 统计结果
    │   │       for arena_*.json:
    │   │           if "win": wins++
    │   │           if "loss": losses++
    │   │           if "draw": draws++
    │   │       return {wins, losses, draws, win_rate}
    │   │
    │   ├── 决策
    │   │   if win_rate < 55%:
    │   │       model = model_old  # 回滚
    │   │       trainer.model = model_old
    │   │       state["rolled_back"] = True
    │   │   else:
    │   │       保持新模型
    │   │
    │   └── 更新状态
    │       state["arena"] = {wins, losses, draws, win_rate}
    │
    ├── ─── 步骤4: 保存checkpoint ───
    │   ├── iter{it:03d}.pt → model + optimizer + scheduler + iteration
    │   └── latest.pt → model + iteration
    │
    └── ─── 记录历史 ───
        state["history"].append({
            iteration, phase, loss, policy_loss, value_loss, lr,
            red, black, draw, term_rate, buffer_size, elapsed_sp,
            arena, rolled_back
        })
```

## 二、数据流

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            数据流                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  模型权重 (GPU)                                                         │
│  ┌──────────┐                                                           │
│  │  model   │ ──→ model.state_dict() → CPU                              │
│  │ (7.3M)   │     ↓                                                     │
│  └──────────┘     传给每个worker进程                                     │
│                   ↓                                                     │
│  Worker进程 (CPU×16)                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  model.load_state_dict()                                         │   │
│  │  ↓                                                               │   │
│  │  MCTS搜索 (400次模拟)                                            │   │
│  │  ├── evaluator.evaluate(state) → (policy, value)                 │   │
│  │  │   ├── HeuristicEvaluator: 纯numpy, <1ms                      │   │
│  │  │   └── NeuralEvaluator: model.forward(), ~10ms (CPU)          │   │
│  │  └── 返回走法概率 (2550维)                                       │   │
│  │  ↓                                                               │   │
│  │  game.step() → 走一步                                            │   │
│  │  ↓                                                               │   │
│  │  重复直到终局 (最多1000步)                                       │   │
│  │  ↓                                                               │   │
│  │  保存到文件:                                                     │   │
│  │  ├── game_{id}.npz: states(18,10,9), policies(2550), values      │   │
│  │  └── game_{id}.json: result, moves, board, last_move             │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                   ↓                                                     │
│  主进程收集结果                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  for game_*.json:                                                │   │
│  │      meta = json.load()                                          │   │
│  │      meta["training_data"] = np.load(game_*.npz)                 │   │
│  │      replay.add(state, policy, value)                            │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                   ↓                                                     │
│  ReplayBuffer (最多200000条)                                            │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  states:   (N, 18, 10, 9) float32                                │   │
│  │  policies: (N, 2550) float32                                     │   │
│  │  values:   (N,) float32  (+1红胜, -1黑胜, 0和棋)                │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                   ↓                                                     │
│  Trainer (GPU)                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  replay.sample(8192) → 随机采样                                  │   │
│  │  ↓                                                               │   │
│  │  DataLoader: batch=512, shuffle=True                             │   │
│  │  ↓                                                               │   │
│  │  for epoch in range(10):                                         │   │
│  │      for batch in dataloader:                                    │   │
│  │          logits, values = model(states)                          │   │
│  │          policy_loss = -π^T · log(p)                             │   │
│  │          value_loss = (z - v)^2                                  │   │
│  │          loss = policy_loss + value_loss                         │   │
│  │          loss.backward() → optimizer.step()                      │   │
│  │  ↓                                                               │   │
│  │  model.load_state_dict(trainer.model.state_dict())               │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                   ↓                                                     │
│  Arena评估 (neural阶段)                                                 │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  8局对战 (新模型 vs 旧模型, 交替执红黑)                          │   │
│  │  ├── 偶数局: new执红, old执黑                                    │   │
│  │  └── 奇数局: old执红, new执黑                                    │   │
│  │  ↓                                                               │   │
│  │  统计: wins, losses, draws                                       │   │
│  │  ↓                                                               │   │
│  │  if win_rate < 55%: 回滚到旧模型                                 │   │
│  │  else: 接受新模型                                                │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 三、关键代码路径

```
self_play_worker (每个worker进程):
    model = AlphaZeroNet().load_state_dict(model_state_dict)  # CPU
    evaluator = HeuristicEvaluator() 或 NeuralEvaluator(model, device='cpu')
    mcts = MCTS(evaluator, num_simulations=400, batch_size=128)
    game = SelfPlayGame(mcts, config)
    
    while not game.is_terminal():
        game.step()
            → mcts.search(state, temperature)
                → _select_leaf(): PUCT选择 (Q + 1.5 * P * sqrt(N_parent) / (1 + N_child))
                → _expand(): evaluator.evaluate_batch(states) → policies, values
                → _backup(): 回传值 (每层翻转正负号)
            → 从policy采样走法
            → state.apply(move)
    
    保存: game_{id}.npz + game_{id}.json

trainer.train:
    for epoch in range(10):
        states, policies, results = replay.sample(8192)
        for batch in DataLoader(batch_size=512):
            logits, values = model(states)  # GPU
            loss = CrossEntropy(logits, policies) + MSE(values, results)
            loss.backward()
            optimizer.step()
```

## 四、问题定位

```
问题1: 终止率0%
  位置: self_play_worker → game.step() → mcts.search()
  原因: evaluator.evaluate() 返回的value信号太弱
        HeuristicEvaluator: 物质+位置评估，无法判断杀棋
        NeuralEvaluator: 随机初始化，搜索400次也找不到杀棋
  解决: 需要更强的评估器 (Pikafish) 或更深的搜索

问题2: GPU未充分利用
  位置: self_play_worker → NeuralEvaluator(model, device='cpu')
  原因: 每个worker用CPU推理，GPU只在训练时使用
  解决: 需要共享GPU推理 (需要重构架构)

问题3: Arena回滚
  位置: run_parallel_arena → arena_worker()
  原因: 新旧模型都很弱，8局全是和棋 → win_rate=0% < 55%
  解决: 需要更强的评估器 或 降低阈值 或 增加局数
```
