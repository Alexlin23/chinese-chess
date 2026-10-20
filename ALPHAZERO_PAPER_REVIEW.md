# 中国象棋 AlphaZero 项目全面审查与重构依据

> 更新：第 1–17 节保留 PR 40ff9cfe 的历史审查；第 18–19 节保留上一阶段记录；Data S1 核对、修正与未决事项见第 20 节，实际持续训练、模型产出与使用见第 21 节。第 20 节对此前等待 Data S1、63 局限速、正则系数、树复用及损失归一化的说明具有优先效力。历史代码链接已固定到该提交，避免与改造后的行号混淆。尚未宣称 100% 复现完成。

审查日期：2026-10-15。审查对象：[Alexlin23/chinese-chess](https://github.com/Alexlin23/chinese-chess) 的 main 基线与 [草稿 PR #1](https://github.com/Alexlin23/chinese-chess/pull/1)。main 基线为 c5552498178ae8c740e24ab65b6f8ef15a970236，PR 头为 40ff9cfe9f8feefc4a49939b87b540344f021899，本地为 review/pr-1。通过 GitHub 连接器重新确认了这两个提交。

**这是一份统一的审查文档，覆盖算法、规则、状态、模型、搜索、数据、训练、恢复、并行服务、评估、网页、数据库、配置和文档。** 内容包含前几轮审查中仍成立的发现，并修正了不准确的判断；后续按 AlphaZero 重构时，以本文件的数学定义、接口契约与验收项为依据。证据脚本和 JSON 是辅助材料，不是另外的审查报告。

## 1. 审查结论与优先级

现在最需要解决的是学习闭环和搜索统计的正确性，随后是规则与完整状态的一致性。4×64 的小网络、8100 动作以及 WDL 可以作为工程成果保留，但不能用这些合理选择掩盖其他缺陷。

本轮新增的关键发现是：**PR 的重复叶节点撤销逻辑仍有错误，虚拟访问数和价值没有完整撤销。** 在 64 次模拟结束后，根访问数为 64，子节点访问数之和却为 67；默认 batch=16 在只有一个合法应手的局面也会触发。此前仅检查重复评估和 Q 符号的测试没有发现这一点。

同时，Arena 回滚确实会在全和等未晋升条件下丢弃本轮网络更新。两轮真实 SGD 训练、受控全和的 Arena 返回值验证了：每轮生成数据的网络相同，最终网络仍等于最初随机网络。这证明条件成立时的控制流程，不代表已证明所有真实训练都会全和或永久失败。Replay 仍会积累数据，不能误写成所有训练状态都完全回到初始状态。

### 1.1 核心问题索引

这里的 P0 指按论文重构前必须解决的基础阻断项；P1 指会导致结果错误、运行失败或关键证据失真的问题；P2 指配置、特定边界、效率、可复现性和部署一致性问题。不能把“与论文结构不同”直接等同于高优先级 bug。

| 编号 | 优先级 | 问题 | 证据与来源 |
|---|---|---|---|
| S01 | P0 | 重复叶节点撤销漏掉叶节点自身，搜索统计被虚拟损失污染 | 实测，PR 新增路径 |
| T01 | P0 | Arena 未晋升就回滚，违背持续更新的 AlphaZero 训练闭环 | 真实训练加受控评估，既有门控在 PR 中保留并收紧 |
| G01/G03/W01 | P0 | 目标规则未固定，搜索、网络、Replay、网页没有统一的完整状态契约 | 规则代码与跨模块实测；部分既有 |
| G02 | P1 | 步数上限最后一手将死被盖成和棋 | 真实局面实测，既有 |
| C01 | P1 | 旧 Adam state_dict 装入 SGD 后，更新时报 momentum 错误 | 实测，旧逻辑被 PR 接入主恢复路径 |
| C02 | P1 | CUDA 恢复把 CPU RNG 张量映射到 GPU | 代码与官方 API 契约确认，未做 CUDA 实测；PR 新增 |
| W02/W03 | P1 | 旧棋盘请求覆盖最新对局；悔棋的删除与盘面更新不在同一事务 | 临时数据库与原函数实测，既有 |
| I03 | P1 | 推理返回 NaN 也会标为成功 | 真实子进程注入非有限模型实测 |
| T02/C03 | P1 | 首轮 actor/learner 权重不同；iter 文件恢复时轮次归零 | 实测，初始化既有、轮次覆盖为 PR 新路径 |
| T03/T04/T05 | P1 | 双重加权、学习率时钟错误、梯度累积尾组缩小 | 数学推导与实测 |
| N02/K01 | P1/P2 | 输入通道配置未传到模型，部分 YAML 参数不起作用 | 实测；直接阻碍历史编码改造 |
| D03/D04 | P2/P1 | 回放采样每批扫描全池，字节预算低估对象内存，坏存档迟延产生 NaN | 微测量与损坏数据探针 |
| E01/E02 | P1/P2 | 确定性重复棋局被当作独立证据，置信区间退化 | 棋谱与统计实测 |
| I04/C04 | P2 | 异常路径清理、局级落盘和跨文件提交仍不完整 | 调用链静态确认，未完成全套故障注入 |
| K02/K03 | P2 | “轻量配置”和默认值相反，README 有不存在的配置与错误搜索说明 | 文件与参数量实测 |

不建议把当前 PR 合并后当作已正确复现的 AlphaZero 基线。PR 中 Q 视角、推理错误协议、经验池压缩等有效改动应保留，修复应集中在已找到的缺口。

## 2. 论文来源、证据标准与必须纠正的判断

### 2.1 实际使用的论文

- [2017 arXiv 初稿](https://arxiv.org/abs/1712.01815)，本地 [19 页 PDF](D:/project/Alphazero/references/alphazero-1712.01815.pdf)。
- [2018 Science 正式研究对应的作者公开稿，含 Methods 与补充图表](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf)，来自 [DeepMind 官方链接](https://deepmind.google/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/)，本地 [32 页 PDF](D:/project/Alphazero/references/alphazero-science-2018.pdf)。

以下页码指 2018 文件页码：第 3 页是价值目标、损失与持续更新；第 17 页是 Search；第 17–18 页是 Representation / Architecture；第 19 页是训练与评估配置。2017 表 S1/S2 用作交叉核对。

32 页文件包含正文、Methods 和图表，**不等于另行取得了独立 Data S1 压缩包中的所有伪代码**。本报告不冒充已读取该压缩包，也不以来源不明的第三方实现替代论文。

### 2.2 证据如何区分

| 类型 | 本报告中的含义 |
|---|---|
| 真实运行 | 执行当前源代码、PyTorch 或真实子进程/临时 SQLite，记录输出 |
| 受控探针 | 对外部环节或输入做明确替换，例如强制 Arena 全和，验证其余真实调用链 |
| 静态确认 | 从具体代码路径和明确 API 契约推出，尚未执行相应设备或故障场景 |
| 性能观察 | 小规模指定场景测量，不能外推为正式训练吞吐量 |
| 设计差异 | 定义与论文不同，需按复现目标修改或作为独立变体验证 |
| 未验收 | 缺少证据，不能说“已验证正常”，也不因缺证据就断言一定失败 |

### 2.3 本轮修正了哪些说法

1. 平铺动作向量不是天然违背 AlphaZero。2018 第 18 页记录作者也试过棋类平铺输出，效果接近而训练略慢；本项目的具体性能仍须自己测。
2. WDL 的胜概率减负概率是有效期望结果估计，但 WDL 交叉熵与论文标量 MSE 不是同一个损失。
3. 两份 Configuration 都描述训练按访问次数比例选点。不能未经进一步来源核实，就宣称论文明确要求“前 30 ply 随机、后面 τ=0”。当前 τ=0.1 应作为明确设计记录。
4. 训练对全动作 softmax、搜索时对合法动作重归一化，是可成立的概率模型。不能仅凭这点判成错误，更不能直接加掩码后留下 0×负无穷的 NaN。
5. 当前紧凑 Replay 能重建已有 18 通道输入。历史观测缺失与压缩错误是不同问题；MCTS 的 GameState 保留重复计数，网络则看不到。
6. 输入颜色通道实际恒为 1，不是真正红黑颜色位；己方将帅所在宫区仍提供隐含颜色线索，不能断言模型完全无法区分红黑。
7. 虚拟损失的符号修正正确，但取消预留仍有漏项。旧报告对此路径“已经修好”的概括已撤回。
8. 现有 AlphaZero/tests 共 10 项，benchmark 测试不需要提前导入 PyTorch；上一轮已纠正漏跑。本轮重新执行仍为 10 passed。
9. 本环境已经安装并运行 CPU PyTorch。驱动能看到 RTX 5070；CPU 版 torch 的 cuda_available=false 不能据此解释为电脑没有 GPU。

## 3. 重构目标：先固定 AlphaZero 学习闭环

### 3.1 数学与接口契约

以当前行动方视角定义网络和价值：

~~~text
(p, v) = f_theta(observation(s))
z(s, player) = +1 胜，0 和，-1 负
L = (z - v)^2 - sum_a pi(a|s) log p(a|s) + c ||theta||^2

Q_parent(action_to_child) = -Q_child
PUCT = Q_parent + C(N) * P(action) * sqrt(N_parent)/(1 + N_child)
C(N) = log((1 + N + c_base)/c_base) + c_init
~~~

网络价值必须说明“当前行动方”还是固定红方，搜索备份每跨一层翻转一次，整局结果再按每个样本行动方转换标签。不能在某一层混用 bool、红黑符号与赢家编号。

应分别定义三个分布：网络原始 p、搜索合法先验 P、由访问次数构造的训练目标 pi。动作采样分布 mu 可以使用温度，但 mu 与 pi 是否相同必须写清，不能因为 select_move 恰好返回一个数组就默认为两个用途永远一致。

完整规则状态应能判断合法走法、应用动作、终局和责任。网络 observation 可以是该状态的特征投影，不必序列化整个引擎对象，但需要足够的历史/规则特征；Replay 必须能重建所选择的 observation 版本。

### 3.2 建议的论文基线与研究变体

| 项目 | 按论文建立的基线 | 可另开实验的变体 |
|---|---|---|
| 模型更新 | 最新 learner 持续更新，actor 使用有版本的最新快照 | best-player 晋升门控 |
| 数据来源 | 双方神经网络加 MCTS 的纯自我对弈 | 启发式对手、one-hot 模仿、人工开局参与训练 |
| 价值目标 | 标量期望结果与 MSE | WDL 分类与期望值转换 |
| 样本权重 | 先用统一权重，明确均匀抽样单位 | 吃子加权、优先级采样及相应校正 |
| 规则 | 已选定、可验证的中国象棋规则 | 有明确定义的简化规则任务 |
| 网络规模 | 先保留可运行的小塔，记录预算差异 | 增大塔、关系特征或候选动作打分 |
| 动作表示 | 可先保留 8100，完整编解码和掩码 | 空间动作平面等新版本 |

“硬件缩小版的论文基线”与“逐项复刻论文架构”是两个明确层级。前者不要求立刻使用 256 通道深塔；若保留 WDL 等替代目标，应在实验名称中说明，不能同时声称其损失与论文完全一致。

### 3.3 应形成的数据流

~~~text
RuleState + History
  -> versioned ObservationEncoder / ActionEncoder
  -> CPU MCTS <-> versioned neural inference
  -> (game_id, ply, observation context, visits/pi, actor_version)
  -> 已确认终局后按行动方添加 z 和终局原因
  -> 有版本且可恢复的 Replay
  -> learner 做 SGD 更新，递增 global_optimizer_step
  -> 发布 latest actor snapshot
  -> 独立评估 / 归档 best，不阻断 learner
~~~

## 4. 逐模块覆盖范围

| 文件或模块 | 核对内容 | 总体判断 |
|---|---|---|
| engine/constants.py | 棋盘、颜色、动作和输入维度、规则常量 | 颜色类型和硬编码需统一 |
| engine/fast_chess.py | 七类走法、马腿象眼、炮架、将军、原地模拟恢复 | 常见固定用例通过，完整重复规则不在此实现 |
| engine/state.py / repetition.py | 快照、终局顺序、计数、观测、共享可变对象 | 需重构状态契约及规则版本 |
| engine/move.py | 8100 编解码、越界、掩码 | 8100 项往返通过，可保留 |
| engine/heuristic.py | 子力评分、平分随机、训练来源 | 辅助实验，不能充当论文基线 |
| model/network.py | ResNet、两个 head、BN、推理概率与价值 | 结构可工作，参数集中在 head |
| model/neural_eval.py | 编码与掩码、设备、批量推理 | 与当前编码一致，需版本与数值校验 |
| search/node.py / tree.py / evaluator.py | Q 方向、PUCT、噪声、预留、备份、预算、返回分布 | 存在新确认的虚拟访问泄漏 |
| train/self_play_worker.py | 数据时间点、双方标签、温度、权重、结束状态、队列 | 标签主视角正确，数据语义与异常处理需补齐 |
| train/timed_worker.py | 计时接口、重复实现、实验模式 | 当前非主循环入口，存在维护漂移风险 |
| train/replay.py | 紧凑格式、FIFO、量化、随机采样、存取、验证 | 压缩方向正确，历史扩展、性能与格式校验待改 |
| train/trainer.py | 损失、优化器、累积、调度、精度、保存与恢复 | 多个具体训练/恢复问题 |
| scripts/pipeline.py | 模型同步、self-play、训练、晋升、恢复、持久化 | 需改为持续学习并定义提交阶段 |
| train/inference_server.py | 进程启动、合批、验证、分发、错误、设备与计时 | 健康路径 CPU 通过；NaN 响应和清理契约不足 |
| train/arena.py / arena_stats.py | 执色、搜索配置、噪声、独立性、晋升统计 | 不应控制论文基线；置信证据失真 |
| train/benchmark.py | random/heuristic、模型加载、得分和区间 | 工具可用，但统计和实验元数据不足 |
| train/config.py / config/*.yaml | 加载字段、真正生效的参数、跨字段约束 | 存在静默忽略和误导配置 |
| train/monitor.py / train/__init__.py | 指标、GPU 监控、延迟导入 | 不足以代替实验日志或性能分解 |
| backend/chess_rules.py | Web 规则与训练规则对照 | 300 个随机可达局面走法一致，不证明完整规则一致 |
| backend/main.py / models.py | 状态转换、AI、走棋、终局、输入、模型版本 | 完整状态丢失，服务端状态校验不足 |
| backend/database.py | 对局、走棋原子写入、步数、悔棋恢复 | 单次 record_step 有事务，但请求版本与悔棋事务不足 |
| frontend/chess.js / index.html / style.css | 请求、历史恢复、轮询、AI 控制、结果展示 | 需棋局版本与在途请求保护；未作视觉/无障碍验收 |
| AlphaZero/tests 与 test/test_api.py | 真实覆盖、缺口、外部副作用 | 算法测试仅 10 项；API 脚本会访问运行中服务 |
| README / ARCHITECTURE / TRAINING_DETAIL / 既有审查 | 描述与当前源码是否一致 | 多处过时或不成立，详见配置与文档章节 |

## 5. 规则、状态和历史审查

### G01 · P0 · 先选择目标规则，不能把简化裁定当作完整中国象棋

位置：[state.py:195](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/engine/state.py#L195)、[repetition.py:35](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/engine/repetition.py#L35)。

当前环境是“三次相同棋盘和轮次即和棋 + 60 个无吃子半回合 + 最大总 ply”。没有长将、长捉、责任判定和相应历史。应在国规、WXF 或某个计算机比赛/平台规则中明确选择，依据例如 [WXF 2018 规则](https://www.wxf-xiangqi.org/images/wxf-rules/2018_World_XiangQi_Rules_English2018.pdf) 建立固定裁定集。

自然限着的计数单位尤其要明确：当前每走一方的一手就加 1，到 60 判和，是 30 个完整回合。若以皮卡鱼默认配置对照，[其规则选项](https://www.pikafish.com/wiki/en/reference/uci-options.html) 的默认是 120 plies，并另有将军计数处理。不能把 60 改成 120 就宣称覆盖全部规则，也不能说所有采用 60-ply 限制的自定义任务都无效。

**影响：** 规则决定 MCTS 的合法性和终局值，也决定整局监督标签。双方都运行同一个错误或简化环境，仍可能在这个环境内训练成功，但不是所声称规则下的中国象棋 AI。

**重构要求：** 保存 rule_id / rule_version；裁定信息包含胜者、原因、责任方和必要的重复证据。特殊规则依赖完整对局轨迹，不能仅依赖最近几张棋盘。训练、Web、评估都调用同一个裁定入口。验收包括双方长将、长捉例外、双方责任、循环被打断、吃子重置、限着边界和规则版本切换。

### G02 · P1 · 最大步数检查覆盖了最后一手将死或困毙

位置：[state.py:211](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/engine/state.py#L211)。

先检查 max_ply，再检查无合法走法。双方将帅仍在、但最后一手已形成将死时，被先判 max_ply 和棋。

**实测：** 合法红车走法 (1,0)->(1,4) 后黑方无合法走法。普通上限返回红胜 no_legal_move；max_ply=1 时返回 max_ply 和棋。代码与完整棋盘已写入复现脚本，main 已有此问题。

**修复与验收：** 规则性胜负优先于人为预算截断；明确重复等其他终局的优先级。覆盖红黑将死、困毙、吃王兼容输入、恰到边界与尚未结束。不能只测“直接吃王”，因为吃王检查原本就排在 max_ply 之前。

### G03 · P0（状态改造）· 完整状态、网络观察与存储尚未形成统一定义

位置：[state.py:274](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/engine/state.py#L274)、[replay.py:88](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/train/replay.py#L88)。

引擎状态包含 repetition_counts，MCTS 的 copy/apply 路径会保留它；18 通道 observation 没有历史或重复特征。Replay 也只保存当前盘面、轮次、计数等字段。

**实测：** 两个非终局状态棋盘、行棋方、总步数和无吃子数相同，仅此前重复次数不同，编码完全相同；执行相同四手马往返后，一个终局和棋，另一个仍可继续。网络对这类历史差异没有可辨识输入。

这不等于 Replay 已经损坏现有 18 通道，也不意味着所有价值标签都错了。终局盘面没有被额外存入 positions；现有样本仍是合法对局的最终结果监督。准确问题是：模型观测不足、后续增加历史时存储格式也须同步变化。

**重构要求：** 划分 RuleState、Observation 和 StoredGame 三个契约。按论文方向加入时间历史与规则特征，但裁定仍使用完整历史。Replay 必须能从 game_id/ply 重建同一 observation，执行前后逐通道比较。历史长度和通道数必须有编码版本。

### G04 · P1/P2 · 红黑类型混用会在边界 API 中转换错误

位置：[constants.py:33](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/engine/constants.py#L33)、[state.py:75](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/engine/state.py#L75)、[state.py:49](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/engine/state.py#L49)、[tree.py:198](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/search/tree.py#L198)。

RED/BLACK 是 1/-1，GameState.turn 是 True/False；构造器对非字符串统一 bool(turn)。因此传入 BLACK=-1 得到 True，实际变成红方。GameResult.to_value 比较的是 1/-1，MCTS 和 result_for_player 却会传 bool；黑方胜且 current_player=False 时返回 -1，而传 BLACK 才返回 +1。

当前正常将死流程中，下一行动方通常就是输家，这让部分错误被结果结构遮住；worker 添加标签时显式转回 RED/BLACK，这一部分是正确的。**不能因此声称现有全部黑方样本都反了。** 引入长将责任判负等终局后，下一行动方可能恰是赢家，类型错配会直接影响搜索值。

**修复与验收：** 统一 Player 类型，在 API 边界转换，非法类型显式拒绝。枚举红胜/黑胜/和棋 × 红/黑当前玩家 × 所有允许输入形式，检查 scalar、WDL 与备份方向。

### G05 · P2 · “不可变快照”目前依赖调用方自律

GameState 构造时直接持有传入 board/dict，copy 继续共享它们；encode 和 legal_mask 又会缓存。正常 apply 会复制棋盘并生成新计数字典，因此主搜索并非必然错。但外部修改共享数组或字典后，会出现盘面变了、缓存没变的状态；并行复用同一棋盘时，规则生成器的原地试走也需要隔离。

**建议：** 明确所有权、公开只读接口和搜索内部可变工作区，不要盲目在每次查询中多做拷贝。增加“修改原输入不能改变已发布快照”与“模拟后盘面逐字节恢复”的测试。

### G06 · 已核对的走法与仍未证明的范围

13 类手工走法场景做红黑旋转对称检查，加一个将帅照面暴露用例，共 **27 项检查通过**，覆盖马腿、象眼、不过河、九宫、炮架、兵卒前进/横移、车阻挡等。扩展了 **300 个随机可达局面**的 Web/训练走法集合对照，差异为 0。

这些结果支持常见走法实现，不能替代外部独立规则引擎、完整残局或重复责任裁定验收。对两个同源实现的一致性必须保持这个证据边界。

## 6. 动作编码、模型结构与概率定义

### N01 · 8100 动作可以保留；不要未经定义就改成“论文规定 7200”

位置：[move.py:35](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/engine/move.py#L35)。

当前 action = from_square*90 + to_square，索引范围 0..8099。全 8100 项 encode(decode(action)) 往返通过；合法动作掩码由规则引擎决定，不可达的起终点组合保留为无效项不影响数学定义。

论文没有提供“中国象棋必须 7200”的结论。7200 可以来自 80×90 的一种动作平面设计，但不是唯一方案。例如按中国象棋位移覆盖，18 个纵向距离方向 + 16 个横向距离方向 + 8 个马步 + 4 个一步斜向 + 4 个两步斜向，共 50 类，形成 4500 个空间动作，也可覆盖七类棋子的位移；这只是候选编码方案，尚未在项目实现和完整验证。

先保留 8100 作为基线。未来 ActionEncoderV2 必须定义平面顺序、坐标朝向、所有合法动作覆盖、互逆关系、旧模型兼容策略，不能仅改输出维度。

### N02 · P1（改造接口）· input_channels 配置被接收却没有真正传入模型

位置：[config.py:96](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/train/config.py#L96)、[trainer.py:39](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/train/trainer.py#L39)、[pipeline.py:234](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/scripts/pipeline.py#L234)、[inference_server.py:119](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/train/inference_server.py#L119)。

YAML 可加载 input_channels，但 Trainer/Pipeline 创建模型时只传 blocks 和 filters；推理预分配与 Replay 解码又固定 18。实测 config.input_channels=131，真实网络仍是 18 输入通道。

**影响：** 无法靠改配置完成历史输入迁移，检查点里的配置还可能与真实结构不符。应由同一版本化模型规格构造编码器、模型、推理缓冲与数据解码；加载时检查实际张量形状和规格。

### N03 · 参数真正集中在策略头，小残差塔不等于整体小模型

按实际实例统计可训练参数：

| 部分 | 默认 4×64 | 占默认模型比例 |
|---|---:|---:|
| 输入层与残差主干 | 306,432 | 约 4.59% |
| Policy head | 5,632,484 | 约 84.33% |
| WDL head | 740,419 | 约 11.09% |
| 总计 | **6,679,335** | 100% |

策略头先 32×10×9 展平，再 512 单元、8100 logits。只继续减少残差块，对总参数与 head 的计算瓶颈帮助有限。test.yaml 的 20×256 模型实测为 **30,040,615** 参数。

2018 架构描述是输入卷积加 19 个残差块、256 通道；本项目 test.yaml 的 20 个残差块也不能不加说明就叫“结构完全一样”。小塔合理，空间卷积头值得实验，但参数减少不自动等于相同预算棋力提升。

### N04 · Colour 恒为 1，固定坐标也未按当前玩家翻转

位置：[state.py:290](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/engine/state.py#L290)。

14 个棋子通道已经区分己方/对方；第 14 通道恒为 1；棋盘空间方向仍固定红下黑上。建议显式编码真实颜色，并在采用当前方规范朝向时，同时映射动作起点、终点及 policy 标签。只翻输入而不翻 policy 是新的严重错配。

历史每一帧应以**当前样本行动方**的视角解释，不能让历史帧各按当时行动方反复交换通道。是否做 180 度翻转及列方向应形成固定规范。颜色线索目前可以从宫区隐含推断，所以这不是“模型必然不能学习”的证明。

### N05 · 全动作训练与合法动作推理：需定义目标，不宜简单判错或简单加 mask

位置：[trainer.py:116](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/train/trainer.py#L116)、[network.py:147](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/model/network.py#L147)。

令 p_all 对全部动作归一化，Z_legal 为合法动作的总概率。推理取 p_legal(a)=p_all(a)/Z_legal。对只支持合法动作且和为 1 的目标 pi：

~~~text
CE(pi, p_all) = CE(pi, p_legal) - log Z_legal
~~~

因此当前训练额外鼓励网络把概率质量放在合法动作上；它与推理的条件分布有清晰关系，不能只因归一化集合不同就说矛盾。

**固定 logits 实测：** 两个合法、一个非法动作，三个 logits 都为 0，目标为第一个合法动作。全动作 CE=1.098612，梯度约 [-2/3,1/3,1/3]；合法集合 CE=0.693147，梯度 [-1/2,1/2,0]。两者是不同但可成立的目标。

若把非法 logits 设为负无穷后仍复用当前完整向量的 pi*log_probs，非法位置会出现 0×负无穷，探针得到 NaN。**不能把“统一 mask”当作一行安全修复。** 重构时明确选定目标，若用合法集合 CE，须正确 gather/mask 损失与梯度，保存足够状态重建合法集合。

### N06 · Value/WDL 选择与数值契约

当前三分类顺序 [Win,Draw,Loss]，value=P(Win)-P(Loss)，范围应在 [-1,1]。在真实 worker 中，终局结果按样本行动方转换 WDL 的主要路径正确。

若本次目标是论文损失基线，增加标量 tanh value 与最终 z 的 MSE；把 WDL 作为独立、可切换且有结构版本的变体。不能把 WDL 判成错误，也不能说它与 MSE 梯度相同。旧 PackedSample 的 argmax 存储方式必须随目标格式明确迁移。

训练仅对前向使用条件 BF16，随后 log_softmax/CE 在 autocast 外使用已有输出 dtype。应明确将损失计算转 FP32并监控误差；现有 finite-loss 检查没有覆盖所有梯度和参数的非有限情况，梯度裁剪也没有启用非有限报错。GPU 数值稳定性仍需真实 CUDA 验收。


## 7. MCTS：模拟数量、价值视角与搜索策略

### S01 · P0：重复叶节点取消预留时泄漏虚拟访问，污染策略标签

**位置：** [tree.py](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/search/tree.py#L172)，特别是 182–208 行。

选择路径时，代码先把父节点加入 path，再进入 child，对 child 加上 virtual_loss。若发现该 child 已在本批 pending_nodes 中，撤销逻辑遍历 path[1:]；此时当前叶节点尚未在 208 行加入 path，因此它自身的虚拟 N/W 没被扣回。普通成功回传的撤销逻辑不能补偿这次已经退出的路径。

**本次实测：**

| 局面与设置 | 完成模拟数/根 N | 根的子节点 N 之和 | 结论 |
|---|---:|---:|---|
| 初始局面，64 simulations、batch=64 | 64 | 67 | 泄漏一次 virtual_loss=3 |
| 只有一个合法应手，64 simulations、默认 batch=16 | 64 | 67 | 默认批大小也能触发 |
| 初始局面，batch=1 或本次 batch=16 | 64 | 64 | 这些局面通过不能覆盖重复预留路径 |

泄漏同时进入总价值，改变后续选择、根访问分布和训练 pi。不是单纯计数展示错误。**撤回旧报告中“重复 pending 路径已经完全修好”的结论。** PR 已修复部分价值视角和重复推理问题，但该路径仍阻断搜索正确性验收。

**重构要求：** 为每次选择显式记录实际预留的节点列表；每笔预留只能释放一次，正常、重复、终局、异常、推理失败都在统一清理路径释放。不要依赖“父节点 path”和“已预留节点集合”刚好一致。批内暂存的树更新与成功模拟计数分开。

**验收：** 包括单一合法应手、长强制线、批量接近模拟预算、最后不足一批、重复叶节点、终局叶节点及推理异常。结束时所有虚拟预留归零；当前根计数约定下 root.N=sum(child.N)=实际完成模拟数。对已经展开的非根节点，另按“首次叶评估也计一次”的约定检查 N=1+sum(child.N)，不要机械套用根等式。检查 W/Q 也恢复正确，而非只修 N。

### S02 · P1：终局 select_move 仍可能返回动作

search 对终局返回全零策略，但 [select_move](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/search/tree.py#L144) 继续读取棋盘上的合法动作；temperature=0 对全零向量 argmax 返回索引 0，即起终点相同的 Move(0,0→0,0)。探针复现了该返回值。非零温度分支还可能从棋盘合法动作中选出终局之后的落子。

worker 通常先检查终局，降低了主链触发概率；公共接口自身仍违约，Web 等调用方不能依赖这个偶然前置保护。

**修复：** finished state 直接返回 None，并使返回类型明确。动作产生、合法性检查和游戏状态推进共用终局约束。验收包含重复和棋、截断和棋、无子可走及王被吃四种终局。

### S03 · P1：prior 下限没有重新归一化

[tree.py](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/search/tree.py#L112) 与展开分支把每个合法先验设为 max(p,0.1/合法动作数)。它改变小概率动作后没有再次归一化。44 个合法动作、单一动作先验为 1 的固定输入，总质量实测 **1.0977272727**。

该额外质量相当于改变有效探索强度；与 Dirichlet 混合后也不再具有声明的概率语义。论文基线直接使用归一化合法先验；如保留平滑，应写出凸混合或重新归一化，并作为配置实验记录。验收所有合法先验非负、有限、和为 1，非法动作为 0。

### S04 · 已确认正确的部分与仍需固定的约定

本次未把整个搜索判为失效。节点 value 以该节点行动方为视角，父节点选择用 -child.q，回传逐层翻转符号，这个方向正确。终局 z 也须使用同一约定。测试已经覆盖部分视角与重复推理行为。

根节点为获得 priors 而执行网络前向、没有把其 value 计入一次根模拟，本身可以成立；关键是明确定义“预算包括什么”，并保证日志与实际完成次数一致。

GameResult.to_value 的参数类型存在另一问题：RED/BLACK 常量是 1/-1，但 GameState.turn 是 bool。False 不等于 -1。主 worker 显式转换的路径正确，搜索内直接传 bool 的调用应统一为 Player 类型。当前普通黑方被将死时可能恰好给对结果，不能因此判接口正确；今后增加长将责任方裁定后尤其危险。

### S05 · 论文搜索设置与本项目预算

固定 c_puct 是可运行的简化；2018 Methods 的探索系数随访问数变化。是否采用动态系数应明确，不应声称当前逐项复现。64 次搜索是资源缩减，不能与论文 800 次搜索的强度预期等同。

根 Dirichlet 噪声用于训练，竞技评估应关闭。alpha 应结合中国象棋典型合法分支数选择；国际象棋的 0.3 不能因名字相同就视作中国象棋的已验证最优值。

2018 正文/Methods 明确训练按根访问数采样、评估选最大访问动作。本次没有取得独立 Data S1 伪代码，**不把“前 30 步温度 1，之后必须温度 0”列为已从该版正文核实的强制条款。** 若采用温度退火，应标明来源、训练行为与标签温度，避免把选动作温度同时无意改变学习目标。

### S06 · 搜索效率和观测

每步新建根树，尚未复用落子后的子树；这是效率改进项，优先级低于修复 S01。以后复用必须匹配完整规则状态和模型版本，跨新权重不能直接沿用旧 priors/value 而不定义策略。

记录实际完成模拟、NN 评估叶数量、重复预留次数、终局命中、批大小分布、根熵和用时。batch_size 无须等于 simulations；二者相等还会扩大重复叶节点压力。任何优化都应在固定预算下比较并保留搜索不变量。

## 8. 自对弈与训练：先恢复连续学习语义

### T01 · P0：晋升失败回滚学习器，不符合 AlphaZero 持续更新

**位置：** [pipeline.py](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/scripts/pipeline.py#L417)。

候选与 best 竞技，未晋升时把 best 同时加载回 actor model 和 trainer.model，并清空 optimizer.state。第一份 best 是随机初始化模型。因此一个更新已经发生的候选可能被直接抛弃，下一轮又由旧 best 生成数据。平局很多时问题更明显：promotion_evidence 在没有决定性对局时拒绝晋升，即使阈值设为 0.5 也如此。

**受控复现：** 使用真实 SGD 连续执行两轮训练，把竞技结果控制为每轮全和棋；两轮生成器相同，候选参数确实更新但均被丢弃，最终生成器仍等于最初模型。该实验验证回滚机制，**不代表实际棋力对局，也不证明所有配置永远不能学习**；replay、迭代号、调度器仍可能继续变化。

这是 AlphaGo Zero 式门控思路与 AlphaZero 持续学习的关键混用。按本次重构目标，学习器每次更新后继续前进，自对弈使用最近发布的模型；竞技只做评估和选取展示模型，不决定是否撤销训练。latest、best_for_serving 与可恢复 learner checkpoint 应有独立语义。

**验收：** 即使评估全和棋、胜率未达阈值，下一轮 actor 仍使用新发布 learner 版本；优化器动量及步数连续，评估故障不会把训练权重退回初始模型。

### T02 · P1：首次自对弈和首次训练使用两个独立随机模型

[pipeline.py](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/scripts/pipeline.py#L234) 先建 model，273 行 Trainer 内又初始化模型；只有 loaded_from 存在时才同步两者。首次自对弈/best 基于模型 A，首次更新从随机模型 B 开始。探针发现 9 个 state_dict 条目不同。后续同步会消除此特定分叉，所以不是“每轮永远各自随机”。

统一由一个初始化入口建立 learner，明确发布 actor 快照；首次训练与首次生成所对应版本必须可追踪。验收 fresh start 和 resume 两条路径的参数、模型结构、版本 ID 都一致。

### T03 · P1：重要性权重被重复使用，改变样本目标分布

[replay.py](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/train/replay.py#L74) 以 weight 为采样概率；[trainer.py](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/train/trainer.py#L119) 再给 policy/value 损失乘相同 weight。在未做逆概率补偿时，相对期望贡献约为 weight²。

默认决定胜负的对局中，吃子前局面的权重为 2+0.3=2.3，相对普通样本 1 的贡献约 **5.29 倍**。这会同时作用于双方样本；terminal_weight 不是只加给真实 terminal state。它不是直接修改 z 的奖励塑形，但确实偏离均匀局面训练。

主分支此前也有双重权重，而旧吃子判断对 NumPy 格子的 is not None 几乎总真；PR 的 !=0 修复使这项偏置真正集中到吃子样本上。因此需分析 PR 与旧代码的组合变化。

论文基线先采用明确的均匀位置采样、无吃子/终局人工加权。以后若做优先回放，单独定义采样分布、重要性修正、归一化和偏置实验，不能把当前权重当成无害加速。

### T04 · P1：学习率衰减按训练轮调用，而配置名暗示优化步

StepLR 在 [trainer.py](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/train/trainer.py#L164) 每次 train 调用结束才 step 一次。默认每轮约 157 个 minibatch×10 epochs=1570 次优化更新，100 轮约 157000 次更新；scheduler 只走 100，而 lr_decay_step=100000 不会发生所暗示的衰减。具体更新数还取决于 batch 和累积配置。

2018 论文的学习率节点以训练更新数计。重构应保存 global_optimizer_step 并按它调度，明确 warmup/衰减节点、实际 batch 和累积步数。from_yaml 没有读取相应衰减字段，见 K01；不能只改 YAML 期待生效。恢复后下一步学习率应与不中断训练一致。

SGD momentum 与 weight_decay 方向合理。PyTorch 的 decay=λ 对应目标中 (λ/2)||θ||² 的梯度，不能因为 loss 日志没显式相加就断言“没有 L2”；应分别报告数据损失和正则化贡献，明确系数约定。

### T05 · P1：梯度累积最后不足一组时更新缩小

[trainer.py](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/train/trainer.py#L131) 每个 microbatch 都除以固定 accumulation_steps，最后不足完整组仍直接更新。三份相同 microbatch、累积 2、学习率设为 0 以隔离梯度时，两次更新的梯度约 -0.9998766 与 -0.4999383，尾组只有前组一半。默认 accumulation=1 不触发。

应按实际组内样本数归一化，覆盖尾组和最后一批较小的情况。不同 microbatch 数量相同但样本量不同也不能只按批数平均。梯度累积不会让 BatchNorm 的统计等价于大物理 batch，这一限制应进入实验记录。

### T06 · 数据生成模式、数量和中断语义

- 每 worker 使用 max(1,games//workers)。20 局/8 workers 实际 16 局；2 局/4 workers 实际 4 局。默认 100/4 恰好整除。应分配商和余数，总和精确为请求值。
- heuristic 对手的 game_idx 在各 worker 内重置，奇数配额时颜色分配偏斜：每 worker 25 局、4 worker 得到网络红方 52/黑方 48。纯同网自对弈不具有相同对手不对称，但外部对手评估应严格配对。
- warmup 使用启发式动作监督，会留下模型参数中的教师信息；把回放分别保存不能使后续模型自动成为“完全从零自学”。strict_baseline 应从未受该监督的初始化启动，并分别报告实验来源。
- 恢复 checkpoint 内 opponent_mode 会覆盖配置意图，应显示有效模式。warmup 上限以重启后的局部迭代差计算，反复恢复可能延长总 warmup，需要持久化阶段进度。
- worker/Arena 在非终局遇到 move=None 后退出，部分路径把它作为 max_ply 和棋。算法失败、推理失败应是废弃/重试的 episode，不能制造和棋标签。显式 stop_event 导致的未完成对局不入库这一处理方向正确。

### T07 · 数据复用率、日志与训练稳定性

固定每轮采样量与多 epochs、放回采样，可能使少量新对局被反复使用。不是“epochs>1 必然错误”，而是目前没有把生成样本数、有效独立局数、优化更新数、平均复用次数与 replay 年龄联系起来。先建立 train-to-data 比率记录和预算，再据表现调节。

pipeline 的 history 主要驻留内存，候选保存早于 Arena，持久文件缺少完整评估与终止原因。最小实验记录应含：规则/编码/模型版本、初始权重来源、配置有效值、随机种子方案、优化步、数据来源和年龄分布、各终局原因、WDL/长度、损失/梯度/熵、实际搜索与推理预算、模型发布 ID。出现大量 max_ply/no_capture 时应定位原因，而非只观察 draw 总数。

## 9. 回放：紧凑存储成立，但新状态格式需重做契约

### D01 · 现有压缩没有“自动丢失当前 18 通道”

PackedSample 保存 90 字节棋盘、行动方、计数和当前编码所需信息，策略稀疏存动作索引和概率。当前编码往返测试通过；不能把缺失历史输入归因于本次压缩独自造成，它首先是整个 Observation 的设计问题。

uint16 足以覆盖 0–8099 动作索引，float16 概率会有量化及极小值下溢，采样时重归一化。应给出误差容忍标准，不能仅因使用 float16 判标签已损坏。

### D02 · 重构数据应保留可重建历史及来源

若输入改成历史帧、长将/长捉相关特征，仅存当前棋盘不够。建议以完整对局或可验证分片保存初始状态、逐手动作、规则状态/足够的重建记录、每手 root visits/pi、最终结果、actor 版本与 episode ID；训练时按 ply 还原历史窗口。

至少有 schema_version、rule_version、observation_version、action_version、value_target_version。选择均匀抽“局面”还是均匀抽“对局再抽步”，分布不同，应明确。FIFO 淘汰不能移除剩余样本引用的历史帧；可按完整 game shard 淘汰或保留必要前缀。不要简单把每帧密集张量复制八遍作为唯一方案。

### D03 · P2：每次采样都扫描整个 replay，内存上限只是估算

sample 把 deque 转为 list，再为所有样本构建 weights，时间/临时空间均 O(N)，每个训练 minibatch 重复。隔离扫描成本的小探针，64 样本 batch：1000 条约 2.736ms，100000 条约 6.294ms（5 次中位数，复用相同样本对象，**不是完整训练吞吐基准**）。随机均匀采样可用索引环形存储；确需加权再考虑维护采样结构。

nbytes 采用 payload+256 的常数估算。例样本报告 522 字节，浅层 Python/NumPy 对象计数约 715 字节，尚未包括全部标量与分配开销。这不等于实测 RSS 上限。保存还要构造拼接数组和压缩缓冲；IPC 队列只限制消息数，单消息大小随对局变化。制定内存预算时覆盖训练批、回放、打包/压缩峰值、队列和模型副本。

### D04 · P1：输入与持久回放校验不足

pack_sample 检查有限值，却允许负策略概率并在选取正项时静默丢弃；探针验证负概率输入可被接受。WDL 用 argmax 压成类别，不能无提示支持软 WDL 目标。

load 使用 allow_pickle=False 是合理选择，但缺少整体 schema 和 offsets、索引范围、WDL 范围、概率非负/和为正等校验。把档案中的全部策略概率置零，文件仍能载入，随后 sample 除零得到 NaN。训练晚些报非有限损失不是有效的档案验收。

载入应先验证整个文件再原子地接纳，错误指出 shard/样本/字段；禁止部分载入后继续训练。验收包括空合法分布、重复动作索引、越界索引、异常 offsets、NaN/Inf/负值、结构版本冲突，以及容量淘汰后仍能重建历史。当前正常往返测试通过仍不足以覆盖这些路径。

## 10. checkpoint 与恢复：区分权重初始化和继续训练

### C01 · P1：旧 Adam 状态可以载入新 SGD，但下一步失败

[trainer.py](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/train/trainer.py#L219) 直接 load_state_dict，未检查优化器类型。用合法的旧 Adam checkpoint 实测：载入阶段未拒绝，SGD.step 抛 KeyError('momentum')。

checkpoint 必须带优化器类型与格式版本。不同优化器只能显式选择“仅加载模型重新开始优化”，或使用经验证的迁移；不能把运行到第一步才失败当作兼容。参数分组、网络结构变化也需要验证。

### C02 · P1：CUDA 恢复会把 CPU RNG 状态映射到 GPU

load_checkpoint 使用 map_location=device，保存的 torch CPU RNG Tensor 因此也会被映射到 CUDA，而 torch.set_rng_state 要求 CPU ByteTensor。此结论由 PyTorch 2.8 官方 [torch.load 的设备映射契约](https://docs.pytorch.org/docs/2.8/generated/torch.load.html)、[set_rng_state 的 CPU 状态要求](https://docs.pytorch.org/docs/2.8/generated/torch.set_rng_state.html) 与代码路径确认；本次 CPU checkpoint 的参数及迭代恢复通过，**未实际运行 CUDA 恢复**。

修复应按字段处理设备，把 CPU RNG 显式转回 CPU，再恢复 CUDA RNG 各设备状态。验收 CPU→CPU、CUDA→同设备及明确定义的跨设备加载；无法实现逐位复现时应说明边界，不能声称所有恢复位级一致。

### C03 · P1：从迭代权重文件恢复时，推断的迭代被 0 覆盖

pipeline 从 iter049 文件名推断 start=50，随后 Trainer.load_checkpoint 在该权重文件不含 current_iteration 时赋 0，再覆盖 start。探针观察到了 iter000_old.pt。candidate 的 iter 文件只保存权重/配置/统计，并非完整训练状态。

文件应显式区分 model export 与 learner checkpoint。导入权重是新实验的 warm start；继续训练必须含优化器、调度器和步数等完整字段。缺失时给出明确错误或显式 warm-start 状态，不能默默回到第 0 轮。latest_training 完整 CPU 恢复通过，不应与此路径混为一谈。

### C04 · P1：模型、回放和迭代阶段没有统一提交协议

replay 在训练前保存，latest_training 在训练/评估后保存，中间失败会得到不同代的数据与学习器。保留较新的 replay、恢复较旧 learner 可以是一种有意的恢复策略，但目前没有 phase、episode ID、消费/提交位置等契约来说明如何避免重复生成、漏算阶段或重复训练。

worker 的 GAME 先进入内存，整轮收集结束后才保存；本轮后续 worker 失败可能丢失已完成但未持久化的局。建议完成一局即写唯一 episode shard，再以原子 manifest 指向已提交 learner、replay 集合与训练步。定义每个中断点的恢复动作并做故障注入，避免要求大型数组文件和模型文件跨文件“假原子”。

### C05 · 随机性、发布与文件寿命

Python、NumPy、torch CPU/CUDA RNG 的保存是改进，但 worker 独立 RNG、请求完成顺序和 episode seeds 尚不完整。每局种子可由 run_seed/episode_id 派生，日志记录 actor 版本；统计可复现与位级可复现分别验收。

Trainer 用临时文件再 replace，能够避免普通读者看到半写入文件；没有文件/目录 fsync 时不能承诺断电持久性。best 和 candidate 仍直接 torch.save，可改统一的原子发布工具。清理旧文件前确认没有 actor、服务请求和恢复 manifest 引用，保留策略也应明确。

### C06 · 副本与恢复开销

父进程 model 和 Trainer.model 并存，自对弈另有推理服务副本，Arena 又加载候选和 best。恢复完整文件两次，父级 ck/state 还可能保留 GPU 优化器张量引用。model_old 每轮另存一次，而当前 Arena 已比较 best，该文件用途需要清理。

这些是静态发现的额外内存/I/O来源，**没有实测默认配置 OOM**。先理清对象所有权、释放临时 checkpoint 引用，按 CPU/GPU/RSS 峰值剖析，再决定是否复用模型实例。


## 11. 推理服务、并行与设备使用

### I01 · 实际执行位置和健康路径

当前 CPU 负责规则、树选择、状态编码和 worker；InferenceServer 在指定 CUDA 可用时将神经网络放 GPU，Trainer 同样可用 GPU；pipeline 内 Arena 可在所选设备运行。网页加载模型使用 CPU，benchmark CLI 默认 CPU。不能说“项目所有部分都在 GPU 上”，也不能因为 CPU 规则成为瓶颈就把树整体搬到 GPU 当作简单修复。

本次真实 CPU 路径验证：两个 worker 请求均成功返回，request_id 为 1、2，与本地 evaluator 的策略最大误差约 5.59e-9；两 worker 各完成一局、每局 4 ply 的极小自对弈共生成 8 样本；两 minibatch 的训练损失有限，CPU checkpoint 恢复参数一致。该集成场景使用很小预算与 batch=1，**不能证明 S01 批量搜索路径正确，也不是棋力或 CUDA 性能验收**。

### I02 · 合批、精度、IPC 与性能计时

[inference_server.py](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/train/inference_server.py#L176) 当前推理未使用 autocast，通常为 FP32；Trainer 的 BF16 设置不会自动作用于这个独立进程。应分别设置、报告并验证推理精度。各 worker 的请求能在中心服务合批，是可保留的架构。

虽然文件注释声称共享内存且 import shared_memory，实际传输的是 Queue 中的 NumPy 数组，仍有 pickle/复制；不要把未实现的优化写成成果。请求中传 8100 位 mask 和 18 通道张量也有固定通信成本；先测占比，再选共享缓冲池等方案，必须有 buffer 所有权及释放协议。

CUDA 前向是异步的，perf_counter 包围 model(...) 的 forward 时间主要记录发射开销，之后 .cpu().numpy() 同步会把部分计算时间记入 post。日志不能据此断定“前向很快、后处理才慢”。用 CUDA events 或受控 profiler 测量 GPU；端到端另记排队、编码、选择、拷贝、推理、分发和 worker 等待。不要为每次生产推理加全设备同步来追求漂亮分段数字。

### I03 · P1：非有限输出被作为成功响应返回

当前服务验证请求 shape、批大小、worker_id，但不验证输入/模型输出的 finite、合法集合、概率归一化和 value 范围。

**真实子进程复现：** 保存结构正确但首层权重注入 NaN 的模型，发送合法状态；收到 response.error=None，而 policies 和 values 都含非有限值。Softmax 不会自行把数值异常变成异常响应。这种输出可能污染访问分布及 Replay，晚到 Trainer 的 finite-loss 检查才退出。

必须在边界验证：合法非终局请求有合法动作，输入有限，输出有限、合法概率非负且归一化、非法为零，WDL 和为 1、value 范围符合容忍。终局请求另有明确约定。失败响应带 worker/request/model 版本，废弃受影响 episode，并使管理进程有可观测的失败结果。模型文件明确指定却不存在时，不应静默初始化随机 8×128 模型继续训练；仅专门的 fresh-start 模式允许随机初始化。

### I04 · P2：启动、退出和异常路径仍需协议化

start 仅说明进程已启动，不说明权重加载、设备分配和预分配张量已经成功；这些操作在请求循环 try 之外。父进程后续能发现退出是改进，但应有 Ready/Error 握手并验证模型版本，避免 worker 先进入等待。

parallel_self_play 在已处理的错误与正常退出路径调用 stop_processes，但外围缺少覆盖全部阶段的 try/finally。部分启动失败、replay.add_packed 抛错或 KeyboardInterrupt，仍可能绕过清理。worker 向结果队列 put 没有统一 timeout/stop 协议，父进程停止消费后可能阻塞；terminate 后缺少完整 join 与队列收尾。上述是静态确认，尚未逐项做真实故障注入，不能写成每次都会泄漏。

统一管理上下文：启动→Ready→有界请求→停止接收→取消/排空→join→必要时 terminate 后再 join→关闭队列。按 worker/request/actor_version 匹配响应。单个 MCTS 请求若超过 inference_batch_size 当前会拒绝，不会自动拆分；启动前验证两个批配置的关系，或实现有协议的分片。

入口 __main__ 设置 spawn 不足以保障其他导入调用方式。Linux/WSL 程序化调用可能使用 fork，涉及 CUDA 初始化继承及 RNG 重复。应在库内部显式创建同一 multiprocessing context 的 Process/Queue/Event，并在 spawn 下验收。不能混用不同 context 创建的同步对象。

### I05 · 进度和重复实现

主进度 games_done 主要等待 worker 的 DONE 才更新，即使已接收多个 GAME 仍可能显示 0，不能用它判任务没有进展。应按成功提交 episode 累加，同时区分已开始、已完成、已持久化、已废弃。

timed_worker 另写一套对局循环，当前主 pipeline 不走它；规则、模式、计数等易漂移。提取单一 episode runner，计时层只观察事件。monitor 的 GPU/CPU/RSS 观察用于诊断，不构成训练正确性或棋力证据。

## 12. 竞技和基准：不能用重复棋局证明强度

### E01 · P1：确定性初始局面的重复对局不是独立样本

Arena 每局从同一初始局面开始，关闭噪声、temperature=0，搜索 tie-break 也确定。相同模型/颜色组合会重复相同轨迹。实测四局相同模型只产生 **1 条独特轨迹**；不同候选一般至多产生对应两种颜色分配的固定轨迹，而不是自动获得 400 个独立测试局面。

Wilson 区间本身不能修复样本依赖。只在决定性对局中估计胜率，也不同于整个 WDL 得分的区间；即使保留这两种统计，也必须区分其问题与假设。更根本的 T01 修复是评估结果不再回滚 learner。

评估应使用固定、留出的多样初始状态/开局集合，候选与基准交换颜色、配对比较；报告独立起点数、独特轨迹数、配对数、原始 WDL、各原因和搜索预算。可从独立自对弈生成测试起点，**不要求把人工棋谱引入训练**。相同起点的两盘属于一组，区间估计应考虑配对/聚类，不能直接把重复起点全部当独立样本。

### E02 · P2：benchmark 的正态区间在边界退化

[benchmark.py](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/train/benchmark.py#L72) 使用样本标准差的正态近似，2 局全和实测给出 [0.5,0.5]，全胜同理退化为 [1,1]。代码注释称描述性区间，但输出字段 score_ci95 容易被误解为可靠覆盖保证。

应保留原始 WDL 与配对数据，使用明确的小样本/多项结果或成组重采样方案，并报告假设。两局全和只能说明观察到了两局全和，不能说明真实期望得分已精确等于 0.5。

random 与只看吃子/局部启发式对手只能作为低门槛和回归参考，赢它们不等于接近 AlphaZero 或强象棋引擎。需要固定版本的更强外部对手和分级预算，但本次没有运行这种强度对局，不能给现有模型下棋力结论。

### E03 · 评估配置与发布版本必须可追踪

Arena 只传入部分搜索配置，MCTS 使用自己的默认 batch=32 等值；self-play 默认 batch=16。benchmark 也未完整恢复 checkpoint 内的探索等设置。候选和 best 在同一 Arena 中使用相同配置，并非因此彼此天然不公平，但跨训练/评估与跨实验的结果不可直接混比。

记录模型 SHA/actor ID、完整有效搜索参数、规则版本、终止上限、设备、精度、实际模拟和耗时。固定模拟预算与固定时间预算回答不同问题，分别报告。warmup 的网络原始 policy 对启发式评估也不能冒充神经网络加 MCTS 的 Arena 成绩。

和棋半分作为期望结果评分成立，v=2*score-1；大量截断和棋是否掩盖差异，要结合终局原因。成绩与元数据持久化为可检索记录，不能只打印终端。

## 13. 网页、API 与数据库：现有演示不能代表同一规则智能体

### W01 · P0：网页棋盘被转成全新的规则状态

[backend/main.py](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/backend/main.py#L319) 转换只有 board 和 turn，重建时 move_count、no_capture_count 和 repetition_counts 都归零。

探针走四步马往返后，真实状态有 move_count=4、no_capture_count=4、4 个重复记录键，网页重建结果均为 0。因而相同盘面在训练搜索中可能接近重复/无吃子和棋，网页却当新局，网络计数特征也不同。

后端应从权威游戏记录还原完整 RuleState，AI 与合法走法/终局接口共用此状态；game_id 之外还需要 revision。若允许纯棋盘分析，接口必须明确为无历史分析，不能把它当作可继续的完整比赛状态。

### W02 · P1：服务信任客户端旧棋盘，可以覆盖已经完成的落子

[api_move](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/backend/main.py#L130) 即使传 game_id，仍以 req.board/turn 判断和更新，未校验数据库当前棋盘、行动方、status 或 revision。

**原函数+临时真实 SQLite 探针：** 从初始棋盘走红兵 (6,0)→(5,0) 成功；再用同一旧初始棋盘、同一 game_id 和红方，提交另一红兵 (6,2)→(5,2)，仍成功。数据库棋谱出现连续两个红方，最终棋盘还原了第一步移动的兵。不是必须恶意操作，网络延迟/重复点击也会制造旧请求。

取 step_count 与写 record_step 分开，同样存在并发竞争。修复应提交 game_id、expected_revision 和 move；服务在事务内读取/比较版本、检查 ongoing、行动方与合法性，再写动作、完整状态和新版本。旧版本拒绝并让前端刷新，重复请求用幂等键处理。棋谱加必要唯一约束，而非仅靠客户端禁用按钮。

### W03 · P1：悔棋分两次提交，失败后棋谱与盘面不一致

[record_step](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/backend/database.py#L118) 的棋盘与落子同一事务是可保留的改进；[undo_game](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/backend/database.py#L163) 却先调用已提交的 delete_last_move，再重建和 update_game。

对 update_game 注入失败，真实临时数据库中棋谱从 2 步变为 1 步，盘面仍为两步后的原盘面。悔棋需要一个事务覆盖读取、版本判断、删棋谱、还原完整规则状态和更新版本；失败全部回滚。新局、正常落子、悔棋都需同一 revision 契约。

### W04 · P2：前端异步响应没有绑定局面版本

[frontend/chess.js](D:/project/Alphazero/frontend/chess.js) 的合法走法请求没有选择/棋盘 revision 标识，旧响应可能覆盖新选中棋子的 validMoves。doMove 没有统一 pending 防重保护；AI 按钮的禁用没有覆盖手动移动/悔棋与迟到 AI 响应的冲突。

请求发出时捕获 game_id、revision、当前选择；回复只在仍匹配时应用。新局、悔棋、取消 AI 时失效旧请求，使用 AbortController 和响应版本检查。服务端 W02 仍是最终保证，前端状态不能代替事务校验。

以下另为静态确认：game_over 展示逻辑把非 red_win 的结果当黑胜，增加 draw 返回后会显示错误；AI 对弈终局后不再安排下一次 aiStep，清理按钮状态的 stopAiVsAi 未必执行，可能停留在运行态。应按枚举处理 red_win/black_win/draw/aborted，终局和异常走统一结束处理。本次没有浏览器交互实测，不能写成 UI 测试已通过。

### W05 · P2：输入边界、模型缓存和探索语义

Position 未限制坐标范围，board 未限制 10×9，turn 也不是枚举。api_move 在完整边界校验前按坐标索引棋盘，越界可抛异常，负索引可能先读取错误位置。DTO 和引擎边界都应验证，坏请求返回明确定义的 4xx；不把服务端异常当作普通非法落子。

_get_neural_mcts 首次加载后一直缓存；磁盘 best/latest 更新不代表网页已经使用新网络。选择显式固定服务模型或实现安全的原子换代，并在页面/响应报告当前模型版本。网页 select_move 用 temperature=0，但没有相应关闭默认根噪声；temperature=0 不等于评估确定性。作为展示/评估 AI 应明确噪声开关。

WebSocket 相关实现与前端轮询、文档中的实时同步说法应一并对齐。此次重点确认了状态一致性和具体事务错误，没有做认证、互联网部署和全面渗透测试；不以此报告代替那类审计。

## 14. 配置、说明文档与实验可复现性

### K01 · P1/P2：配置没有形成可验证的单一事实来源

[config.py](https://github.com/Alexlin23/chinese-chess/blob/40ff9cfe9f8feefc4a49939b87b540344f021899/AlphaZero/train/config.py#L87) 按字段手写解析，lr_decay_step/lr_decay_rate、repetition_limit 等没有读取；有些声明字段如 save_interval 并未控制所声称行为。policy_size、wdl_size 和 input_channels 与多处硬编码常量也未形成一致传递。input_channels 改为历史编码值却仍构建 18 输入网络，是直接阻碍重构的问题。

未知键可被静默忽略，零/负预算、空 YAML、错误数值类型、跨字段限制缺少统一校验。应在启动前解析为严格 schema，拒绝未知键、验证范围和关系，保存“用户配置→补全默认值→最终有效值”；所有模型、编码、推理预分配、Replay 和评估从同一版本规格构建。

关键验收：改动每个公开字段，能证明影响了正确对象；无法支持的字段直接报错。结构变更时旧 checkpoint/replay 版本不兼容必须在加载前发现，不能到卷积或矩阵乘时报形状错误。

### K02 · P2：test 配置比默认重，且共用输出目录

默认配置是 4 residual blocks×64 filters，约 668 万参数；test 配置 20×256，约 3004 万参数，训练 batch=512（默认 64）。称其“轻量”与实际内容相反，不能用来做低资源快速验证；本次未运行它验证显存上限。

两者共用 checkpoint 路径时，自动恢复 latest_training 可能加载另一结构。每个 run 独立目录，包含不可变结构规格与指纹；测试配置先小规模覆盖完整闭环，再单独做目标规模负载测试，不能让快速验证依赖旧实验目录。

### K03 · P2：README 与真实入口、参数和成果不一致

根 README 提到仓库没有的 quick.yaml、stage1_v2.yaml，表中 800 simulations、25000 games 等不是当前 64/100 默认；“MCTS batch 必须等于 simulations”也不正确。AlphaZero/README 部分目录名称与现有 model/train 等结构不符，某些从根目录执行的 scripts 路径也不存在。

训练说明中“iter 模型包含优化器”等描述与 candidate 保存内容不一致；硬件耗时/强度若无相应日志不能作为当前成果。旧 CODE_REVIEW/PLAN/RESEARCH_STATUS 可保留为历史，但应指向本报告，标明旧结论时间和已撤回项，防止以后 AI 根据过时描述继续修改。

本轮仅更新这一份主报告，没有把历史说明批量改写成另一组审查文档。代码重构时同步更新入口、参数、有效配置示例、恢复语义、规则版本、已完成与未完成验收。

## 15. 已运行的验证、实际结果与边界

### 15.1 运行环境

WSL Ubuntu 22.04，Python 3.10.12，独立审查环境 /tmp/alphazero-review-env，PyTorch 2.8.0+cpu。机器驱动报告 RTX 5070 12GB，但该审查环境安装的是 CPU wheel，未运行 CUDA/BF16、CUDA 恢复和正式 GPU 性能测试。

原有代码检查和探针都在临时目录生成测试 checkpoint/数据库，未改训练业务代码，未用真实网页棋局数据库做故障注入。HTTP 入口的特定探针通过 AST 提取原函数并提供轻量 DTO，再调用真实 SQLite 操作，因此准确名称是“API 函数/数据库集成探针”，不是已启动 FastAPI 并通过 HTTP 验收。

### 15.2 验证矩阵

| 项目 | 实际结果 | 能说明什么/不能说明什么 |
|---|---|---|
| AlphaZero/tests | 10 passed | 现有用例通过；未覆盖本次全部新缺陷 |
| Python 编译检查 | AlphaZero、backend、审查脚本通过 | 语法可解析，不代表运行正确 |
| 动作编码 | 8100 项往返一致 | 当前索引编解码成立 |
| 固定走法 | 七类走法及相关阻挡/将军共 27 项通过 | 常见用例，非完整规则证明 |
| 同仓库规则差分 | 300 个可达局面，0 差异 | 两份实现一致，非独立裁判 |
| 步数边界将死 | 被判 max_ply 和棋，对照为红胜 | G02 实际错误 |
| 历史观测 | 相同输入对应不同重复终局 | G03 观测混叠 |
| MCTS 计数 | 64 vs 67，含默认 batch 强制应手局面 | S01 实际错误 |
| 终局选步 | 返回 Move(0,0→0,0) | S02 实际错误 |
| 先验平滑 | 和为 1.097727… | S03 概率语义错误 |
| 首轮双初始化 | 9 个张量条目不同 | T02 实际分叉 |
| 两轮回滚 | 真实梯度更新被受控全和评估丢弃 | T01 条件机制成立，非棋力测量 |
| 累积尾组 | 梯度比约 0.5 | T05 实际错误 |
| Adam→SGD | 更新时 KeyError('momentum') | C01 实际恢复失败 |
| iter 权重恢复 | 轮次归 0，创建 iter000_old | C03 实际错误 |
| CPU 推理服务 | 两请求成功，策略误差约 5.59e-9 | 健康 IPC 与 evaluator 一致 |
| 极小自对弈/训练/恢复 | 8 样本、2 批有限损失，参数恢复一致 | CPU 冒烟闭环；非学习收益证明 |
| 非有限模型推理 | error=None，策略/value 非有限 | I03 实际错误 |
| Arena 重复 | 4 局仅 1 条独特轨迹 | E01 实际退化 |
| 全和区间 | 2 局区间 [0.5,0.5] | E02 统计边界 |
| 全动作/合法动作 CE | ln3 与 ln2，梯度不同 | 两目标不同且都有清晰解释 |
| 直接掩码乘零目标 | NaN | 不可机械加负无穷 mask |
| 正常 Replay 往返 | 当前 18 通道重建通过 | 当前格式可用，历史输入需升级 |
| 负概率/坏 Replay | 前者被接受；后者载入后产生 NaN | D04 校验缺口 |
| 内存估算/采样 | 522 vs 浅层 715 字节；O(N) 微测量 | 不是完整 RSS/训练吞吐基准 |
| 网页重建状态 | 4 步历史变为零计数 | W01 实际信息丢失 |
| 旧棋盘提交 | 连续两红方，首步盘面被覆盖 | W02 原函数与数据库复现 |
| 悔棋中途失败 | 棋谱已删、盘面未更新 | W03 事务缺口 |

探针会故意构造错误输入、错误模型和受控竞技输出。“脚本完成运行”表示已收集证据，**不表示这些错误路径通过验收**。

辅助材料：[可复现探针](D:/project/Alphazero/references/review_probes.py)、[本次原始结果](D:/project/Alphazero/references/review_evidence.json)。运行方式：

~~~bash
cd /mnt/d/project/Alphazero
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/alphazero-review-env/bin/python -u references/review_probes.py
/tmp/alphazero-review-env/bin/python -m pytest AlphaZero/tests -q
~~~

不建议直接在仓库根目录执行无范围 pytest：test/test_api.py 在导入时会访问 localhost:8001，属于依赖运行中服务的脚本，可能操作真实服务。本次没有执行这条路径。

### 15.3 还不能作出的结论

尚未证明：完整 WXF/其他指定象棋规则符合性；长将长捉全部责任裁定；相对独立引擎的全量差分；CUDA/BF16 数值和峰值显存；所有进程故障的清理与恢复；浏览器异步操作验收；长时持续学习确有提高；相对强象棋引擎的棋力。本报告给出需要验证的接口和用例，不把“需要验证”包装成已完成成果。


## 16. 按 AlphaZero 重构的实施顺序和验收门槛

本节是修改顺序，不表示本次已完成这些代码改造。每一阶段要保留对应证据，后续性能和棋力比较使用通过前序门槛的版本。

### 阶段 A：冻结实验定义与最小规格

**涉及：** config、constants、state/repetition、checkpoint/replay 元数据。

1. 选定象棋规则版本：是否完整 WXF、其他竞赛规则或有明确差异的简化版；把重复、长将长捉、无吃子、将死、人工截断的优先级写为可执行规格。
2. 定义 Player、Outcome、TerminationReason，消除 bool 与 ±1 混用。区分规则终局、资源截断、取消和算法失败。
3. 冻结 ObservationSpec、ActionSpec、TargetSpec 与 schema 版本；先保留 8100 是可接受的降低迁移成本方案。
4. 建立 run_id、独立输出目录、有效配置快照、结构指纹、种子/episode ID 方案。旧权重只能以显式兼容规则导入。
5. 选择论文损失基线：标量 value/MSE；若初期保留 WDL，标为过渡/独立变体，不能省略其不同点。

**门槛：** 任一状态可说明规则终局与原因；输入/动作/标签版本可从文件读取；不兼容文件在开始训练前报错。不得一边调整无吃子规则，一边混用旧 replay 并比较胜率。

### 阶段 B：统一规则状态、搜索与状态观测

**涉及：** engine、search、model/neural_eval、backend/chess_rules 适配层。

建议公开的概念接口：

~~~text
RuleState:
  current_player
  legal_actions()
  apply(action) -> RuleState
  outcome() -> Optional[Outcome]
  repetition/history/rule counters

ObservationEncoder.encode(state, perspective) -> tensor
ActionEncoder.encode/decode(action, perspective)
Search.run(state, actor_version, budget) ->
  visits, target_pi, selected_action, diagnostics
~~~

RuleState 负责完整判罚；Observation 是投影；Search 不独自另写一套终局；网页规则函数尽量成为同一引擎的适配层。使用只读快照或明确所有权，不能依赖外部永远不修改数组的约定。

先修 S01 的预留释放和 S02/S03，再实现历史特征及相关动作变换。训练目标 pi、实际采样 mu 独立命名。根缓存/子树复用放在正确性验证之后。

**门槛：**

- 七类基础走法、马腿象眼、炮架、将帅照面、自陷将军、应将、困毙、双方胜负、边界将死/和棋全部有固定局面。
- 根据所选规则覆盖长将/长捉责任双方、混合重复、计数边界与历史分叉；加入独立参照，而不仅两个同源实现互比。
- 动作编解码全范围往返；若采用视角翻转，棋盘、全部历史帧、动作、pi 与标签能对应。
- 多种 batch/预算和故障下访问计数与价值备份满足不变量，无遗留预留；终局不返回动作。
- 同一完整状态经序列化、Replay、网页恢复后，合法动作、终局、编码和搜索输入一致。

### 阶段 C：重建连续 learner 与明确的数据分布

**涉及：** pipeline、self_play_worker、trainer、arena、replay。

只保留一个 authoritative learner；actor 读取明确发布的不可变快照。每轮或规定优化步发布版本，自对弈记录版本。小规模先用同步发布，确认正确后再引入异步陈旧度上限，避免一次改入过多并发变量。

移除竞技对 learner 的回滚，把 latest actor、learner recovery、best_for_serving 分离。关闭启发式监督和人工吃子/终局权重，按所选单位均匀采样；确需 warmup 的实验从单独 run 开始，不能污染 strict-baseline 结论。

SGD momentum、L2、损失与梯度累积使用明确数学定义；学习率按 global_optimizer_step，记录数据复用率和梯度统计。修复新启动同步、数量分配、尾组与失败局标签。

**门槛：**

- 从零启动，首轮生成器与 learner 同源；一次优化确实改变参数。
- 连续至少若干发布后，actor 版本单调前进；全和/失败评估不丢弃学习更新。
- 固定小批的 value 视角、CE/MSE、L2、累积梯度与手算/直接大批参考相符（BN 差异需隔离说明）。
- 分配局数精确；记录真实原因；取消和异常局无伪造 z；每个 episode 只提交一次。
- 数据/训练指标落盘，而非只在终端观察“loss 下降”。

### 阶段 D：升级回放、恢复与进程协议

**涉及：** replay、trainer、inference_server、pipeline、workers。

实现版本化完整 episode/分片、提交 manifest、模型发布原子写入和明确保留策略。优化回放索引访问，先用简洁可验证的结构，再考虑复杂优先采样。以测得峰值而非估算 payload 设资源上限。

推理进程 Ready/Error 握手、请求版本、非有限防线、显式 spawn context、所有路径清理；限制在途数据字节或给出可证明的消息大小界限。模型更新不与正在读取的快照文件竞争。

**门槛：**

- 同规格连续训练 vs 保存/恢复：参数、优化器、步数、下一步 LR 和已提交数据语义一致；另测试 C01–C03。
- 在局完成、分片写入、训练前后、发布前后、manifest 更新前后注入中断，恢复后无重复 episode/未说明的阶段重做。
- 推理启动失败、响应超时、worker 崩溃、队列阻塞、父进程取消均结束可控，无错误局入库、无长期遗留进程。
- CUDA 路径实际验证 CPU RNG 恢复、BF16/FP32 差异、finite 输出/梯度和峰值显存。CPU 通过不代替这些结果。

### 阶段 E：独立评估与网页接入

**涉及：** arena/benchmark/stats、backend、database、frontend。

建立留出起点、颜色配对、预算固定的评估集；训练晋升逻辑不再充当统计评估。保存每局棋谱、原因、配置和版本，统计使用真正的独立单位。随机/启发式为底线，强引擎比较另记录外部版本和预算。

网页所有可继续对局通过 game_id/revision 在服务器维护完整规则状态；落子和悔棋同一事务，AI 回复绑定请求时版本。页面能显示和棋、终局原因及服务模型版本。

**门槛：** 两客户端并发落子只接受正确版本；旧 AI 回复、重复点击、新局/悔棋中途返回都不污染新状态；终局后不落子；数据库故障不产生半更新。评估重复起点不会虚增独立样本数，小样本全和不输出零宽置信保证。

### 阶段 F：证明学习有效，再扩大模型或改动作空间

正确性闭环通过后，以多个独立训练种子和固定计算预算做学习曲线；比较固定旧版本、低门槛对手、留出战术/规则局面和较强引擎。训练损失、GPU 利用率、吃子次数都不能单独证明棋力提高。

先改变一个主要因素：历史输入、标量/WDL、网络规模、采样分布、搜索预算或动作 head；固定规则和评估，记录训练总生成步数、总优化步和实际计算时间。8100→新空间平面意味着 checkpoint/replay/action 全链版本迁移，不能仅改 POLICY_SIZE。

**门槛：** 曲线显示跨多个快照及独立起点的可重复收益；和棋/截断率、颜色差异和区间同时报告；性能优化不得破坏阶段 B 不变量。没有这些证据时，只能称“训练流程可运行”，不能称“复现了 AlphaZero 棋力”。

## 17. PR 处理建议与报告使用方式

PR #1 含有效改动：搜索价值方向、去重意图、集中推理、有界紧凑回放、显式实验模式、优化器与恢复状态等。不要整份丢弃，也不要因现有 10 个测试通过就合并为正确基线。先修搜索预留、持续 learner、完整规则状态与终局边界，再按以上阶段调整。

新增问题应拆成有清晰契约的提交，例如“预留节点完整释放并加守恒验收”“评估不回滚 learner”“权重导入与训练恢复分离”，避免把规则、编码、动作维度、WDL、并发和所有超参数同时重写。每个提交要说明复现输入、修复机制、有效配置及尚未验收边界。

用户提供的详细文本已作为逐模块深度和覆盖范围的参照重新核对。报告保留了其中对学习闭环、状态历史、工程恢复和分阶段实验的关注，并补充了新的复现证据。关于动作空间、WDL、概率掩码和压缩，本文分别给出成立条件与迁移要求：7200 只是候选编码，WDL 是可成立的变体，全动作 CE 有明确概率意义，当前压缩能够恢复已有编码。最终依据是原论文、当前代码、明确规则和可复现证据。

**最终判断：项目已有可运行的工程骨架，但还没有一个经验证、状态一致、持续更新的 AlphaZero 学习基线。重构应先恢复这个基线，再决定扩网、换动作 head 和追求棋力。此文件是本次唯一主审查文档，后续在同一文件维护修复状态与验收证据。**

## 18. 本轮代码改造与验收进度（2026-10-15）

**状态：已落地第一轮核心重构，不是“100% 复现已经完成”。** 第 1–17 节是 PR 头 40ff9cfe 的历史审查快照；本节说明当前工作区已经改了什么、哪些判断已有新证据、哪些仍阻断最终验收。旧审查脚本及旧 JSON 用于复现旧提交，不能直接当成新接口的检查结果。

### 18.1 用户要求与本次实现边界

目标是保留 AlphaZero 的学习、搜索与关键预算，仅对网络规模作本机资源适配，同时优化 CUDA 执行，并在浏览器显示所有并发对局。没有继续采用启发式监督、吃子权重、WDL 替代损失或晋升失败回滚。

原论文没有中国象棋的官方编码、重复/长捉裁定、噪声浓度等具体实现；这些必须作为有版本的适配，不能用“完全复现”掩盖未核实细节。Science 官方 Data S1 下载受到验证页阻挡，尚未取得该原始压缩包。温度日程、正则化系数约定、发布间隔等仍须逐项与原始伪代码核对。

### 18.2 已修改的核心路径

| 审查项 | 本轮改造 | 当前证据 |
|---|---|---|
| T01/T02 | 单一 authoritative learner；发布其快照；Arena 不再回滚训练；actor/learner 同源初始化 | CPU/GPU 闭环与 checkpoint 检查 |
| N06/T03 | 标量 tanh value + 最终结果 MSE，策略 CE；删除双重权重和启发式生成路径 | 真正 SGD 更新、uniform Replay 测试 |
| T04/T05 | 按 global_optimizer_step 调度；完整 batch=4096，不使用梯度累积，因此没有尾组缩放/BN microbatch 差异 | 默认 batch 实际 CUDA 运行 |
| S01 | 显式路径预留与统一释放；重复、推理异常都撤销叶节点自身 | 大批搜索、单一应手、异常注入和计数守恒测试 |
| S02/S03 | 终局无动作；合法 priors 归一化，移除未归一化 floor | 固定用例与概率检查 |
| S04/S05 | Player 转换统一；动态 PUCT 系数；选动作分布与训练访问数目标分开 | 红黑、价值备份和终局用例 |
| S06 | 落子后复用子树；核对完整规则历史；新根移除原 incoming leaf 的一次计数 | 复用前后计数守恒测试；每手仍完成声明的新模拟预算 |
| G02/G04/G05 | 将死/困毙先于人为截断；拒绝混乱颜色；快照公开盘面只读、核心字段冻结 | 边界将死、黑方胜、外部改盘面检查 |
| G03/N02 | 8×16+3=131 通道，当前玩家朝向；模型/回放/推理一致 | 历史往返、黑方策略坐标映射 |
| D01–D04 | 新 SQLite 回放，保存完整局的轨迹和每手版本，lossless 二进制观测及稀疏 float32 pi；均匀位置采样 | 完整局 FIFO、重开恢复、负策略/损坏目标拒绝 |
| C01–C04 | SGD learner 与 actor 导出明确区分；结构指纹、run ID、回放代次校验；CPU RNG 在 CPU 加载；原子写入 | CPU 继续更新、CUDA 参数恢复、跨 run 和旧优化器拒绝 |
| I01–I04 | 显式 spawn、Ready/Error 握手、版本化合批、非有限输出失败响应、正常退出排空已完成局 | 实际子进程、NaN 模型、版本固定与进程退出检查 |
| E01/E02 | 评估完全脱离训练门控；移除零方差窄区间，确定性同起点结果不给虚假精度 | 统计单元检查；多样留出评估集仍待建立 |
| K01–K03 | 严格平铺配置、未知键报错；实际传递通道；小型检查配置隔离；重写启动说明 | 配置检查、实际命令启动 |
| 新需求 | 独立只读 HTTP/SSE，活跃对局动态棋盘、损失和资源信息 | 浏览器真实 4 盘、8 盘；25 盘状态接口扩展检查 |

当前代码入口：

- [连续训练管道](D:/project/Alphazero/AlphaZero/scripts/pipeline.py)、[配置规格](D:/project/Alphazero/AlphaZero/train/config.py)、[SGD learner](D:/project/Alphazero/AlphaZero/train/trainer.py)。
- [规则状态与历史编码](D:/project/Alphazero/AlphaZero/engine/state.py)、[标量 Policy/value 网络](D:/project/Alphazero/AlphaZero/model/network.py)、[搜索实现](D:/project/Alphazero/AlphaZero/search/tree.py)。
- [磁盘回放](D:/project/Alphazero/AlphaZero/train/replay.py)、[中心推理服务](D:/project/Alphazero/AlphaZero/train/inference_server.py)、[连续自对弈](D:/project/Alphazero/AlphaZero/train/self_play_worker.py)。
- [观测服务](D:/project/Alphazero/AlphaZero/train/live.py)、[浏览器页面](D:/project/Alphazero/frontend/training.html)、[前端逻辑](D:/project/Alphazero/frontend/training.js)。
- [启动说明](D:/project/Alphazero/README.md)、[默认预算](D:/project/Alphazero/config/default.yaml)、[WSL 启动脚本](D:/project/Alphazero/scripts/train.sh)。

### 18.3 网络规模、数据规模与单机执行约束

新默认模型为 **19 个残差块、128 通道、7,254,267 参数**：保留论文深度，只把 256 通道缩为 128，另用平铺策略 head 适配动作编码。先后实测了 4×64、8×128、19×128；完整 4096 batch 的峰值预留显存分别约 1.85GB、4.05GB、7.97GB，因此没有继续把不必要的小塔当成默认。训练完整 batch=4096，SGD momentum=0.9，LR=0.2，在 100k/300k/500k 更新后衰减，总更新预算 700k。当前按论文显式公式 c||theta||² 取 c=1e-4，对应 PyTorch weight_decay=2e-4；这项系数约定仍待 Data S1 逐行核对，不能仅凭公式就声称原程序完全一致。

[论文第 27 页 Table S3](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf#page=27) 给出国际象棋约 **4400 万训练局 / 70 万更新**。本机 actor 数量远少于原实验，GPU 更新又比生成新局快。为保留声明的数据规模，默认设置 training_games=44000000，按累计完成局数线性限制 learner 更新，首次更新等待 ceil(44000000/700000)=63 局。该限速是**显式的单机执行约束，不是宣称论文也用了相同的限速器**。完整复现实验的耗时尚未测算；GPU 等待新对局时不能把这种等待误诊为 CUDA 没启用。

小预算检查可显式设 training_games=0，标记为 smoke。800/4096 的短运行也只是搜索和训练 batch 的执行验收，不等于已经完成 4400 万局的训练数据预算。日志记录累计生成局面数及 training-to-data 比率，防止“loss 下降、更新很多、实际新数据极少”被当成学习成果。

回放按最近 100 万**局**管理，使用磁盘和 256MB SQLite 缓存；没有把它偷偷替换成 100 万条局面上限。实际长时磁盘容量与 I/O 仍需按局长测量。已提交数据可以比最近 learner checkpoint 更新；恢复契约是保留这些已提交回放、从完整 learner 步继续，拒绝跨 run、缺失已引用回放、或回放来自比该 learner 更新的已发布模型。不是逐位重现异步进程调度的承诺。

### 18.4 浏览器真实对局与扩展方式

训练进程直接提供 http://localhost:8765，不依赖旧 FastAPI 棋局数据库。每个活跃 episode 对应一个动态卡片；事件中按 worker/sequence/ply 标记，结束后移除，迟到事件不能复活已结束棋局。恢复连接得到完整最新快照，而不是依赖浏览器记住每一条事件。

页面显示每盘行棋方、棋盘、最后落子、手数、耗时、实际 actor 版本、上一次搜索模拟数，并显示优化步、策略/价值损失、完成局数、回放量及硬件使用。默认只监听本机；不同 run 可设置不同端口。WSL 内存与主机内存区分显示。

观测队列有界且允许丢帧，完整训练局走独立可靠队列。慢浏览器、断开重连不会阻塞规则搜索或生成错误标签。浏览器已实际看见 CUDA 4 盘和默认 8 盘不同步推进；接口测试另检查 25 个活跃盘面，**没有把 25 盘界面检查误称为本机已实际并发搜索 25 盘**。

### 18.5 本轮验收证据

稳定环境已建立在 WSL /home/lin/.venvs/xiangqi-alphazero，PyTorch 2.8.0+cu128；识别 RTX 5070，compute capability (12,0)，CUDA 实际矩阵运算、前向、反向及恢复均执行成功。主机为 32GB，当前 WSL 可见约 15.2GB；本轮没有修改 WSL 系统分配。

| 检查 | 当前结果 | 边界 |
|---|---|---|
| 算法/回放/恢复/服务回归 | **24 passed** | 覆盖新契约，不能代替完整规则、所有进程故障和长时训练 |
| 默认 19×128、4096 batch CUDA | 两次真实更新完成，loss/梯度/参数/优化器状态有限 | 固定合成批硬件检查，无棋力结论 |
| 显存 | peak allocated 约 7.774GB，reserved 约 7.973GB | 独立 learner 测量，不是并发运行总峰值 |
| BF16 与 FP32 | 本次固定局面策略最大绝对差约 2.53e-4，value 差约 5.99e-4 | 非全部局面误差界 |
| CUDA checkpoint | 参数完全相等，优化步恢复为 2 | 非全训练轨迹位级重现 |
| CPU 新启动与继续训练 | 新 run 到 step 4，再从同一 learner/replay 到 step 6 | 小网络、小预算 |
| CUDA 多进程小预算 | 4 actor、32 次搜索、100 更新完成 | 显式 smoke，不代表论文预算 |
| 完整搜索/batch 短运行 | 8 actor、800 次新模拟、4096 batch，完成真实局和两次更新 | 数据预算限速关闭；不冒充4400万局实验 |
| 虚拟预留与子树 | 大 batch、单一应手、失败释放和复用计数守恒 | 仍可继续扩展随机/长期不变量检查 |
| 推理版本/故障 | 一局固定旧快照、新局能切新快照；NaN 模型返回错误 | 未做完整系统断电、显存耗尽、网络故障矩阵 |
| 页面 | 真实活跃盘面随训练更新，显示数与活跃列表一致 | 不宣称已完成所有浏览器/设备验收 |

当前证据：[硬件 JSON](D:/project/Alphazero/references/implementation_hardware.json)、[回归输出](D:/project/Alphazero/references/implementation_tests.txt)、[搜索/batch 短闭环](D:/project/Alphazero/references/implementation_full_budget.txt)、[继续训练输出](D:/project/Alphazero/references/implementation_resume.txt)。辅助脚本仅验证对应执行场景，不是另一份审查报告。

### 18.6 仍然阻断“100% 完成”的事项

1. **完整象棋规则：** 目前是明确标记的开发规则，长将已有固定检查；长捉、混合循环、责任例外、无吃子特殊计数及完整独立裁判验收尚未完成。此时不应投入最终正式规则的大规模训练，未来必须换 rule_version，隔离旧回放。
2. **原始 Data S1：** 需要取得官方文件，核对训练温度日程、完整 PUCT/初始化细节、正则化系数、发布间隔等；当前以已获取正文/Methods 为依据的选择已写明，不能把未核实内容自动当成完全一致。
3. **旧网页对战链路：** W01–W05 的权威规则状态、revision/CAS、悔棋事务和迟到 AI 请求尚未全部迁移。新的训练观战不依赖这些旧接口，但不能因此关闭原审查问题。
4. **真实规模和长时正确性：** 一百万局窗口的磁盘压力、完整故障注入、长时 CUDA 数值、统计上独立的棋力评估和学习曲线仍待验收。
5. **充分利用硬件的下一阶段：** 已验证 BF16、channels-last、合批、树复用和磁盘回放；CPU 规则/选叶仍可能限制 GPU 供给。需要按真实有效批、队列、CPU 走法和端到端耗时继续剖析，再决定原生规则内核、共享缓冲或编译优化。GPU 占用或显存填满不是优化完成的证明。

本节记录的是可运行、可核对的第一轮改造。后续仍在本文件更新，直到目标规则、来源核对、工程恢复和学习/评估全部满足验收条件，再使用“复现完成”的表述。


## 19. 用户确认的目标规则与待上传来源

用户已明确选择中国象棋竞赛规则，规则目标据此固定为中国象棋协会审定的《象棋竞赛规则（2020版）》，替代此前暂拟的 WXF 2018。国家体育总局棋牌运动管理中心发布的 [2026 年全国象棋等级赛规程](https://www.sport.gov.cn/qpzx/n5387/c29676858/content.html) 仍明确执行该版。

本次同步修改目标说明与页面标签，没有把现有开发版裁定器冒充为已实现国规，也没有改变已生成数据的实际规则语义。完整国规实现时需另发规则版本，按该版逐条覆盖长将、长捉、循环责任、例外与限着计数。

用户将自行尝试下载官方 Data S1。官方下载入口为 [aar6404_datas1.zip](https://www.science.org/doi/suppl/10.1126/science.aar6404/suppl_file/aar6404_datas1.zip)，也可从 [论文页面](https://www.science.org/doi/10.1126/science.aar6404) 的 Supplementary Materials 选择 Data S1。自动访问仍被出版社验证/403 阻挡；收到文件前，不改变“尚未取得原始 Data S1”的证据状态。

## 20. 收到 Data S1 后的逐项复核与纠正

### 20.1 来源与本节效力

2026-10-15 用户提供目录 [aar6404_datas1](D:/project/Alphazero/aar6404_datas1/)，已完整阅读其中 [pseudocode.py](D:/project/Alphazero/aar6404_datas1/pseudocode.py)。文件大小 10,298 字节，SHA256 为 `9c0d7092baa1687bb7169318eb36ff004784dca38d082db925309ebc0697b6fe`。本地材料来自用户上传；没有声称重新从出版社独立下载并校验过签名。附带 PGN、CSA 和 JSON 是原实验材料，不是可以直接拿来监督训练中国象棋的样本。

**本节替代第 18–19 节中“Data S1 尚未取得”、63 局硬限速、weight_decay=2e-4、默认复用子树和默认单树批量搜索的当前状态说明。** 前节实测结果保留为旧算法版本的历史证据；不能拿它们自动证明本节修正后的版本已完成同样的全链路验收。仍只维护这一份主审查报告。

伪代码是说明材料，含 `pass`、固定返回值、未实现的 softmax_sample、旧版 Python/TensorFlow 接口。已通过阅读和 AST 提取常量核对，**没有导入或执行附件**，也没有把其中内容当作新的用户操作指令。

### 20.2 核对矩阵与已经落实的修改

| 事项 | Data S1 证据 | 修正后的实现与判断 |
|---|---|---|
| 温度日程 | 22、233–241 行：history 长度小于 30 采样，否则 max(visits, action) | `selfplay_temperature` 在 ply 0–29 返回 1，ply 30 起返回 0；30 指半回合，并非红黑各走 30 次。训练 pi 仍为原始访问数归一化，不能改为 greedy one-hot |
| 根搜索预算 | 213–230 行：每次 run_mcts 新建根，逐次做 800 次模拟 | 默认不复用树，单树 batch=1、virtual_loss=0；每手根访问数严格为 800。子树复用、单树批量选叶仅保留为显式实验选项，不再称与参考搜索等价 |
| PUCT | 252–259 行：动态系数、sqrt(parent.visit_count)、child N+1 | 移除 sqrt(max(N,1)) 的额外下限；零访问根的探索分数为 0。分数相等按动作索引较大者选，与 244–247 行 tuple max 一致；greedy 动作计数平局也如此 |
| 根噪声 | 27–28 行与 285–290 行 | alpha=.3、混合比例 .25；保留真正归一化的 Dirichlet，未照抄未归一化 Gamma 的省略，见 20.3 |
| 网络发布 | 35–36、300–310 行：700k 更新、每 1000 步发布、step 0 和最终发布 | 默认 `checkpoint_interval=1000`，替代旧 100；初始化发布 step 0，更新到 1000 时发布该快照，最终发布；未把每局结束当模型晋升 |
| 异步训练 | 165–195 行：独立 learner、actor，通过快照和完整局交换 | 删除 training_games、required_selfplay_games 和 63 局等待器。4400 万是 Table S3 实验产出，不是 Data S1 每次优化的硬配额；首个完整局可采样后继续异步训练 |
| 每局模型一致性 | 189–195 行：局开始只取一次 latest_network | 保留每局锁定版本、不同版本分组合批。step 0 初始化后启动 actor 是伪代码并发的一种有效排序；不强制先用 uniform 网络生成整代数据 |
| 回放粒度与采样 | 121–143 行：百万局窗口，按局长抽局再均匀抽 ply | 保留最近 100 万完整局；均匀局面采样与该两阶段分布等价。不是等概率抽局，也不是百万局面窗口 |
| 优化器/学习率 | 40–48 行、300–310 行 | SGD momentum=.9、batch4096、LR .2/.02/.002/.0002，按全局更新数 0/100k/300k/500k 切换；总上限 700k |
| L2 系数 | 40、323–324 行：1e-4 × tf.nn.l2_loss | 修正 SGD weight_decay 为 1e-4；对应损失 c/2 × sum(theta²)。旧 2e-4 是把论文符号 c 当同名 TensorFlow 系数造成的差异 |
| 批损失归一化 | 313–326 行：循环逐样本相加，没有显式除 batch | 新增 `loss_reduction` 并纳入规格指纹：默认 mean，字面 sum 为核对实验。两者的梯度分别有自动检查；**这里仍是明确差异，并未关闭为完全一致**，见 20.4 |
| CUDA 与并发 | 21 行 num_actors=5000；无本机调度实现 | 保留跨对局 GPU 合批、BF16、channels-last、磁盘回放。默认 8 actor 是本机执行安排，生成速度/数据复用比与 5000 actor 原实验不同，尚未证明轨迹或棋力等价 |
| 版本与恢复 | 附件未实现工程 checkpoint/恢复 | 新 schema=3、algorithm=alphazero-datas1-aligned-v3；名称表示本轮对齐过程，不是完成认证。回放版本包含算法，默认新目录 paper-v3，拒绝旧数据/状态隐式续训。mean 与 sum 指纹不同；resume 校验温度、复用与发布配置 |

动作编码 8100、131 输入、中国象棋终局规则不是 Data S1 给出的定义；其中 Game.make_image、legal_actions、terminal_value 都是桩函数。保留这些适配不能当作“官方源码已证明”。19×128 网络仍是约 725 万参数的架构让步，不使用 4×64 小检查网络冒充默认。

### 20.3 不应机械照抄的省略与矛盾

1. **Gamma 并不自动成为 Dirichlet。** 287 行取独立 Gamma，随后直接混入先验，没有除以 Gamma 总和。正文说 Dirichlet，因此这里保留 `rng.dirichlet`；否则 priors 不再和为 1，探索项规模会随随机总和改变。此处按正文概率定义处理，并记录伪代码字面差异。
2. **价值视角存在歧义。** 277–280 行按 node.to_play 备份 0–1 值，但 258 行直接加 child.value；按“子节点轮到对方”的常见状态语义，这与父方选择方向不一致。正文使用 [-1,1] 的结果和值，当前实现据此逐层变号、父方使用 -child.q。未访问 Q=0 也不能直接用概率值换元来声称 UCB 严格等价：换元涉及常数和探索项尺度。附件不足以唯一还原原训练程序的全部内部约定，这一点继续开放。
3. **终局不能继续问网络猜。** run_mcts 的 evaluate 未展示 terminal_value 分支，Game 本身又是桩函数。当前搜索遇到终局使用规则真实结果，不按缺少终局分支的字面路径产生标签。
4. **FIFO 边界。** save_game 在追加前判断 len > window_size，会实际保留 window+1；当前实现精确保留目标最近 N 局，按正文“窗口”语义处理，未复制这个边界偏差。
5. **softmax_sample 未实现。** 名称不能证明应对原始次数再取 exp；结合论文访问数温度定义，前 30 ply 按次数比例采样，后续取最大；测试将选动作与学习目标分离。
6. **初始化不是一套已交付的网络代码。** Network.inference、get_weights、make_uniform_network 是占位实现。没有据此宣布已核实初始化分布、BatchNorm 超参数、全部网络算子或原实验浮点数行为。

上述取舍均显式列出，不能用“复现”一词掩盖来源中的空缺，也不能以修复省略为由加入未声明的启发式、Arena 回滚或吃子权重。

### 20.4 损失归一化：发现了真实失败，不能宣布解决

官方 [TensorFlow l2_loss 定义](https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss) 为 sum(t²)/2。因此，SGD 的耦合 weight_decay=c 对应 c/2 × sum(theta²)，本轮已纠正为 c=1e-4，并用实际参数更新验证梯度，非仅核对日志数字。

Data S1 的 outer loop 明确将每个样本的 MSE+CE 相加；代码没有给出 batch 归一化。按字面将数据损失从 mean 改为 sum、保留 batch4096 和 LR .2，在本机 19×128/BF16、固定初始局面重复组成的合成批上，两次更新后的 FP32 推理出现 `FloatingPointError: non-finite neural output`。固定 seed=20261015 重复核对仍失败。检查过程中训练损失和参数有限，不代表 eval 模式推理一定有限；本轮也扩展训练后有限性检查到模型 buffers。

失败证据：[datas1_sum_check.txt](D:/project/Alphazero/references/datas1_sum_check.txt)。另外，发布 actor 前现已增加 eval 模式的实际局面前向检查；若失败，保留旧发布文件，不用只检查训练参数有限就放行。单一初始局面探针不能证明全部状态有限，推理服务原有的逐请求检查继续生效。可复现入口是在 WSL 使用独立环境运行：

~~~bash
python -m AlphaZero.scripts.runtime_check --config config/default.yaml --loss-reduction sum
~~~

同环境、同 seed、同 batch 的 mean 模式完成两次更新和参数恢复。**这只是一个数值反例，不是证明 Data S1 错误，也不是证明原训练使用 mean。** 合成重复局面、缩窄网络、初始化、BN 和精度均影响现象；本轮未完成逐项归因。不能从这个实验推导“调低 LR 就等价复现”：同时涉及 MSE 与 CE 的尺度、L2 相对权重、动量和非线性优化轨迹。

当前默认明确选择论文逐局面目标的 batch 均值，同时保留字面 sum 作为可重现核对模式；此处属于**算法复现未决项**，不是用户已经授权的“模型架构让步”。在拿到更多原实现证据或完成来源/数值归因前，不应称“只改架构，其余100%一致”。未静默改 LR、加裁剪或降低 batch 来让检查通过。

### 20.5 新版本验收与证据边界

- 新回归 **28 passed**，见 [datas1_tests.txt](D:/project/Alphazero/references/datas1_tests.txt)。覆盖附件常量提取、29/30-ply 温度边界、全访问数 pi、新树预算、零访问根与动作平局、mean/sum 实际 SGD 梯度、半平方 L2，以及原有规则/回放/并发/恢复检查。附件不需要 TensorFlow 或导入执行。
- 默认 19×128、4096 batch、mean、CUDA BF16 的两次更新完成：约 2.522 秒，peak allocated 约 7.774GiB、reserved 约 7.973GiB；checkpoint 参数恢复相等、step=2，见 [datas1_hardware.json](D:/project/Alphazero/references/datas1_hardware.json)。独立 learner 测量不能冒称整个并发系统峰值。
- 本次固定样本 BF16 与 FP32 的策略最大绝对差约 0.00160、value 差约 0.000913；不同于上一版本的实测值，不能宣称全状态数值等价。
- CPU 新 run 到 step4，从同一 learner/replay 恢复到 step6，见 [datas1_resume.txt](D:/project/Alphazero/references/datas1_resume.txt)；这是小预算恢复检查，不是长时轨迹重现。
- 默认参考搜索实际运行约五分钟，状态接口确认 **8 个不同活跃对局均到 ply22**，最近搜索 simulations=800、reused_visits=0，见 [datas1_live_snapshot.json](D:/project/Alphazero/references/datas1_live_snapshot.json)。随后主动发出中断，训练管道执行清理；[datas1_full_budget.txt](D:/project/Alphazero/references/datas1_full_budget.txt) 保留该中断记录。**本次尚未等到首局完成，没有完成新版本的 800/4096 真实对局两更新闭环**；不能沿用第 18 节的批量树结果关闭此项。

### 20.6 当前尚未满足的最终目标

Data S1 获取问题已解除，待办变为：损失归一化及价值/UCB 约定的来源与数值归因；完整国规2020长捉、混合循环、责任例外和独立裁判验收；单机 actor 生成量与 learner 数据复用比的长期实验；真实规模磁盘、故障恢复与独立棋力评估。默认串行单树搜索减少了单局推理批量，下一轮吞吐优化必须在保持搜索决策依赖顺序的前提下推进；跨局合批可以保留，树内同时选多叶不能直接列作等价优化。

本轮没有启动正式 70 万步训练，也没有删除用户上传材料、旧 run 或旧证据。浏览器多盘功能保留，移除了来自错误固定配额的“下一更新需多少局”提示；完整国规未验收的标签继续显示。

## 21. 从短程验收转为真实模型训练与使用

用户明确要求产出模型供实际使用，因此已启动持续训练，而非以第 20 节的短检查作为任务完成。输出目录为 [xiangqi-v3-20261015](D:/project/Alphazero/AlphaZero/runs/xiangqi-v3-20261015/)，run ID 为 `2a0db781-55a1-46c7-8622-7c09f21182d6`。进程通过 WSL2 在后台运行，训练页面为 http://localhost:8765 。

### 21.1 实际产出的模型

已产生并保留不可变的 [model-step-001000.pt](D:/project/Alphazero/AlphaZero/runs/xiangqi-v3-20261015/model-step-001000.pt)。它完成了 **1000 次真实优化更新**，结构为 19×128，batch4096，8 个并发 actor，每手800次顺序搜索。发布时回放有2盘真实完整对局、70个局面。与保留的初始化文件比较，252个状态张量全部发生变化，见 [model-verification.json](D:/project/Alphazero/AlphaZero/runs/xiangqi-v3-20261015/model-verification.json)。状态张量包括训练参数和BN缓冲，不能把该数值误写为252组独立学习效果。

`latest.pt` 是每1000步更新的自对弈快照，`learner.pt` 是恢复训练状态，`model-step-001000.pt` 是本次留存的初版模型；不把step0文件交付为训练成果。第1000步时指标约为loss6.886，单更新约0.619秒。原始数值在该run的metrics.jsonl中，包含真实样本量与复用比。

### 21.2 实际使用入口与检查

新增 [ModelPlayer](D:/project/Alphazero/AlphaZero/model/player.py) 严格核对结构指纹并拒绝step0初始化；新增 [浏览器对弈服务](D:/project/Alphazero/AlphaZero/scripts/play_server.py)，地址 http://localhost:8767 。该入口使用完整历史规则状态、当前模型和真实MCTS；校验合法走法、game_id和revision，换局后旧AI结果不能覆盖新局。

已在浏览器实际点击红兵从(6,0)到(5,0)，1000步模型完成800次搜索，合法应手黑卒从(3,4)到(4,4)，页面更新到第2手并回到人类行棋。最初使用CPU/FP32核对，状态保存在 [browser-play-verification.json](D:/project/Alphazero/AlphaZero/runs/xiangqi-v3-20261015/browser-play-verification.json)。随后接入同一模型的CUDA Graph推理，在持续训练及评估同时运行时，同样800次搜索完成该应手约7.986秒，无运行错误，见 [browser-play-cuda-verification.json](D:/project/Alphazero/AlphaZero/runs/xiangqi-v3-20261015/browser-play-cuda-verification.json)。该初始位置测量不能当所有局面的延迟保证。

新增 [独立评估入口](D:/project/Alphazero/AlphaZero/scripts/evaluate_model.py)，记录具体模型步数、精度对应的设备、搜索预算、逐局胜负和终局原因。8盘随机对手评估正在运行，逐局结果即时保存，不能把未完成的局数算入胜率。

### 21.3 执行加速与恢复

发现小批推理的CUDA算子启动开销很高，新增不可变eval快照的 [CUDA Graph重放](D:/project/Alphazero/AlphaZero/model/cuda_graph.py)。不并发选取同一棵树的多个依赖叶，不减少800搜索预算，不更改训练目标。固定输入检查中，batch1/4/8的前向输出最大绝对差均为0；包含输出回传的实测时间分别从约11.84/8.79/16.83ms降至0.96/0.95/1.17ms，见 [cuda_graph_benchmark.json](D:/project/Alphazero/references/cuda_graph_benchmark.json)。这项孤立测量受同时运行的训练影响，不能直接宣称整个训练加速12倍。

另以4 actor、小网络、32搜索、128 batch完成4盘与100更新的独立CUDA Graph闭环，见 [cuda_graph_acceptance.txt](D:/project/Alphazero/references/cuda_graph_acceptance.txt)。完整默认模型也在真实持续运行中使用该推理路径。补充了非二次幂batch填充、重复输入及模型加载/人机状态测试，当前回归31项通过。

为接入加速曾中断第一段训练并从step0恢复，**此前108次尚未形成checkpoint的更新没有保留**；已提交完整局保留，因此日志里出现108之后重新从1计数，不能把两段数字相加宣称优化步数。本轮新代码增加独立recovery_interval=100和session_id，恢复存档不等于向actor发布新版本；当前已启动的进程以它启动时载入的代码为准，下一次启动/恢复才应用之后修改的恢复频率和回放解码优化。

新增 [PowerShell管理入口](D:/project/Alphazero/scripts/training.ps1)，Status查询实际进程，Stop仅定位同一run发送中断，Resume继续同一目录，避免后台进程无法管理。训练不会因本轮答复结束而主动停止；机器关机或进程失败后仍需明确恢复。

### 21.4 当前模型质量边界

**已经产出可加载并下棋的初版；尚未证明达到用户所需的实用棋力。** 发布时仅70个局面被反复抽样，1000×4096/70约为58514，数据复用非常高。loss下降和参数变化不能排除过拟合，更不能证明已完成AlphaZero复现。评估中的GPU/BF16与CPU/FP32可因数值和搜索分支不同得到不同结果，不能合并成同一条精确训练轨迹。

后续结论必须以更多完整自对弈、独立评估和实际模型表现为依据；完整国规、损失归一化及单机数据生成/优化速度不平衡仍未解决。此处保留真实运行结果，不用“模型能走棋”替代“训练已经成功”。

### 21.5 已完成的对局结果与32盘持续运行

| 模型与执行 | 对手 | 搜索预算 | 实际结果 | 证据与限制 |
|---|---|---|---|---|
| 1000步，CUDA/BF16 | 随机合法走法 | 每手64次；8盘红黑各4 | **4胜、4和、0负**，得分率75% | [完整结果](D:/project/Alphazero/AlphaZero/runs/xiangqi-v3-20261015/evaluation-random-cuda.json)，约817.7秒；这是得分率，不是75%胜率；区间仍为[0,1]，不能宣称强棋力 |
| 2000步，CUDA/BF16 | 仓库启发式棋手 | 每手64次；红黑各1 | **0胜、1和、1负** | [完整结果](D:/project/Alphazero/AlphaZero/runs/xiangqi-v3-20261015/evaluation-heuristic-cuda.json)，约130.4秒；同一确定起点、样本极少，不给虚假Elo |
| 1000步，CPU/FP32 | 随机合法走法 | 每手64次；8盘红黑各4 | **4胜、1和、3负**，得分率56.25% | [完整结果](D:/project/Alphazero/AlphaZero/runs/xiangqi-v3-20261015/evaluation-random.json)，约1612.1秒；不与CUDA结果合并，同样不足以推导成熟棋力 |

第2000步模型已另存为model-step-002000.pt，第3000步另存为 [model-step-003000.pt](D:/project/Alphazero/AlphaZero/runs/xiangqi-v3-20261015/model-step-003000.pt)。这些快照均不是网络随机初始化，但目前对启发式对手的检查已经表明棋力尚未合格。

为提高新数据供给，在第3000步完整发布后中断并从该完整checkpoint继续；第3001步等未提交checkpoint的更新不会冒算进恢复步数。当前实际使用 [local-train.yaml](D:/project/Alphazero/config/local-train.yaml)：**32盘独立对局、8个CPU工作进程、每进程4个线程槽**。每个槽有自己的RNG、规则历史、MCTS树、请求/响应队列和整局网络快照；线程仅重叠独立对局等待，不同时选同一棵树的依赖叶节点。学习率、batch4096、800次顺序搜索、访问数目标、百万局窗口均保留。

现场已确认32个槽全部推进到至少第2手，模型版本3000，上一次搜索均800次；恢复后learner超过3100步，run ID保持相同，没有进程错误。界面明确区分“32盘并发”和“8个工作进程”。新进程实际已写入step3100恢复文件，独立100步恢复频率生效。此次32盘尚未完成完整一轮，不能据此宣称已量出稳定的数据生成倍率。

另外缓存同一搜索节点对应的不可变规则状态，避免每次沿共同路径重复执行走法和合法性生成；不跨落子复用搜索统计。扩展后的神经输入缓存及时释放，限制32盘内存占用。固定800次、相同随机种子、常数评估器的三次CPU对比中，访问分布完全相等，中位耗时约1.384秒降至0.750秒，约1.85倍，见 [state_cache_benchmark.json](D:/project/Alphazero/references/state_cache_benchmark.json)。不能把这一隔离数字当作真实系统整体倍率。

新增组内独立身份/序号、组内异常停止与状态缓存等价检查，当前 **34项回归通过**，见 [training_delivery_tests.txt](D:/project/Alphazero/references/training_delivery_tests.txt)。真实CPU多槽恢复到step6与CUDA多槽4局/100更新的记录分别在 [group_recovery_acceptance.txt](D:/project/Alphazero/references/group_recovery_acceptance.txt)、[group_cuda_acceptance.txt](D:/project/Alphazero/references/group_cuda_acceptance.txt)。模型、实训、界面和棋力证据分别记录，避免再次用测试通过代替可用模型。

交付时训练已超过3500步，32个活跃槽均在推进，errors为空，见 [training-32-snapshot.json](D:/project/Alphazero/AlphaZero/runs/xiangqi-v3-20261015/training-32-snapshot.json)。人机服务已切至latest.pt，新局自动读取最新已发布快照、局内保持同一版本；当前网页实际载入3000步模型。该模型再完成一次真实800次搜索应手，约6.948秒，见 [browser-play-3000-verification.json](D:/project/Alphazero/AlphaZero/runs/xiangqi-v3-20261015/browser-play-3000-verification.json) 与 [实际页面](D:/project/Alphazero/AlphaZero/runs/xiangqi-v3-20261015/browser-play.png)。这证明实际使用链路，并不改变上面“棋力尚未合格”的评估结论。

### 21.6 修改搜索次数后续训（2026-10-16）

按用户要求，取消 `num_simulations` 在 learner 恢复时的相等检查。搜索次数不改变模型张量、输入编码或回放目标格式；继续恢复权重、SGD 动量、累计更新次数及随机数状态，并保留调用者指定的新搜索次数。其他训练语义和模型兼容性检查仍在。旧回放继续保留，不要求重新生成。

恢复后的新进程从当前配置创建 MCTS。每次启动在 `metrics.jsonl` 追加完整配置、恢复步数及旧值/新值差异，每次更新记录当前搜索次数；新生成完整对局的 trace 记录其搜索预算。`config.json` 与 checkpoint 保存当前配置，追加日志保存变化历史；旧日志与旧棋谱无需迁移。旧版本生成的数据未记录搜索次数，不能据此补造历史。

新增13项检查：恢复预算1/3/32/800时权重、动量、步数与Torch随机数状态保留；非法预算拒绝；其他受限参数继续拒绝；真实 worker 每手恰好完成1/3/17次搜索且目标有效；独立临时目录实际完成8→3→11次搜索、累计2→4→6次更新的三段训练，旧棋谱逐项保留，新棋谱预算与追加历史一致。相关搜索、回放、恢复、并发与对弈回归合计 **43项通过（20.91秒）**。检查使用CPU小模型与短局，不代表完整模型棋力验收。本次没有停止或修改持续运行任务的配置。

## 22. 独立规则状态修正分支（2026-10-16）

本节针对用户上传的网页版分析，先处理底层颜色转换与训练、MCTS、旧 Web 的完整规则状态一致性。分支 `codex/rule-state-consistency` 基于438ac18，使用独立Git worktree；原运行目录和 `review/pr-1` 分支不切换、不修改。50平面策略头、长捉/长杀/将捉、循环责任和国规例外裁定不在此轮实现，不能将本轮通过等同于完整国规验收。

底层原先 `bool(-1)==True` 的问题确认存在于合法走法、定点走法检查、结果检查及有无合法走法等接口。共用的player_id转换移入独立模块，避免state与fast_chess循环导入；将军检测、重复局面key、GameState也使用同一入口。布尔值、正负1及r/b字符串可互换，其他值明确拒绝。现有自对弈传布尔值的行为保持一致。

旧Web不再从请求棋盘创建无历史的新GameState，也不再用独立棋盘规则函数确定训练相关终局。服务端存储明确的起始局面、规则版本、最大手数和完整着法记录，每次通过GameState.apply重放，恢复重复次数、无吃子计数、将军历史、最近八帧、总手数和last_move。走棋与悔棋均使用SQLite事务和单调递增revision，阻止并发写入、同棋盘但不同历史的旧请求以及AI思考期间已失效的结果。旧棋谱可验证重放时兼容；缺步、错子或棋盘不一致时返回409，不补造历史。该方案每次读取重放棋谱，性能缓存可另行优化。

API明确要求game_id与revision，随本分支前端一同更新；仅传board/turn的旧客户端返回422。响应给出规则版本和终局原因。旧手工API脚本改为显式启动，pytest导入时不会向服务发送走棋。独立8767 PlaySession原本持有GameState，相关回归继续通过。旧神经Web搜索沿用已训练模型配置的搜索次数，并通过当前ModelPlayer检查模型规格。

验证使用独立临时数据库、进程内FastAPI客户端及CPU小模型，显式隐藏GPU，不访问现有服务端口。回归68项通过、1项GPU检查跳过（41.18秒）；前端JavaScript语法检查通过。新增规则检查覆盖正负颜色、非法颜色、重现相同棋盘后的历史差异、三次重复、单方长将、吃子清零、120手未吃子、max_ply、悔棋、搜索收到完整状态、真实小模型搜索、并发提交、失效AI结果和旧数据库兼容。依赖的一项httpx测试客户端弃用提示不影响结果。以上仅验证现有开发版规则的一致性。

## 23. 自然限着的将军计数（独立分支，2026-10-16）

依据[2020版第4条](https://cnchess.net/rules/ChapterOneSection4.html)及[第22条](https://cnchess.net/rules/ChapterFiveSection22.html)，实现60回合与提出方将军最多计10着的计算。条文针对提出方，并不要求将双方的超额将军相加，也不要求双方同时拥有申请条件。自对弈采用显式工程约定：轮到行棋的一方视作自动提出方，满足其条件自动判和；申请、审核扣时等真人流程不模拟。

从最后一次吃子之后，分别保存红黑将军计数，吃子后同时归零；检查超过10次的着数，从提出方自己的有效着数扣除。有效完整回合为min(原始未吃子半回合数//2, 提出方未吃子着数-max(提出方将军数-10,0))。例如120半回合、红方15将且轮到红方时，红方仅55个有效回合；若之后一着轮到黑方，而黑方将军不超10，则黑方可以满足60回合。此前错误地只看120半回合的立即判和已替换。将死/困毙、已实现的重复/单方长将裁定仍优先。

计数进入不可变GameState、MCTS复用检查、Web完整棋谱重放/悔棋及观测API。直接导入非零未吃子计数的局面时，必须同时提供对应将军历史或显式双方计数，不能默认历史上的将军为零。检查次数随当前捕获阶段传递，避免每个MCTS新节点扫描整段历史。训练/对弈API输出双方有效回合和计数，已停止的原训练任务及原分支均未改动。

规则版本升级为xiangqi-development-v3-natural-limit，配置指纹和回放版本随之变化，拒绝旧规则learner/replay静默恢复。此轮不改变131输入通道和8100行动编码；原无吃子输入平面仍为原始计数，双方将军次数尚未显式加入独立输入平面。这是后续规则观测升级的已知缺项，不能声称已实现完整可观测的国规模型。长捉/长杀/将捉、循环责任、早期和棋和512手截断等先前缺项保持待办。

本轮CPU回归100项通过、1项GPU检查跳过（34.53秒）；新增边界检查覆盖双方颜色、119/120半回合、将军10/11/15次、62回合13将不足60有效回合、双方均超额、对方下一轮符合条件、吃子兼将军清零、历史导入验证、MCTS复用隔离，以及旧规则checkpoint/replay拒绝静默混用。Web重放/悔棋逐步比对自然限着计数。前端语法检查通过；FastAPI测试客户端存在一项httpx弃用提示。原分支跟踪文件未改变，训练进程确认为已停止。

## 24. 显式规则迁移与继续训练（2026-10-16）

用户授权从规则分支续训、实际运行验证后合并到main。本节承接22、23节，之前关于训练已停止和原目录未修改的描述是当时状态。

新增 `AlphaZero.scripts.migrate_natural_limit`，限定已知v2规则到自然限着v3，模型、输入、行动及算法规格必须匹配。迁移持有源训练目录锁，防止在源learner运行期间操作；只读取源learner及SQLite快照，在新的临时目录生成并整体发布目标运行。保留模型、SGD动量、累计更新数、学习率日程和随机状态，生成新的run ID与规则指纹；旧目录留作回退。普通resume的严格版本检查继续保留。

迁移逐盘重放完整棋谱，核对每个输入的128个历史平面和3个全局平面、策略支持的合法性及终局价值。新规则提前终局或改变最终终局原因的整盘剔除；损坏输入、非法棋步、缺失棋谱或不一致的来源身份则中止迁移。保留原worker序号，避免新对局复用已用种子。**保留的策略标签仍是旧规则下搜索产生的访问分布，验证着法没有重新计算旧MCTS的反事实分支。** 来源规则、目标验证规则和源run ID写入棋谱与迁移报告，不能称为全部由新规则重新生成的数据。

实际从 `xiangqi-v3-20261015/learner.pt` 第81300步迁移到 `xiangqi-rules-20261016`：145盘、18517局面全部通过，零盘剔除，新run ID为12547b1e-9441-455f-bdb1-286e3cb2627a。初始actor参数与旧learner逐个张量完全相等。旧停止前81301—81324的未存档更新没有恢复，累计步数从真实存档81300继续。

全套CPU回归105项通过、1项GPU检查跳过（34.49秒），包括真实SGD动量/步数/随机状态保留、迁移后下一次更新、损坏数据拒绝、源进程锁和终局变化过滤；迁移记录调整后定向5项再次通过（4.03秒）。JavaScript与PowerShell语法检查通过，FastAPI客户端有一项弃用提示。真实CUDA验证使用原19×128网络、BF16、batch4096、每手800搜索、32盘/8进程。已观察32盘全部推进，搜索诊断每盘800次，恢复存档按100步保存且规则版本正确；损失和参数有限，无worker/inference错误。

真实验证已正常结束：81300→82000，共700次CUDA更新，最终loss为1.73229599，更新耗时中位数约0.631秒；learner及latest均保存为82000且参数逐项相等。现场记录32盘全部至少推进15手、搜索800次、errors为空。这段运行尚未完成新的整盘，回放仍是145盘、18517局面。精确记录见 `references/rule_training_acceptance.json`。本次合并目标为main，之后从82000存档在主目录继续长期训练；默认管理脚本已指向新运行目录。

本次验证不能替代完整国规验收或棋力评估。尚未实现的长捉、循环责任、完整规则观测平面及策略头架构等缺项见前文；145盘的数据规模与高重复采样仍限制模型质量。原8767人机进程仍读取旧训练目录，新训练目录须在该服务后续重启时显式指定。

## 25. 五模块轻量性能监控与短时、五小时观测（2026-10-17）

用户本轮要求先观测、短时运行后停止并分析，等待其评审再决定优化。实现基于main的ff23e9b，在独立分支 `codex/performance-monitoring`，不混入另一个工作树中尚未提交的 `alphazero_Policy` 改动。GitHub有写权限，按本轮要求提交独立PR，保留main供审阅。此次真实运行仍是131输入、8100全连接策略头的19×128模型，800次搜索、32盘/8进程、BF16、4096 batch及既有SGD配置。

### 25.1 实现与计时边界

全部CPU边界使用perf_counter墙钟。每个worker独立累计，推理进程独立累计，learner与采集线程共用带锁计数器；每5秒通过单独的有界队列发送累计快照。中间快照可丢，后续累计快照补回，不复用可能丢棋盘事件的显示队列，也不在每次模拟写日志。均值由总和/样本数得出。固定128桶的对数直方图合并所有来源后估计P95上界，通常桶宽约19%，不平均各worker的P95；极端溢出桶取最大值作为保守上界。

| 模块/指标 | 计算方法与边界 | 解读限制 |
|---|---|---|
| MCTS整手 | worker在select_move外层计时，包括根评估和完整800次搜索 | 只在整手成功完成后计入；未完成搜索不进入模拟速率分子 |
| MCTS同步神经调用 | 同一整手中evaluate_batch同步调用的墙钟和，含输入编码、RTT和结果转换 | 包含调用内部的CPU工作，不能视为纯GPU等待 |
| MCTS本地估算 | max(0,整手墙钟−同步神经调用墙钟) | 是墙钟差值，含调度/GIL等影响，不是精确进程CPU执行时间 |
| 推理等合批 | 取到首个请求后，到当前轮等待合批结束 | 不含空闲等待首请求；后者另列统计 |
| 推理数据准备 | 每个实际模型版本批次的数组拼接、pin_memory、传输提交与布局处理墙钟 | CUDA异步传输可能尚未完成，字段明确标为主机时间 |
| 推理服务 | 每个实际批次准备开始至结果D2H、有限性/归一化检查结束 | 不含合批等待与请求排队，不等于单个请求RTT |
| 推理执行墙钟 | forward开始至策略softmax、D2H、结果检查结束 | 包括GPU等待和回传，不只是提交CUDA任务的Python时间 |
| 实际平均Batch | 同一模型版本的真实输入局面数/实际执行批次数 | CUDA Graph可能按2的幂填充，另列含填充的执行Batch |
| 请求往返RTT | worker开始put请求至get返回结果 | 包含队列序列化、排队、合批、数据准备、GPU和回传，不能叫纯通信成本 |
| SGD采样 | replay.sample完整调用 | 包含SQLite访问、抽样、解码、稠密训练张量构造及锁等待 |
| SGD传输准备 | 采样后至pin_memory、to、channels_last提交结束的主机墙钟 | 异步DMA尾部可与后续主机提交重叠，不代表传输已完成 |
| SGD计算更新 | 传输提交后至前后向、SGD、既有有限性检查、统计取回结束 | 包含未完成异步工作的等待和校验；不是纯GPU核执行时间 |
| SGD完整更新 | Trainer.step开始至原统计值构造结束 | 其顺序墙钟阶段可在同一调用内分析占比，但不能与其他并发模块时间相加 |
| Replay写入 | add_game整盘校验、事务写入与提交 | 只在真实收到完整局时观测；未遇到不能伪造0ms均值 |
| Checkpoint | 原save_checkpoint的完整调用 | 包含参数CPU复制、序列化、fsync和原子替换 |
| 模型发布/载入 | actor导出校验与落盘；推理进程载入另列 | 二者可与其他模块并发，不相加当全系统停顿 |

CUDA每64次更新/实际推理批次抽样一次，采用enable_timing的Event，在原有传输、执行边界记录；结束事件放在原D2H或bool有限性检查之前，随后只在query确认完成时读取elapsed_time，毫秒换算秒。**监控不调用synchronize或增加等待事件完成的轮询**；未就绪样本丢弃并计数。推理CUDA Graph首次建图/预热另列，SGD首个抽样仍含首次预热；分析采用累计差分排除首60秒。[PyTorch 2.8官方Event文档](https://docs.pytorch.org/docs/2.8/generated/torch.cuda.Event.html)定义的elapsed_time是事件之间的流时间，包含共享GPU竞争与主机提交间隙，不能当独占GPU内核活跃时长或纯DMA带宽测量。

模拟数、推理局面数、完整局数和新增回放局面数均以本次会话墙钟为分母。模拟数来自已完成整手的diagnostics.simulations；推理还包括根评估，而终局模拟可不调用网络，且窗口存在未完成搜索，两种速率不必相等。小时产出只统计新提交整局，不把启动前的242局或36378局面计入本次速率。

### 25.2 本机约5分钟观测结果

从138800步存档恢复。外部watchdog于北京时间2026-10-17 09:34:59启动，300秒后对该pipeline发送SIGINT，约09:40:01完成退出，总计301.82秒；未更改learner的停止、保存频率或算法。最后完成139234步，最近完整存档139200步，指纹仍为e793151f623789e9a47d950f64b779d2ddc63ea611cae24769cc885e53d9fcba，checkpoint键集合与原存档相同。最后34次尚未到保存点的更新未保留，已提交回放仍为242局/36378局面。该次短时观测结束后训练停止；之后经用户另行授权的五小时续训记录见25.5。

精确数据见 `references/performance_observation_20261017.json`；原始5秒序列及watchdog记录留在run目录。下表是排除首60秒后约238.98秒的窗口，通过累计总和、次数差分计算，不是把整段均值相减。来源每5秒上报，边界可有一个周期的滞后；P95只保留整个会话的近似上界，不能冒充稳态窗口分位数。

| 指标 | 窗口结果 | 样本数/说明 |
|---|---:|---|
| 每手800次搜索 | 25.501秒 | 312个完成整手 |
| 同步神经调用/手 | 24.348秒 | 同一312手，约95.5%的整手墙钟 |
| 本地MCTS墙钟估算/手 | 1.153秒 | 同一312手，约4.5%；不是精确CPU时间 |
| 完成整手的模拟速率 | 1044.45次/秒 | 窗口249600次，排除未完成搜索 |
| 每次请求RTT | 29.753ms | 240611次，包含推理/排队 |
| 每轮等待合批 | 2.385ms | 28478轮 |
| 实际输入Batch均值 | 8.544 | 28478个模型版本批次；配置上限128未变 |
| 含Graph填充的执行Batch | 11.364 | 同批次，填充属于原有实现 |
| 推理数据准备（主机） | 0.526ms/批 | 28478批 |
| 推理执行与回传墙钟 | 3.272ms/批 | 28478批 |
| 每实际批次服务墙钟 | 3.810ms | 28478批，不含合批等待 |
| 推理CUDA执行流区间 | 1.081ms/批 | 445个暖态抽样，包含softmax等既有GPU后处理 |
| GPU推理输入速率 | 1018.12局面/秒 | 窗口243306局面，不含填充数量 |
| 每次SGD完整更新 | 638.612ms | 367次，约1.54次/秒 |
| Replay采样 | 178.955ms/次 | 367次，约占SGD墙钟28.0% |
| SGD传输准备（主机） | 27.076ms/次 | 367次，约4.2% |
| SGD计算与更新墙钟 | 432.257ms/次 | 367次，约67.7%，含校验与同步等待 |
| SGD CUDA传输/计算流区间 | 28.359 / 423.589ms | 各5个暖态抽样，样本少，不能视作精确总体分布 |
| Checkpoint保存 | 660.226ms/次 | 4次，累计2.641秒 |
| 暖态模型导出发布 | 315.897ms | 1次；启动时另一次2.760秒已排除 |
| 模型载入 | 569.419ms | 1次 |
| 新完整局与新增回放 | 0局 / 0局面 | 无Replay写入样本；无法评价长期每小时产出 |

硬件是3秒采样、在5秒快照中保留，稳态窗口49个样本：整卡GPU利用率均值70.4%，范围14—99%；显存约9.68GiB，范围9.48—10.07GiB；WSL CPU均值23.2%；WSL内存约7.08GiB，WSL可见总内存约15.2GiB。CPU/内存是WSL视角，GPU利用率是整卡，不能直接当作本进程纯推理占用率或Windows主机全部资源。此次monitor丢报、CUDA未就绪、输出错误计数均未出现。

这是恢复后首批32盘的短窗口：保存的现场快照均使用actor138800，期间没有整局结束，整局固定版本约定使其继续使用该版本。139000虽已发布并载入，尚未进入新局。因此本次实际推理尚未覆盖长期运行的多版本批次分裂，显存和Batch结果不能直接外推到多小时训练。

### 25.3 粗粒度瓶颈判断

自对弈每手的主要墙钟消耗位于**同步神经服务链路**：24.35/25.50秒。该范围包括请求前的输入构造、队列/排队、合批、GPU执行和返回转换；结果不支持优先把绝大部分整手延迟归于本地树搜索/规则计算。当前每请求近30ms，而每个实际批次服务约3.81ms、另等合批2.39ms，且输入Batch平均8.54，提示后续若继续定位，先看请求排队、序列化与共享GPU调度链路。不过这些指标是不同粒度和总体，**不能用29.75−3.81ms推出纯IPC成本**，也不能从中直接断言调小合批等待或增大并行度一定更快。

learner内部仍以模型计算和参数更新为大项，约432ms；Replay采样约179ms也已是明显成本，占自身更新约28%。这足以把后续调查聚焦到采样/解码与GPU计算两个粗模块，但本轮不继续到逐函数埋点。传输主机时间和CUDA流时间有异步重叠，不能与计算时间重复相加。GPU平均利用率约70%且波动很大，不等于存在固定30%的可直接回收加速，也不证明所有等待都来自GPU算力不足。

Checkpoint在该窗口累计2.64秒，约为窗口的1.1%，暖态发布一次约0.316秒，暂无证据表明它们是此次主要阻塞。Replay写入未发生，不能据此说写入快或磁盘没有瓶颈。32局每手约25秒，5分钟不足以完成较长棋局，0局/小时只是本窗口计数折算，**不是长期自对弈吞吐量为零**。这项五分钟观测止于上述分析，没有改变搜索次数、模型、规则、损失、并行度、合批等待、数据格式或任何优化参数；五小时续训复核见25.5。

### 25.4 开销、验证与文件

默认关闭：不创建监控队列、直方图、日志或CUDA Event，仅保留少量None分支和checkpoint包装调用。开启后每次请求/更新增加少量时钟读取和内存累计；每5秒JSON汇总，Event每64次抽样。`references/performance_overhead_check.py`对CPU learner做19×128、batch2的ABBA小对照，每轮预热3次、计12次更新：关闭约88.310ms、开启约88.868ms，中位数增加0.63%，四次最终参数SHA256完全相等，见 `references/performance_cpu_overhead.json`。该数值只覆盖CPU learner计时，不覆盖IPC、GPU Event与汇总I/O；没有完整GPU开关A/B，不能承诺整系统开销也是0.63%。任何观察都可能轻微影响异步排队时序，固定种子对照用于验证算法与随机调用保持，而非声称两次真实异步GPU运行逐局逐步必然一致。

全套CPU回归112项通过、1项GPU检查跳过（48.06秒），含7项新增Python检查：总和/次数/合并P95上界、累计快照丢报后恢复与过期快照拒绝、并行速率以墙钟计、精确SGD参数/动量/随机状态对照、固定种子800搜索棋谱和完整sample字节一致、RTT边界、CUDA Event低频且只读已完成事件、开启/关闭与旧checkpoint恢复。页面另有Node原生轻量检查，验证字段、空样本、开关隐藏过期数据；JS和PowerShell语法检查通过。FastAPI原测试有一项httpx弃用提示。真实本机BF16+CUDA Graph推理及SGD的Event路径已获得有效样本；未额外验证torch.compile、其他精度/设备或完整GPU开销A/B。内置浏览器策略阻止读取已有的错误页面，未绕过该限制；本机HTTP确认8765实际返回新增区域，脚本渲染检查通过，真实浏览器截图未验收。

最后补充完整NumPy随机状态断言并加入模型载入计时后，定向7项Python检查再次通过（17.42秒），Node页面检查1项通过。CPU对照中0.63%的差异与短测波动同量级，不宣称其具有统计显著性。

开启：`./scripts/training.ps1 -Action Resume -Performance`，关闭：不带Performance或WSL使用 `--no-performance`。直接CLI启用 `--performance`；可用 `--performance-interval 5`、`--performance-gpu-every 64`调整观测频率。它们都不是AlphaZeroConfig字段，不改变spec/fingerprint和checkpoint键集合；本次原存档已直接恢复验证。摘要在run内的performance.jsonl及performance.json中，关闭后的新会话不覆盖旧性能文件，按session_id区分。外部限时运行器现允许1至18000秒，并以UTC时间戳分别创建stdout、stderr和watchdog记录，避免后续观测覆盖先前日志。

改动文件：train/performance.py负责累计、合并和低频CUDA事件；self_play_worker.py负责整手/往返边界；inference_server.py负责推理；trainer.py负责SGD；scripts/pipeline.py负责存储计时、汇总及网页数据；frontend/training.html/js/css负责同风格的六个简明区域与可展开明细；scripts/training.ps1负责开关；tests/test_performance.py及test/training_performance_ui.test.cjs负责轻量验证；references目录保留CPU对照、外部限时运行器、分析脚本和实测JSON。唯一审查文档仍为本文件，README仅补充使用方法。

### 25.5 用户授权五小时续训复核

用户在短时观测后授权继续训练五小时，再更新本报告并提交PR。本次从 `learner.pt` 第139200步恢复，仍为同一run ID和规则/模型指纹；配置保持19×128、131输入、800次搜索、32盘/8 actor进程、BF16、batch4096、CUDA Graph及原有SGD参数。没有修改算法、参数、随机数消费、Replay或checkpoint格式。五小时限时运行器另存时间戳日志，不覆盖前述五分钟记录。

watchdog于北京时间2026-10-17 10:07:48.277启动，设定18000秒后发送SIGINT；pipeline于15:07:50.802退出，墙钟18002.525秒，仅比目标晚2.525秒。性能计时器记录17999.565秒。停止走既有KeyboardInterrupt清理路径，未触发强制终止；训练进程已退出。stderr中的异常栈止于预期SIGINT打断 `trainer.step`，没有发现CUDA OOM、CUDA运行错误或worker异常。最终完成第168652步，最近完整learner存档为168600步，latest actor为168000步；最后52次SGD更新尚未到恢复存档点，没有写入checkpoint。指纹仍为 `e793151f623789e9a47d950f64b779d2ddc63ea611cae24769cc885e53d9fcba`，checkpoint键集合保持原样。没有重新启动训练。

Replay本次新增49盘、6994局面，总量从242盘/36378局面增至291盘/43372局面。分析排除计时开始后的首60秒，通过累计计数差分得到17934.535秒稳态窗口（第139273至168652步）；每5秒报告边界可能滞后一周期。完整计数、直方图P95近似值、硬件采样和watchdog记录见 `references/performance_observation_20261017_5h.json` 及run目录下 `performance-run-20261017T020748Z-*`。本节表格采用稳态窗口均值；P95明确为全会话固定直方图上界近似值。

| 指标 | 五小时运行稳态结果 | 样本数/边界 |
|---|---:|---|
| 每手800次搜索 | 56.301秒 | 10172手；全会话近似P95上界87.284秒 |
| 同步神经调用/手 | 55.132秒（约97.9%） | 与上述同一批10172手 |
| 本地MCTS墙钟估算/手 | 1.169秒（约2.1%） | 墙钟差值，不是精确CPU执行时间 |
| 完成整手模拟速率 | 453.74次/秒 | 8137600次完成模拟/窗口墙钟 |
| 请求往返RTT | 69.075ms | 8061946次；全会话近似P95上界210.225ms，含排队与推理 |
| 等待合批 | 2.405ms/批 | 1015510批 |
| 实际输入Batch均值 | 1.964 | 4106003个模型版本批次；最大32 |
| 含Graph填充执行Batch均值 | 2.278 | 同一批次集合 |
| 推理数据准备（主机墙钟） | 0.663ms/批 | 4106003批 |
| 推理执行墙钟 | 2.554ms/批 | 4106003批，含既有等待及回传路径 |
| 推理服务墙钟 | 3.228ms/批 | 4106003批，不含合批等待 |
| 推理CUDA执行流区间 | 0.973ms/批 | 64254个低频抽样；不是独占GPU核时间 |
| 推理局面速率 | 449.66局面/秒 | 8064406个真实输入，不含Graph填充 |
| SGD完整更新 | 603.498ms/次 | 29379次；全会话近似P95上界707.107ms |
| Replay采样 | 166.465ms/次（约27.6%） | 与完整更新同一时段 |
| SGD传输准备（主机墙钟） | 25.466ms/次（约4.2%） | 异步DMA完成可能跨越阶段边界 |
| SGD计算与更新墙钟 | 411.259ms/次（约68.1%） | 含同步等待与原有有限性校验 |
| SGD CUDA传输/计算流区间 | 28.448 / 406.298ms | 各459个暖态抽样 |
| Checkpoint保存 | 535.379ms/次，累计157.401秒 | 294次保存 |
| Replay写入 | 86.158ms/局，累计4.222秒 | 49次完整局事务 |
| 模型导出发布/actor载入 | 281.722 / 478.989ms | 窗口各29次；跨版本运行已覆盖 |
| 新增局面/完整局速率 | 1403.91局面/小时；9.836局/小时 | 窗口新增6994局面、49局 |

硬件在稳态窗口共有3576个采样：整卡GPU利用率均值69.24%（0—99%），显存均值10.848GiB、峰值11.468GiB/总计11.940GiB；WSL CPU均值17.50%（14.6—25.3%），WSL内存均值7.727GiB、峰值8.114GiB，可见总量15.199GiB。运行期间未见OOM，但采样峰值距可见显存上限约0.47GiB；低频硬件采样可能漏过短暂峰值。CPU和内存是WSL口径，GPU利用率是整卡口径。

### 25.6 五小时结果与瓶颈判断

五分钟窗口的输入Batch均值8.544、每手25.501秒、推理速率1018.12局面/秒；五小时窗口分别为1.964、56.301秒和449.66局面/秒。前次短时现场是32个对局都固定使用actor138800，而五小时窗口跨过多个模型发布周期，稳态内记录29次模型发布和29次actor加载，实际推理按模型版本分组。**这与Batch缩小、推理速率下降、整手时间增加同时出现，支持“跨版本分组后批次利用率较低”作为主要待查方向；但这不是受控因果对照，不能断言版本分组是唯一原因，也不据此改并行度、等待时间或模型。**

整手墙钟约97.9%落在同步神经调用期间；每次请求平均RTT约69ms，实际每版本批次的服务墙钟约3.23ms，另测等待合批约2.41ms。这里RTT为请求级指标，服务与合批为批次级指标，样本总体和边界不同，**不可相减并称为纯IPC耗时**。数据表明较大部分时间处于服务批次计时之外的请求往返链路，但现有粗粒度计时不能进一步区分worker等待、队列/序列化、调度和返回转换。GPU推理事件每批约0.97ms，不能单独解释每手约55秒的同步等待；整卡约69%的利用率同时包含4096 batch的SGD与推理竞争。

SGD完整更新约604ms，其中采样约166ms、计算与更新约411ms；这是learner内部清楚可见的第二类成本。Replay写入49次的累计墙钟约4.22秒，Checkpoint共约157秒，模型发布约8.17秒；就这段运行的数据而言，存储调用不是主要耗时来源。并行模块间仍存在重叠，不将模块墙钟求和当成训练总时长。硬件均值没有显示CPU饱和，但GPU显存接近容量上沿；未见OOM仅说明该次设置在这五小时内完成，不代表其他并发/版本组合有同样余量。

该实测把优先调查方向从五分钟时“同步推理链路”收窄为**请求往返链路与跨模型版本的小批次现象**，然后是Replay采样和SGD计算；它尚不能识别请求链路内的单一函数原因，也没有做任何加速、参数扫描或算法更改。训练已按用户要求停止，报告更新后推送到原有PR，不合并。

## 26. 用户指定400搜索、32000步发布的继续训练（2026-10-17）

用户随后明确要求把模型发布间隔从1000次SGD更新改为32000次，并将搜索次数由最初口述的4纠正为400。本次实际启动前已落实最终的400次配置，没有启动4次搜索的真实续训。改动位于独立分支 `codex/training-search400-publish32000`，基于性能监控分支b39db5d；此前五小时结果属于800搜索/1000步发布的历史配置，不能与本节混用。

`config/local-train.yaml` 设置 `num_simulations=400`、`checkpoint_interval=32000`。解除恢复时对发布间隔的相等限制，仍保留模型/规则指纹及其他受保护训练设置的检查；恢复时按新间隔重新计算辅助发布轮数，优化器累计步数仍是学习率和训练进度的权威时钟。每100步的 `recovery_interval` 不变，700000次总更新上限仍生效。发布沿用累计步数取模，因此从168600恢复后的下一次定期发布为192000步，再下一次224000步；启动时另按既有流程导出168600步快照供新局使用。

从原目录 `xiangqi-rules-20261016/learner.pt` 第168600步继续，run ID保持 `12547b1e-9441-455f-bdb1-286e3cb2627a`，规则/模型指纹保持 `e793151f623789e9a47d950f64b779d2ddc63ea611cae24769cc885e53d9fcba`。启动前用原19×128存档逐项检查模型张量和SGD动量恢复一致； `metrics.jsonl` 新会话 `73742ee104b247de9f304e4ef10625a6` 记录800→400和1000→32000的配置变化。原有800搜索Replay保留，新生成完整局继续以trace记录400搜索，属于用户授权的混合历史数据续训。

15项定向CPU测试通过（19.76秒）。覆盖400/32000配置恢复时的权重、动量、步数、RNG及变更记录；小网络实际管道从旧发布间隔恢复到新间隔，检查只在初始恢复快照及新累计步数边界发布，同时恢复存档继续生效。为短程测试使用2→4步发布边界，没有把它误称为已经跑满真实32000步。其他学习率或模型规格不兼容仍拒绝恢复。

真实CUDA续训已观察到第168766步，32个对局全部至少到第9手，每盘最近一次搜索诊断均为400，页面错误列表为空；本次stderr只有WSL代理提示，没有CUDA/worker错误。已实际写出的168700步 `learner.pt` 带400搜索、32000发布间隔和100步恢复频率，原run ID与指纹一致。现场记录见run目录的 `continuation-400-32000-snapshot.json` 与 `training-20261017-163146*`。核查时本会话尚无新完整局完成，也未到192000步的下一次真实发布点；短程恢复与运行验证不能证明长期吞吐或棋力改善。训练继续运行，性能监控仍启用。

## 27. 用户调整为1500步发布的继续训练（2026-10-17）

用户随后要求暂停训练，将发布间隔改为1500次SGD更新后继续。等待原进程写出新的恢复检查点后，对核实过命令行和工作目录的原训练进程发送SIGINT；停止后的最后一条SGD记录和 `learner.pt` 均为192700步，本次切换没有丢失已完成的优化更新。原进程已退出，随后从同一存档恢复，保留模型、SGD状态和原Replay；`config/local-train.yaml` 仅将本次发布间隔由32000改为1500，每手400次搜索及每100步恢复存档的设置继续使用。

新会话 `3bb4c1105307470ca1787def2bf19b4e` 的 `session_start` 记录唯一配置变化为 `checkpoint_interval: 32000 → 1500`，run ID及规则/模型指纹与上节相同。启动仍按既有逻辑发布192700步快照，后续按全局累计步数取模，下一次定期发布为193500步，此后195000、196500步。真实CUDA续训核查至192709步，8765快照显示32个对局均已完成首手、配置为400搜索/1500步发布，错误列表为空，新stderr为空。本轮仅调整运行配置，未修改训练算法，也未等待跑到193500步验证下一次真实发布；控制记录见run目录 `continuation-400-1500-control.json`，运行日志为 `training-400-1500-20261017-203727*`。

## 28. 用户调整为8000步发布的继续训练（2026-10-17）

按用户要求等待恢复存档原子写入完成，对核实过身份的原训练进程发送SIGINT并等待退出。最后完成的SGD记录和完整 `learner.pt` 均为206200步，随后将本地配置的发布间隔由1500改为8000，从同一存档继续训练；400次搜索、100步恢复存档及其余训练设置继续使用。累计步数发布边界为208000、216000步，启动时另按原有逻辑导出206200步actor快照。

新会话 `f08121ee6d36460cbe35a382780a276e` 已实际运行到206207步，run ID保持 `12547b1e-9441-455f-bdb1-286e3cb2627a`。8765状态显示32盘均已完成首手、发布间隔8000、搜索400、错误列表为空，新stderr为空。本轮为配置调整及真实恢复运行核查，未等待208000步定期发布，也未新增算法修改或性能实验。现场证据为run目录下 `continuation-400-8000-control.json`、`continuation-400-8000-snapshot.json` 和 `training-400-8000-20261017-225830*`。

## 29. 用户指定最近10万局面的Replay与继续训练（2026-10-18）

用户在272100步暂停训练后要求把训练局面上限改为10万，再继续训练。此前 `replay_window_games=1000000` 按盘保留，实际1106盘/194192局面均参与采样。本次新增 `replay_window_positions`，默认0表示不增加局面限制；本地配置设为100000，按入库顺序从最旧完整棋局开始淘汰，保留连续后缀且总局面数不超过上限，不拆分边界棋局。启动时先验证checkpoint/run一致性，再执行限额；每次提交新完整局后也在同一事务中执行限额。采样继续在当前保留局面中均匀、有放回抽取4096个样本，不按棋力筛选、不对最新局面额外加权。这是用户指定的数据保留策略调整，与默认按最近100万盘保留的基线不同，不据此声称棋力已经提高。

实际淘汰594盘/94470局面，保留第595至1106盘共512盘/99722局面，位置ID为94471至194192且连续。累计生成局面数、全局棋局generation、worker后续sequence及run ID不因淘汰归零；恢复存档的引用和采样范围仍一致。运行前通过SQLite backup保存一份非活跃备份 `replay-before-position-cap-20261018-154835.sqlite`，该备份不参与训练。完整模型张量和SGD状态逐项核对恢复相等，初次从272100步继续，400次搜索、8000步发布、100步恢复存档沿用当前配置；下一次定期发布为280000步。

28项定向CPU检查通过（18.56秒），覆盖整盘边界淘汰、已有Replay缩容、保持连续ID及恢复元数据、超长单局拒绝、旧checkpoint缺少新字段时恢复模型/动量/RNG、无效参数、既有搜索预算与发布间隔恢复、跨run恢复拒绝等。为兼容仍在运行的8767旧进程，actor导出省略仅用于learner的新保留参数，learner存档仍完整保存该参数；扩展后的恢复/导出兼容用例单独再次通过（2.66秒）。此修正需要重载训练进程，因此在272300步新的完整恢复点停止后再次续训，最后完成更新与存档步数一致；未重启对弈进程，原棋局保持第36手、模型272000步。

最终会话 `46eab06fc67d4fd0b567a8630d4cc16a` 已在CUDA上实际推进至272329步，32盘均完成首手，Replay仍为99722局面，8765错误列表和新stderr均为空；发布的272300步actor已检查能被旧配置字段集合读取。现场记录为run目录的 `continuation-position-cap100000-control.json`、`continuation-position-cap100000-snapshot.json` 及两段 `training-position-cap100000-20261018-*` 日志。本轮核查尚未等到新完整局入库，持续淘汰路径由定向测试覆盖，不宣称已经验收长期效果。

## 30. 显存零值、GPU采样失败与推理超时退出排查（2026-10-19）

用户先询问偶发显存归零与unavailable，再要求暂停训练排查。前一次现场读取时训练仍在第368923步运行、错误列表为空，直接nvidia-smi查询得到11266MiB显存和98%利用率，但8765同时返回gpu=unavailable且缺少显存字段；四次采用2秒超时的查询中前两次实际触发TimeoutExpired，随后两次成功。此次暂停操作前检查发现原PID131541已自行退出，未发送停止信号；stderr和最终性能文件最后写于北京时间09:50:33附近。不能将这次异常退出写成用户请求触发的正常SIGINT停止。

已确认的三个问题：

1. **硬件采样失败被误显示为显存零。** `train/live.py`的硬件线程使用2秒nvidia-smi超时，捕获异常后用仅含gpu=unavailable的新字典覆盖整份硬件快照，不保留最后一次有效GPU数据，也不记录失败类型或采样时间。`frontend/training.js`将缺失的显存字段用0兜底，因此显示0/0。缺失值不是实际测得的0。故障退出前末段791份性能快照有790份标记unavailable，不能将该末段视为连续可信的GPU测量。
2. **推理冷启动路径的长停顿触发worker超时。** `self_play_worker.py`把一次请求的等待上限固定为120秒；stderr明确记录TimeoutError("neural inference timed out")。同一会话的首次CUDA Graph建立调用最大墙钟时间先后升至15.559秒、28.340秒，最终达到136.178秒；最终一次推理服务调用最大值也为136.178秒。末段快照中推理完成批次计数从elapsed=64504.761至最终64659.439之间停滞后才增加，窗口上界约154.678秒，期间learner步数仍从376164推进至376392。该冷启动指标包含预热、建图、执行、D2H及检查，尚不能把136秒全部归因于图捕获内部。另在368000步发布后测得actor载入58.736秒。现有记录支持推理服务串行请求路径出现长停顿并超过等待预算，尚不能进一步判定驱动调度、显存驻留/分页、分配或某个CUDA调用为根因；未见进程日志明确报告CUDA OOM，不能宣称已经爆显存。
3. **主进程丢失具体错误来源。** worker异常会设置共享stop事件；collector检查`failed or stop.is_set()`时，子进程可能尚未完成退出，`failed`仍为空，最终抛出actor/inference process failed: []。这掩盖了已在worker日志中记录的实际超时。应保留可靠的错误原因和进程身份，区分主动停止、worker失败与推理失败。

最终性能会话步数为376392（104092次更新，加上恢复起点272300），最后一条已落盘SGD指标为376391；完整learner存档停在376300，故376300之后的92次已完成更新没有进入恢复存档。已在CPU上实际加载该learner，检查全部模型状态和SGD状态张量有限、Replay run/generation引用一致，可从376300继续；Replay累计完成1687盘，活跃窗口为99991局面。暂停后四次nvidia-smi查询分别约0.648/0.038/0.049/0.210秒成功，显存约1.51GiB；8767人机进程仍在，训练进程不存在。

08:00至11:50的Windows System事件查询没有找到42/107/4101匹配项；WSL内核末尾可见dxg ioctl -22，但不能据此断言发生驱动重置或确定上述长停顿的原因。本轮只进行离线日志、计时序列、状态读取和存档检查，未恢复训练或执行GPU压力复现。修复顺序应先纠正缺失数据显示、记录GPU采样失败详情与数据新鲜度、传递可靠的worker/inference错误，再给冷启动调用补充有限的阶段诊断并验证长停顿原因；单纯延长120秒等待不能算根因已经修复。运行状态记录为 `training-paused-for-gpu-monitor-debug.json`，主要故障证据为 `training-position-cap100000-20261018-155252.stderr.log` 及该会话的performance.jsonl。

## 31. 推理容错、独立阶段诊断与实际续训（2026-10-19）

用户随后授权设计解决方案并继续训练。本轮先修正已确认的监控/错误传递缺陷，再恢复原存档，通过运行中的阶段证据继续定位长停顿。每手400搜索、每8000步发布、每100步恢复存档、最近不超过100000局面的Replay及模型/损失/SGD参数沿用当前配置；没有将尚未确定的底层GPU长停顿写成已经根治。

### 31.1 监控与容错实现

- `train/live.py`将GPU查询与CPU/内存刷新分开：GPU查询由单个独立daemon线程执行，前一任务仍存活时不启动第二个查询，CPU/内存继续每3秒更新。nvidia-smi的普通超时从2秒改为5秒，记录成功时间、耗时、失败类型及连续失败次数；失败保留最后有效值，无历史值则留空。快照在读取端动态计算数据年龄，超过15秒即标记过期，即使查询线程尚未返回也生效；可在Linux上记录查询子进程PID、状态和wchan。
- 网页将缺失显存显示为暂未读取，缓存值明确标注上次读数，显示采样失败/过期及距离上次更新时间。GPU采样失效时仍能看到训练/推理进度，以及两个进程各自的CUDA内存池保留量；后者不是整卡总显存，也不包含驱动等全部开销。
- 新增 `train/runtime_diagnostics.py` 的共享推理阶段记录及父进程观测线程。阶段覆盖模型读取、创建/载入权重、输入准备、网络执行、图缓冲分配、图预热、图捕获、结果回传、发送与空闲；共享锁仅保护少量CPU字段，不跨CUDA调用持锁。每5秒追加 `inference-diagnostics.jsonl`，带session/PID、阶段年龄、完成轮次、版本/批量及缓存的推理分配器读数。独立观测可在推理进程阻塞时读取阶段，15秒起提示停顿；阶段结束超过5秒也打印慢阶段记录。模型载入/图冷启动各阶段另累计CPU墙钟到performance指标。
- CUDA Graph阶段回调位于原有计算之间，监控没有新增CUDA同步点或消费训练RNG；阶段墙钟包含异步提交、原有等待和调度，不能当独占GPU内核耗时。推理分配/保留内存只在模型载入和冷调用等时机刷新，为缓存的进程内分配器读数；learner每10步读取一次自己的分配器统计。
- worker和推理进程通过独立错误队列回报PID、异常、完整Python栈和请求/模型信息，父进程落盘 `runtime-failures.jsonl`，并在主错误中保留真实原因；进程突然退出而没有报告时也记录各进程PID/exitcode和共享停止状态，避免只有空列表。
- 请求等待上限由120秒调整为默认300秒，可通过CLI `--inference-timeout`指定；这是运行容错边界，不写入模型规格或checkpoint配置。它给此前136秒调用留出恢复余量，但超过边界仍失败退出，不无限等待、跳过模拟或用伪造网络结果继续搜索。`--runtime-diagnostics`默认关闭，本次明确开启；关闭后不创建阶段共享状态、独立阶段线程或阶段日志。
- `references/analyze_performance.py`排除过期/不可用及显式年龄超过15秒的GPU样本，CPU/内存仍使用各自有效记录，避免保留旧GPU值后被统计成连续新测量。历史没有新状态字段的有效样本仍兼容。

### 31.2 新捕捉到的GPU管理查询阻塞

第一段真实续训中，训练和推理持续推进，但Linux nvidia-smi子进程337708进入D状态，持续数分钟；原设5秒超时的一次采样实际581.108秒才返回TimeoutExpired。这表明仅修改Python超时不能保证底层不可中断等待立即返回。另发起的一次Windows原生nvidia-smi诊断查询也长时间未返回，已核实其PID43244/创建时间/命令后仅终止该诊断进程，没有重置驱动或停止其他GPU应用。

隔离补丁实际载入后，读数过期判断保持更新，捕捉到新采样子进程338522为 `State: D (disk sleep)`，wchan为 `dxgvmb_send_sync_msg`；同时SGD和推理轮次继续增加、CPU统计继续更新。可以把此次采样卡顿收窄到WSL/Windows GPU通道的同步等待位置，不能把D状态中的“disk sleep”直接解释成磁盘损坏，也不能据此断言显卡损坏。该采样现象与上一节136秒推理调用是否同一底层原因仍待证据；目前没有把它们合并成已证实的因果链。

后续按已捕捉阶段分流排查：模型读取长停顿核对文件读取；模型创建/图分配长停顿核对分配器及驻留版本；预热/捕获长停顿核对同步等待与并发时序；回传长停顿核对执行完成和传输等待。只有定位阶段后再做对应的短对照；本轮不据猜测关闭Graph、减少搜索或改训练目标。

### 31.3 验证与运行状态

首轮43项CPU检查通过（37.17秒），覆盖监控采样失败/恢复、无有效值、共享阶段在子进程阻塞时可读、固定边界超时、worker错误可靠传递、实际CPU管道、原有分组/固定种子样本与SGD兼容、旧存档与Replay等。补充分阶段聚合后18项检查通过（10.96秒）；过期GPU统计过滤用例单独通过（1.69秒）。最终采样隔离、数据动态过期、CPU继续刷新且不堆积查询任务等10项检查全部通过（234.94秒，期间真实训练并发运行）。页面Node轻量检查再次通过，覆盖缺失显存、缓存读数过期和停顿告警。多轮用例有重叠，不能相加冒称独立用例总数。真实CUDA恢复运行验证了Graph回调和诊断输出，未另行运行长时间GPU压力测试。

北京时间19:39:06先从376300步恢复，会话 `5ab72851db66470f99a0e38367ca2892`。真实初段采集到6次图分配/预热/捕获：阶段最大墙钟分别约0.153/0.566/0.154秒；这是短时冷启动观测，不代表已排除17小时后再发生长停顿。为载入采样隔离补丁，在377600步新的完整恢复点请求SIGINT并等待退出，最后指标与存档同为377600步，切换没有丢失已完成更新。

19:54:22从377600步再次继续，当前会话 `5981fcc2512241ac9391aed9ab8ab8f4`，PID338192，run ID保持 `12547b1e-9441-455f-bdb1-286e3cb2627a`。核查已推进至377821步，32盘运行、活跃Replay99662局面、错误列表和新stderr为空，推理未停顿；GPU管理查询虽处于D状态，页面/性能快照正确标为stale。learner内存池保留约8.037GiB，推理缓存保留约0.299GiB，不能相加冒充实测整卡显存。下一次定期发布仍为384000步。

CLI使用 `--performance --runtime-diagnostics --inference-timeout 300`；控制与快照留在 `continuation-inference-diagnostics-control.json` / `continuation-inference-diagnostics-snapshot.json`，两段日志为 `training-inference-diagnostics-20261019-193906*` 和 `training-inference-diagnostics-20261019-195422*`。8767进程和正在进行的棋局保留。另创建每30分钟复查的当前聊天定时监控，仅在新推理停顿、SGD长期不推进、异常退出或正常完成时提醒；健康运行或已知GPU采样过期且计算仍推进时保持安静，用户暂停后不得自行恢复训练。

### 31.4 定时监控捕捉到新模型图预热长停顿（2026-10-19 22:03复查）

本次核对了控制文件中最新的 sampler_isolation_reload、process.pid、进程命令/启动时间/工作目录及推理父子关系：仍为19:54:22恢复的会话 `5981fcc2512241ac9391aed9ab8ab8f4`，pipeline PID338192、推理PID338230、run ID `12547b1e-9441-455f-bdb1-286e3cb2627a`，没有把较早PID337367误当当前进程。配置仍为400搜索、8000步发布、最多100000个Replay局面、请求超时300秒、阶段告警15秒。下述时间均为北京时间（UTC+8）。

- 384000步模型已经发布。21:34:22.338左右，模型384000、实际batch1首次进入 graph_warmup；至21:35:07.173结束，阶段墙钟 **44.835秒**。5秒诊断记录在21:34:52.732、21:34:57.733、21:35:02.733分别捕捉30.394、35.395、40.395秒，完成批次固定为799185。随后 graph_capture 另耗9.982秒，21:35:17.155结束；21:35:17.735采样已完成799188批，确认恢复，不能把两个相邻阶段合称44.8秒或把阶段墙钟当纯GPU计算时间。
- 停顿时记录两个驻留模型版本；推理分配器缓存为allocated约0.1135GiB、reserved约0.3262GiB。邻近性能快照显示SGD从384223继续推进到384327，并未随推理一起停止；同期GPU管理查询PID341832处于D状态，wchan为 dxgvmb_send_sync_msg。该共现不足以证明查询导致预热停顿，也不能用过期的整卡显存缓存判断当时是否逼近容量。
- 同一新模型后续还记录batch8预热22.190秒/捕获16.677秒，以及batch16预热25.422秒/捕获8.361秒；各单阶段均未达到本次30秒提醒阈值。更早的输入prepare停顿27.414秒也已恢复。本次新证据将至少一次长停顿定位到图预热阶段，但仍未区分阶段内部等待、驱动调度、显存驻留/分配和SGD竞争，不能宣称根因已修复或已确定。
- 22:02:58附近现场已到387044步、推理936717批，错误列表为空，runtime-failures.jsonl和本会话stderr均为0字节；发布actor为384000，活跃Replay99923局面。learner缓存reserved约8.037GiB，推理缓存allocated约0.1554GiB/reserved约0.5313GiB，均不是整卡实测值。GPU管理查询仍可能在D状态等待，计算持续推进。
- 后续先读取当时原子落盘learner.pt的ZIP元数据：第387200步、58203713字节、包含同一run ID；该文件观察到的修改时间为22:04:37.362。随后另一次CPU加载成功读取了继续更新后的第387300步learner，run ID一致，含模型、SGD和随机状态等原有键，是本轮最后确认的恢复检查点；运行中仍每100步覆盖更新。本轮未逐项检查该新文件的全部张量有限性，也未实际恢复该存档，不将成功加载冒称完整恢复验收。

证据保存在同一run目录的 inference-diagnostics.jsonl、performance.jsonl、training-inference-diagnostics-20261019-195422.log 及对应空stderr/runtime-failures日志。本轮仅记录证据、更新本报告和监控状态，没有停止/重启训练、改变算法或参数，也没有主动发起GPU管理查询、GPU负载或提交推送。

### 31.5 batch32图预热再次出现已恢复长停顿（2026-10-19 22:33复查）

核对最新控制文件、PID/启动时间/命令和工作目录后，仍是31.4节同一run、会话及pipeline338192/推理338230。新事件发生在北京时间22:08:15.663至22:08:57.777：模型384000、实际batch32的 graph_warmup 阶段墙钟42.114秒，期间完成批次固定963816；其后 graph_capture 另耗13.926秒，于22:09:11.703结束。22:09:13.144诊断已有963940批，确认恢复。该事件与上次batch1预热44.835秒是两次不同事件，表明长预热至少涉及不同批量形状；仍不能由此确定阶段内部的底层原因。

停顿时有两个驻留版本，推理分配器缓存allocated约0.1558GiB、reserved约0.5313GiB。22:33现场SGD到390000步、推理1089489批，错误列表为空，本会话stderr及runtime-failures.jsonl仍为0字节；Replay99825局面，发布actor384000。learner保留约8.037GiB，推理保留约0.5742GiB。同期最新快照中的GPU管理查询PID341888仍在D状态、wchan为 dxgvmb_send_sync_msg，过期整卡读数未用于容量或故障判断，不能据此认定查询导致预热停顿。

本轮读取的learner.pt原子存档ZIP元数据为390000步、58203713字节，含同一run ID；这是较新的元数据确认，上一轮实际CPU加载成功的是387300步。本轮没有再次进行完整张量加载/有限性或恢复验收。原始证据仍为31.4节所列同一组日志与诊断文件。仅更新本报告与监控状态，配置保持400搜索/8000步发布/100000局面上限/300秒超时，没有修改代码、重启进程、主动发起GPU查询或提交推送。

### 31.6 392000步模型创建与图预热均出现长停顿（2026-10-19 23:03复查）

当前控制文件的最新恢复项、process.pid和进程启动时间/命令/工作目录仍匹配31.4节同一会话，pipeline338192、推理338230及run身份未变。392000步actor已发布。此次新增两次达到30秒阈值且已经恢复的事件，时间均为北京时间：

| 模型/阶段 | 开始至结束 | 阶段墙钟 | 停顿时完成批次及驻留版本 | 推理allocated/reserved缓存 |
|---|---|---:|---|---|
| 392000 / model_create | 22:54:30.966—22:55:13.006 | 42.040秒 | 1214843批，2版本 | 约0.1656/0.5742GiB |
| 392000 / graph_warmup，batch1 | 22:59:26.593—23:00:05.283 | 38.691秒 | 1241430批，3版本 | 约0.1919/0.6191GiB |

第二次事件后 graph_capture 另耗13.125秒，于23:00:18.408结束；后续推理计数已增加，确认恢复。model_create计时包住 create_model(config, config.device) 整个调用，尚未在其内部细分CPU创建、设备迁移或等待。**新证据说明长停顿不局限于图预热/捕获；不能把本会话所有长延迟都解释成CUDA Graph捕获。** 表中内存是阶段记录保留的进程分配器缓存，创建阶段尚未完成后的新版本不包含在当时2个驻留版本中，不能用该缓存替代瞬时峰值。

23:03现场SGD为392897步、推理1254815批，发布actor392000，Replay99814局面，错误列表为空，本会话stderr及runtime-failures.jsonl仍为空。learner保留约8.037GiB，推理allocated约0.2247GiB、reserved约0.7813GiB，驻留版本3个。最新快照记录GPU管理查询PID342055仍处于D状态、wchan为 dxgvmb_send_sync_msg；这是查询状态，不能据此确定上述计算阶段停顿的根因或显卡故障。

本轮读取的learner.pt ZIP元数据为392900步、58203713字节，含同一run ID；未再次做完整张量加载或恢复验收，最近一次CPU加载成功的387300步验证属于先前记录。证据为同一run的当前stdout、空stderr、runtime-failures.jsonl、inference-diagnostics.jsonl及performance快照；没有停止/重启训练、改代码或参数、主动发起GPU查询/负载或提交推送。持续记录阶段与版本变化，底层原因仍未确定。

### 31.7 batch32预热长停顿达到138秒后恢复（2026-10-19 23:33复查）

核实控制文件最新恢复项、process.pid及实际进程命令/启动时间/工作目录后，仍为会话 `5981fcc2512241ac9391aed9ab8ab8f4`、run `12547b1e-9441-455f-bdb1-286e3cb2627a`、pipeline338192及其推理子进程338230。本次新事件：北京时间23:29:30.450至23:31:48.599，392000步模型、实际batch32的 graph_warmup 阶段墙钟 **138.149秒**；5秒诊断采样中完成批次持续固定1383833。随后 graph_capture 另耗7.881秒，23:31:56.480结束；23:32:00.375诊断已有1384154批，确认恢复。本次阶段本身已超过旧120秒请求等待预算；现行300秒预算容纳了此次延迟且未报worker超时，但延长预算未消除停顿，不保证下一次仍能恢复。

该阶段记录2个驻留版本，推理allocated/reserved缓存约0.2034/0.6973GiB。事件窗口内27份performance快照的SGD步数从395442增加到395689，推理停止出结果期间learner仍在更新。相应GPU管理查询PID342124处于D状态、wchan为 dxgvmb_send_sync_msg；这仍只是共现证据，未证实查询与预热停顿的因果关系，过期整卡显存值不用于判断是否OOM。

23:33现场SGD395878步、推理1392734批，发布actor392000，Replay99831局面；错误列表、本会话stderr和runtime-failures.jsonl均无新增异常。learner缓存reserved约8.037GiB；推理allocated约0.2133GiB、reserved约0.7402GiB，当前已不处于停顿。新的learner.pt ZIP元数据为395900步、58203777字节、同一run ID；本轮未做完整张量加载或实际恢复验收，先前成功CPU加载387300步的记录保持其原有边界。

本次将观测到的慢阶段范围从约40秒扩大到138秒，不据此判断显卡损坏、显存耗尽或确定具体CUDA调用。原始证据仍为同一run的当前stdout、inference-diagnostics.jsonl及performance.jsonl，报告仅追加本次事实并更新监控状态。400搜索、8000步发布、100000局面上限、300秒请求超时和15秒告警保持不变，未修改代码、停止/重启训练、主动发起GPU查询/负载或提交推送。

## 32. 用户重启电脑后的显卡基础检查与Windows图形栈证据（2026-10-20）

用户要求检查显卡是否正常，并确认刚才重启过电脑。本轮没有恢复训练或对弈服务，没有重置驱动或运行压力测试。旧pipeline338192及推理338230已不存在，8765无法连接；Windows最后启动时间为00:04:12.942，WSL本次内核启动记录在00:08附近，不能继续把旧PID文件或旧性能文件当成活跃训练。训练最后一条已落盘SGD指标为398767步（00:03:23.482），performance最后快照为398762步；本轮CPU成功读取398700步learner，同一run ID，252个模型状态张量和129个SGD状态张量均有限。至少最后67次已记录更新不在该恢复点中，不能声称全部训练进度已保存。

### 32.1 当前基础功能检查

Windows设备枚举中RTX5070为Status=OK、ConfigManagerErrorCode=0，驱动版本32.0.16.1714。重启后的WSL实际使用PyTorch2.8.0+cu128、CUDA12.8，cuda.is_available为True；创建128×128全1矩阵，在GPU执行矩阵乘法并同步完成，逐元素结果均为128，设备名为NVIDIA GeForce RTX5070。分配/计算/同步约0.632秒，测试进程随后正常退出。该轻量验证只确认当前设备识别、分配和基本CUDA计算可用，不涵盖长时稳定性、全显存、温度或供电验收，不能据此保证硬件完全无故障。本轮没有再发起nvidia-smi查询。

### 32.2 新增Windows证据与结论边界

过去8小时System日志检索未发现Display/nvlddmkm/WHEA/DxgKrnl提供程序或4101匹配记录，但扩大到Application中的Windows Error Reporting后，发现LiveKernelEvent **141和193**。因此此前仅查System没有发现驱动错误的结果，不能延伸为没有任何图形栈异常记录。

- 141最早在本轮12小时查询窗口内于10月19日19:53:14.369上报，引用 WATCHDOG-20261019-1953.dmp，ReportId为050236d3-5353-4a47-a450-0d7811c8390f。微软定义0x141为显示引擎未及时响应的live dump标识；它本身不是系统蓝屏代码。[微软0x141文档](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0x141---video-engine-timeout-detected)
- 193引用 WATCHDOG-20261019-0825.dmp，ReportId为3ef3af92-8677-421f-bafc-9444bc1d5aac。微软定义0x193为dxgkrnl触发的live dump；不能仅由该编号确定物理损坏。[微软0x193文档](https://learn.microsoft.com/en-au/windows-hardware/drivers/debugger/bug-check-0x193--video-dxgkrnl-livedump)
- 上述同一ReportId在19:53、20:22、21:21及重启后再次上报，引用文件也相同，**不能把每条重复上报都数成一次新显卡故障**，也不能将00:07的上报时间当成新的故障发生时间。系统还有AMD核显和ToDesk虚拟显示设备；尚未分析转储中的负责模块，不能仅凭代码指定5070或某个驱动为根因。
- 00:04启动时记录Kernel-Power41，BugcheckCode=0、PowerButtonTimestamp=0，另有6008。事件41只确认上次未正常关闭，单凭它不能确定电源或显卡故障；用户已确认重启过电脑，不能将这次重启写成已证实的GPU自动崩溃。6008消息显示的23:28:21也早于训练日志的最后写入，不作为精确故障时间。[微软事件41说明](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/event-id-41-restart)

WER队列的Report.wer及LiveKernelReports转储读取受Windows目录权限限制；本轮未绕过权限或完成转储分析。现有证据支持“当前基础CUDA功能可用，但图形栈曾发生超时，长期稳定性仍需排查”，不能再把所有现象仅解释为网页采样显示问题，也不能判定显卡硬件已损坏。下一项有针对性的证据是分析141转储里的负责模块、堆栈和TDR上下文，再决定驱动或负载对照。旧训练的定时监控随本次已停止状态暂停，避免继续检查不存在的旧进程；未改模型、算法、参数、分支或推送。

## 33. 远程设备在线但会话连接失败：用户澄清与代码复查（2026-10-20）

用户进一步澄清：所谓网络错误是远程软件连接不上这台电脑，具体表现为“设备在线，但连接失败”，因此才重启电脑；不是8767对弈网页的HTTP错误。因果顺序应为远程连接失败在先、用户重启在后、旧训练和本地Web服务因重启消失，不能以重启后的8767未监听解释重启前的远程连接失败。本机发现ToDesk_Service及其进程；重启后服务Running/Auto，本轮未重启或配置远程服务。

### 33.1 证据范围

故障前ToDesk服务日志 service_gxwkwn_2026_10_20.xlog 最后写入00:03:17.249，训练metrics最后写入00:03:23.482，说明该时间附近仍有服务/训练写入活动；不能由远程失败认定整台机器早已完全死机。ToDesk的xlog是不可直接按文本读取的二进制格式，本轮仅核实文件属性及格式，没有将其臆译为具体连接错误，没有读取账号、密码或修改配置。23:00至00:04:10的System相关网络/服务事件、NetworkProfile和WLAN-AutoConfig Operational检索未找到匹配断线事件，Application检索未找到ToDesk崩溃记录；缺少事件不等于已排除网络、中继或远程服务内部故障。“设备在线”支持当时在线状态仍被报告，不能保证从控端到被控端的所有会话通路正常。

Windows141/193证据及去重边界仍按32节，未分析转储就不能把远程连接失败、WSL管理查询D状态及CUDA冷启动长停顿合并成一条已证实的因果链。ToDesk官方文档涉及DXGI画面采集和硬件编解码，因此图形栈是远程会话的一条合理待查路径；不意味着本次连接已走到该阶段。[ToDesk采集方式说明](https://docs.todesk.com/faq/111.html)

### 33.2 项目代码检查结果（未修改实现）

| 位置 | 已确认的行为/缺口 | 与本次远程故障的关系 |
|---|---|---|
| train/live.py 的 _sample_hardware / _hardware | 单查询线程防止重叠；前一查询结束后约3秒周期可再启动，不因连续失败退避或熔断。timeout=5不能保证已进入内核D等待的子进程5秒内退出 | 已知不健康GPU管理路径仍被长期重复访问，是应优先隔离的诊断设计缺口；尚未证明查询触发了图形异常，也可能是同一底层问题的受害者 |
| train/inference_server.py 的 load / _serve、model/cuda_graph.py | 推理服务串行创建新模型、按版本及批量形状惰性预热/捕获；历史已测创建42秒、预热最高138秒，期间旧模型请求也要等待；旧版本按仍活跃对局保留并清理 | 直接解释自博弈等待尖峰，不直接解释ToDesk会话失败；Python阶段墙钟不能推成同样时长的单个GPU内核或TDR触发原因 |
| train/trainer.py / config/local-train.yaml | batch4096的SGD与独立推理进程共用CUDA；未设置跨进程统一显存预算，也没有与桌面/远程图形负载协调的调度。play_server.py的5%只是该对弈进程的分配器限制 | 属于共享桌面显卡上的运行保障缺口，不是已证明的显存溢出或坏卡；进程reserved值不能相加当实测整卡容量，也不等于实际持续计算占用 |

在当前训练/推理/启动路径的代码检查中，没有发现操作ToDesk、修改防火墙/网卡或主动断网的实现。不能因为项目没有这些调用，就排除它通过共享GPU资源间接影响桌面/远程稳定性的可能。

继续训练前的建议顺序：先让已知会阻塞的GPU管理查询停止自动重试，使用现有进程内分配器与CPU侧进度监控；再在受控短对照中分别检查模型创建/Graph冷启动与learner并发的影响，并保留远程会话建立阶段证据。若要判定具体驱动模块，应读取141转储；当前Windows目录权限仍未允许。本轮仅复查代码和记录证据，不实施这些方案、不启动训练/对弈、不改算法或运行参数、不运行新增测试、不提交推送。无法从现有记录唯一判定本次远程会话失败的根因。

## 34. WSL dxg管理/提交错误与推理长停顿的时间序列复核（2026-10-20）

本节继续只读检查，未启动训练、对弈或GPU压力测试，未改代码和配置。

### 34.1 新增直接证据

- 19:39之后的运行窗口内，与GPU管理查询相关的dxg错误共97条：dxgvmb_send_sync_msg wait_for_completion failed: fffffe00 35条，process_completion_packet did not find packet to complete 35条，dxgkio_query_adapter_info Ioctl failed: -512 23条。它们的发生时刻与 performance.jsonl 中 nvidia-smi 查询启动时刻逐次对应。
- 21:17:01和21:17:11另出现 dxgvmb_send_async_msg: vmbus_sendpacket failed: fffffff5 及 dxgkio_submit_command_to_hwqueue: Ioctl failed: -11。这正是27.414秒 prepare 停顿发生的时间窗，属于GPU命令提交路径的直接错误证据，不只是显示层异常。
- 查询子进程进入 State: D (disk sleep)、wchan=dxgvmb_send_sync_msg 后，会等待数分钟至数十分钟：341888约从21:46:45持续到22:55:12，约4103.8秒；342156到00:03仍等待，已达1734.0秒。subprocess.run(timeout=5) 对D状态子进程没有硬超时保证，因此监控线程本身也会被拖住。
- 长推理阶段与查询D状态的时间对应如下：

| 长阶段结束时间 | 阶段 | 阶段墙钟 | 当时GPU管理查询 |
|---|---|---:|---|
| 21:35:07 | graph_warmup | 44.835秒 | 341832，D状态，约174.8秒 |
| 22:08:57 | graph_warmup | 42.114秒 | 341888，D状态，约1335.4秒 |
| 22:55:13 | model_create | 42.040秒 | 341888此前已等待约4103.8秒 |
| 23:00:05 | graph_warmup | 38.691秒 | 342048，D状态，约138.5秒 |
| 23:31:48 | graph_warmup | 138.149秒 | 342124，D状态，约820.3秒 |

这些窗口内learner的SGD仍持续更新，没有CUDA OOM、Python异常或runtime-failure记录。因此长停顿不是损失发散、Replay错误或已证实的整卡硬件宕机。

### 34.2 根因判断

- 高置信度：WSL dxg GPU管理查询路径已实际损坏；训练监控每约3秒调用WSL内 nvidia-smi，会让查询子进程在D状态不可中断等待，Python超时无法真正约束它。继续自动重试只会反复进入该异常路径。
- 中等偏高置信度：该异常与推理冷启动长停顿时间高度重合，并且21:17出现了GPU命令提交失败，支持它是停顿的触发或放大因素；但不能仅凭共现断言每个阶段都由此单一原因造成，也可能是WSL内核、Windows显示驱动和NVIDIA WSL驱动共有的底层异常同时影响查询与CUDA提交。
- 高置信度：没有证据支持Wi-Fi适配器自身掉线。23:00至00:04的WLAN/NetworkProfile事件未见断链重连；远程连接失败更可能来自主机/图形栈冻结、ToDesk采集或整机无响应。Kernel-Power 41确认非正常关机，但不足以确定电源或显卡故障。
- 尚不能判定：RTX 5070物理损坏。LiveKernelEvent 141/193的相同ReportId被WER重复上报，转储负责模块仍未读出；当前CUDA轻量计算正常不能替代长期稳定性验收。

### 34.3 修复前建议

优先彻底移除WSL内 nvidia-smi 自动监控，改用Windows主机侧性能计数器，或仅使用训练进程内的CUDA分配器读数；不要再依赖可在5秒内杀死GPU查询子进程的假设。完成隔离后，经用户批准再做短时对照：观察 dxgvmb_send_* 错误是否消失、graph_warmup 最大值是否回落。若长停顿仍存在，再评估CUDA Graph冷启动/批量形状策略或WSL与驱动版本组合，而不是直接归因坏卡。当前不启动训练，不修改模型、搜索参数或运行配置。

### 34.4 WinDbg转储分析结果（2026-10-20补充）

使用微软WinDbg 10.0.29661.1004的内核调试器kd.exe对两份转储执行 !analyze -v、.bugcheck 和 lm；完整输出保存在 diagnostics/analyze-141.txt 和 diagnostics/analyze-193.txt，符号缓存在 diagnostics/symbols。两份转储均为 Mini Kernel Dump，只有寄存器和调用栈，不包含完整内核内存。

- 141转储 WATCHDOG-20261019-1953.dmp：bugcheck为 VIDEO_ENGINE_TIMEOUT_DETECTED (0x141)，表示某个显示引擎未及时响应。WinDbg给出的Failure.Bucket为 LKD_0x141_IMAGE_nvlddmkm.sys，Failure.Exception.IP.Module为 nvlddmkm，MODULE_NAME为 nvlddmkm，IMAGE_NAME为 nvlddmkm.sys，IP为 nvlddmkm+0x193b0a0。Arg2 fffff8074b65b0a0 被解释为 responsible device driver module 指针。
- 141的调用栈是：nt 的 live triage dump采集 -> dxgkrnl!TdrCollectDbgInfoStage1 -> dxgmms2!VidSchiResetHwEngine -> dxgmms2!VidSchiResetEngines -> dxgmms2!VidSchiCheckHwProgress -> dxgmms2!VidSchiWaitForSchedulerEvents -> dxgmms2!VidSchiWorkerThread。该栈说明是GPU调度器/显示引擎超时后的TDR恢复路径，而不是训练代码异常栈。
- 193转储 WATCHDOG-20261019-0825.dmp：bugcheck为 VIDEO_DXGKRNL_LIVEDUMP (0x193)，Reason Code=0x815 (Internal)。WinDbg栈只显示 watchdog!WdpDbgCaptureTriageDump 和 watchdog!WdpProcessWatchdogTimeoutReportThread 等转储采集线程，Failure.Bucket为 LKD_0x193_watchdog!WdpDbgCaptureTriageDump。它不是直接责任模块，不能据此把193归给某个驱动。
- 141转储中没有可解析的TDR恢复上下文内存：dq 读取 ffffa90f0f48f010 全为问号，且公开符号没有导出 dxgkrnl!_TDR_RECOVERY_CONTEXT；同时没有NVIDIA私有符号，因此无法把 nvlddmkm+0x193b0a0 进一步解析成具体函数名。该限制不能抹去WinDbg已给出的 nvlddmkm 责任模块结论。
- 当前 nvlddmkm.sys 文件版本为 32.0.16.1714，驱动日期2026-09-17，文件位于 C:\Windows\System32\DriverStore\FileRepository\nv_dispi.inf_amd64_da865124972e1f80\。设备枚举中RTX 5070状态仍为OK，ConfigManagerErrorCode=0。
- 该结果把问题从WSL dxg查询症状进一步推进到Windows NVIDIA显示驱动路径：141是一次由nvlddmkm负责的显示引擎超时。它也不能直接证明物理GPU损坏；TDR通常会把超时归责给当时占用显示引擎的KMD，根本原因仍可能是驱动缺陷、GPU固件/调度、供电/温度、超频/降压或WSL GPU-PV并发负载。
- 现有WHEA日志未见错误级记录、System日志未见4101或nvlddmkm错误级记录，因此没有独立硬件错误证据；但也正因141已明确指向nvlddmkm，后续不能只做WSL监控隔离，还应核查驱动版本/已知问题、GPU供电与温度、是否超频或降压，并在用户明确批准后做受控稳定性对照。
- 193的挖因需要更完整的Kernel Dump或启用DxgKrnl/Watchdog ETW；当前迷你转储不足。ToDesk与远程连接失联仍应作为主机/显示栈冻结的上层表现，而不是把193直接归给ToDesk或Wi-Fi。

## 35. 移除WSL GPU监控的实现记录（2026-10-20）

按用户要求，本轮删除本次取证产生的临时诊断产物（diagnostics目录、转储副本、WinDbg输出、临时解析器和UAC临时文件），并从训练监控中移除不安全的WSL GPU采样。
- train/live.py 不再导入 subprocess，不再调用 nvidia-smi，不再创建GPU查询线程，不再检查D状态子进程，也不再依据GPU查询超时构造状态。每3秒仅保留psutil的CPU和内存采样；hardware快照返回 gpu_status=disabled 和说明文本。
- frontend/training.js 与 training.html 在 disabled 状态下明确显示 GPU监控已禁用 和 显存监控已禁用，历史快照的stale/unavailable显示分支仍保留兼容。
- 保留 train/runtime_diagnostics.py 及其错误传递、推理阶段心跳和请求超时记录：它们只读共享内存和CPU侧状态，不调用nvidia-smi或NVML，不属于本次危险GPU监控路径。
- 保留references/analyze_performance.py和现有性能指标；GPU字段缺失时分析器会自然跳过，不会伪造读数。
- CPU侧验证：py_compile通过；AlphaZero/tests/test_runtime_diagnostics.py 6项通过（12.17秒）；frontend Node性能与禁用状态测试1项通过。均未启动训练、对弈或GPU压力测试。
- 仍保留本报告中的取证结论：141转储的责任模块为nvlddmkm.sys（偏移0x193b0a0），193为dxgkrnl/watchdog livedump但迷你转储不含直接责任模块。移除监控只消除WSL nvidia-smi触发或放大因素，不替代NVIDIA驱动和GPU稳定性排查。

## 36. 移除GPU管理采样后的15分钟续训与定时停止（2026-10-20）

用户授权继续训练15分钟后停止。本次从398700步learner恢复，配置沿用400次搜索、8000步发布、100步恢复存档、最多100000个Replay局面、19×128/BF16/batch4096、32盘/8个actor进程及CUDA Graph；启用performance和runtime-diagnostics，请求超时300秒。WSL GPU管理采样保持disabled，没有发起nvidia-smi/NVML或额外GPU测试，也未修改算法/参数。run ID为 `12547b1e-9441-455f-bdb1-286e3cb2627a`，本次session为 `9627f0e2392d41bdaca49f2299d36587`。

### 36.1 定时停止与可恢复进度

独立watchdog PID387于北京时间06:29:18.564742启动pipeline408；在预定06:44:18.564742期限，于06:44:18.564887发送SIGINT。pipeline于06:44:20.800242退出，启动至退出902.235秒，即15分2.235秒，其中约2.235秒为停止清理；未触发强制终止。退出码-2、stderr末尾KeyboardInterrupt与watchdog主动中断一致，不属于本次新CUDA或worker故障。检查 exact-run status 已无匹配训练进程，推理509及已记录的actor/资源追踪子进程均已退出；复查中PID408已被新的/init复用，启动时间06:44:53、PPID2，不把这个后来进程误当仍在训练，也未对其发送信号。

本次完成398700→400094，共1394次SGD更新；最后指标和最终performance均为400094。完整learner与最新actor均为400000步，已在CPU实际加载，模型状态逐张量相等，learner全部模型/SGD张量均有限，run ID及指纹与恢复前一致。**400001—400094这94次已完成更新未到下一恢复点，没有保存在learner中**；下次应从400000恢复，不能声称从400094继续。learner引用Replay generation1860，同run活跃库generation1861、max_actor_step398700，满足当前恢复校验的generation上界与actor步数条件；活跃491盘/99969局面。本次新增6个完整对局、265个局面，尚未完成的自博弈未计入完整局产出。

### 36.2 粗粒度实测与停顿复核

用现有 references/analyze_performance.py 分析本会话累计快照，结果保存在run目录 resume-15min-20261019T222916Z-analysis.json。按现有脚本选首个elapsed≥60秒的报告作为基线，实际稳态窗口64.317—899.160秒，共834.843秒，步数398732—400094。下表均为累计总和/计数差分的窗口均值，模块并行时间不能相加；停止打断了下一次SGD，采样阶段比完成更新多1个样本，不将它算成已完成更新。

| 指标 | 窗口结果 | 样本/边界 |
|---|---:|---|
| 每手400次搜索 | 11.654秒 | 2284个完成整手 |
| 同步神经调用 / 本地墙钟估算 | 11.090 / 0.564秒每手 | 后者不是精确CPU执行时间 |
| 完成模拟 / 推理输入速率 | 1094.34 / 1100.53每秒 | 分别为完成整手模拟和真实输入局面 |
| 推理实际平均Batch | 8.762 | 104859个模型版本批次 |
| 推理服务 / 请求RTT | 3.386 / 27.126毫秒 | 批次与请求两种粒度，不能相减当纯IPC |
| SGD完整更新 | 604.856毫秒 | 1362次完成更新 |
| Replay采样 / SGD计算更新 | 172.687 / 406.695毫秒 | 采样1363次；计算含原有校验/同步等待 |
| CPU / WSL内存 | 24.01% / 6.922GiB均值 | 各168个快照；无整卡GPU/显存采样 |

本会话runtime-failures为空，stdout没有15秒推理告警，176份5秒独立阶段记录中没有≥30秒活动阶段。全会话模型创建最大0.354秒（2次），图预热最大0.565秒、捕获最大0.102秒（各7次）；包含首次冷执行/回传的整次Graph准备最大7.488秒，另有forward阶段结束日志6.636秒，不能把0.565秒误称整个冷调用只需0.565秒。本轮未复现此前38—138秒的长阶段；400000步发布/加载和两个驻留版本已覆盖，但新版本图形状与长时驻留覆盖有限。

此次短观察与移除GPU管理采样后的表现一致，但还经历过整机重启，模型/Replay进度、驻留版本组合和观察时长也不同；没有只切换一个因素的同状态对照。**不能据这15分钟认定nvidia-smi是此前全部故障的唯一根因、驱动问题已根治、显卡硬件已完成稳定性验收，或ToDesk连接已验证恢复。** 只确认本轮训练推进且按时停止，未见新的应用层运行错误和长推理阶段。未额外验证远程反复连接或主动采集GPU管理状态。

控制、stdout/stderr、分析JSON和独立watchdog均位于同一run目录，前缀 resume-15min-20261019T222916Z；新session的metrics/performance/阶段记录按session_id区分。停止后的定时复核任务已暂停，不再自动恢复训练。本轮只追加本报告与运行检查记录，没有优化或提交推送，保留当前分支和独立Policy工作树。

## 37. 五次模型发布续训完成复核（2026-10-20）

用户授权从最新存档继续训练，完成5次新发布后停止。本次恢复400000步，沿用400次搜索、8000步发布、100步恢复存档、最多100000个Replay局面及原有模型/优化器配置。通过已有命令参数 `--max-iterations 55` 将本次绝对停止步数限定为440000，没有修改训练配置文件、算法或模型。启动时重新导出400000步供actor加载，不计为一次新发布。run ID为 `12547b1e-9441-455f-bdb1-286e3cb2627a`，session为 `c18871bcc2d444ffac8fde2629978a1c`。

### 37.1 发布、自动退出与存档

stdout依次记录408000、416000、424000、432000、440000，恰好5次新发布，随后输出最终结果440000步、累计generation2127、99942个活跃局面。pipeline退出码0，stderr为空，supervisor标记completed；这是达到原生步数上限后的正常结束，未发送停止信号或强制终止。记录的北京时间为09:16:57.143508启动、16:10:00.218346退出；supervisor单调计时24824.890秒（6小时53分44.890秒），performance计时24821.128秒。前述民用时钟两时间戳相减为24783.075秒，比单调计时少41.815秒，原因本轮未另行查证，因此吞吐量以performance单调计时计算，不用民用时间戳反算。

进程身份按boot ID `3a2362ed-ed2a-471b-9680-f79f8f07c997`、pipeline PID1876、启动ticks183046、命令及cwd核对。结束后1876、推理1916和supervisor1874均不存在；枚举/proc未发现session ID1876的残留子进程。保留PID文件作为历史记录，不能只凭文件认定进程仍在运行。8767对弈进程438及监听保持运行。

CPU实际加载learner.pt与latest.pt：二者均为440000步、同run ID及指纹，模型张量逐项相等，learner模型和优化器张量全部有限。最终完成更新数与保存步数相同，本次没有已完成而未保存的尾部SGD更新。learner引用Replay generation2127；只读SQLite复核同run/指纹，活跃510盘、99942局面，max_actor_step432000≤440000，满足恢复校验的步数条件。整个会话新增266个完整对局、53278局面；停止时未完成的对局不计为完整产出。440000版本刚发布即结束训练，不能声称它已生成新完整对局。

### 37.2 本次性能测量

使用现有 `references/analyze_performance.py`，结果写入同run目录 `resume-5pub-20261020T011623Z-analysis.json`。排除启动阶段的实际窗口为64.808—24821.128秒，24756.320秒，400043—440000步；下表为累计总和与计数差分，使用该窗口均值。不同阶段的快照边界可能跨越正在执行的一次更新，例如采样39956次、完整更新39957次，不能将这些阶段视为严格相同样本集合。

| 指标 | 窗口实测 | 样本或解释 |
|---|---:|---|
| 每手400次搜索 | 13.573秒 | 58349个完成整手 |
| 同步神经调用 / 本地墙钟估算 | 13.073 / 0.500秒每手 | 本地部分不是精确CPU执行时间 |
| 完成模拟 / 推理输入 | 942.77 / 933.00每秒 | 完成整手与推理服务上报存在窗口边界差异 |
| 推理实际平均Batch / Graph执行平均Batch | 5.960 / 7.831 | 3875348个模型版本执行批次；后者含形状补齐 |
| 等待合批 / 服务 / 请求RTT | 2.385 / 3.443 / 32.364毫秒 | 合批为收集轮次、服务为版本执行批次、RTT为单个请求，粒度不同 |
| SGD完整更新 | 612.515毫秒 | 39957次完成更新 |
| Replay采样 / 主机侧传输 / 计算更新 | 177.027 / 25.334 / 409.794毫秒 | 墙钟含各自原有同步与等待 |
| 抽样CUDA传输 / 计算 | 28.217 / 404.826毫秒 | 各624次CUDA Event样本，包含并发竞争影响 |
| Replay写入 / 恢复存档 / 新模型发布 | 191.981 / 566.744 / 288.833毫秒 | 分别266、400、5次调用 |
| 新局面 / 完整对局 | 7747.55 / 38.68每小时 | 本次窗口产出速率，不代表棋力 |
| CPU / WSL内存 | 21.99% / 8.582GiB均值 | 4937个快照；内存范围7.777—8.763GiB |

自博弈每手约96.3%的墙钟处于同步神经调用，支持继续将请求等待、合批/版本拆分和共享GPU推理作为性能分析重点；请求RTT已包含推理、排队及调度，不能把它直接归为IPC成本。learner中计算更新墙钟最大，其次是Replay采样。以上模块相互并发，不相加当作整轮耗时；CUDA Event也不能据此换算整卡利用率。相较第36节短窗口，本次搜索均值更高、实际Batch更小，但局面、模型版本组合、对弈共享负载与观察时长不同，未做同状态对照，不能据此认定某一因素导致退化。本轮仅记录测量，不实施优化。

### 37.3 稳定性与结论边界

runtime-failures.jsonl仍为空，stdout/stderr未出现新CUDA、worker失败或15秒阶段告警。4963份独立阶段快照中没有≥30秒活动阶段；活动阶段抽样最大0.123秒不代表所有调用最大值。全会话计时统计另记录：模型创建6次，最大0.377秒；Graph预热/捕获各30次，阶段最大0.629/0.205秒；包含冷执行的整次Graph准备最大1.027秒。独立快照最多观察到3个驻留版本，推理进程CPU侧分配器读数最大allocated0.335GiB、reserved1.037GiB，均不是整卡显存读数。GPU管理采样始终disabled，本轮没有调用nvidia-smi/NVML，也未另行检查Windows图形事件或重复远程连接。

这次约6小时54分钟、覆盖5次新发布的运行没有复现历史38—138秒长阶段，是比15分钟观察更长的应用运行证据；仍不能据此认定此前图形驱动/远程连接问题根治或显卡硬件完成稳定性验收。控制及日志前缀为 `resume-5pub-20261020T011623Z`，原始metrics、performance和推理阶段记录用上述session ID筛选。完成后定时复核已暂停，训练保持停止；对弈服务继续运行，新对局可加载440000步模型。本轮只更新本报告和运行检查记录，保留当前分支、用户未跟踪文件及独立Policy工作树，没有提交、推送或合并。

## 38. 用户调整强制和棋上限为300步，续训至下一次发布（2026-10-20）

用户新的指令替代此前“继续发布10次”的目标：先暂停，设置双方合计300次落子后强制和棋，再训练到下一次发布即停止。原先从440000续训的session `fc1e5281c0fe46eab660e57367869400` 已由明确的停止请求发送SIGINT结束；最后完成448888步，可恢复learner为448800步，最近actor为448000步。88次尾部已完成更新未保存，此后从448800恢复。旧supervisor的exited_before_verified_target/-2是本次主动停止造成，不应作为新运行故障，也不再追赶旧520000步目标。

`max_game_ply` 原值512，既决定强制和棋，也进入配置指纹以及输入平面129的 `move_count/max_game_ply`。因此只修改配置、绕过指纹会混入错误归一化的历史输入。本次新增 `AlphaZero/scripts/migrate_ply_limit.py` 显式迁移至新的兄弟目录 `AlphaZero/runs/xiangqi-rules-ply300-20261020`，原目录保留；新run ID为 `5f1600f2-b1be-4c4e-ad9e-b95e2b6120b8`，指纹为 `2e63722f7524b98de14985b48841883c359581fae347815a7b767648d8a289b8`。保留模型、SGD动量、448800步数、学习率日程及RNG状态，CPU逐项核对模型和优化器状态与源存档一致。

源活跃Replay为518盘/99903局面。396盘/53819局面在300步内正常终局，逐步重放验证旧观察、策略合法性和结局标签，重建步数归一化后再次按新上限验证；超过300步的122盘/46084局面整盘排除，没有截断后伪造原结局。保留策略仍是旧512步上限下生成的MCTS访问数，写入迁移来源，不声称已用300步重新搜索。worker序号继续保留，避免复用已丢弃对局的种子；新run的累计产出统计重新以迁入局面为基数。由于全局SGD步数保留，而generated_positions分母重新统计，training_to_data_ratio会跳升，不能与旧run直接比较。新完整对局继续补充回放池，上限仍为100000局面。

`config/local-train.yaml` 修改max_game_ply=300及checkpoint_dir；新增 `AlphaZero/tests/test_ply_limit_migration.py` 的3项CPU测试已通过（2.24秒）：检查真实SGD存档迁移及下一次更新、长局剔除与观察转换、损坏输入拒绝，以及299/300步判和边界和终局胜负优先。没有修改自然限着、重复规则、MCTS预算、模型结构、SGD设置或已有Checkpoint字段要求。将死/困毙、重复、自然限着等已实现规则仍可先结束，300步是整局总步数上限，不是连续不吃子的上限，也不是双方各300步。

本次通过现有 `--max-iterations 57` 将绝对停止步数设为456000；启动重导出448800不计为新发布，456000发布后由原生循环正常保存退出。performance、runtime-diagnostics、300秒推理超时继续开启，GPU管理采样保持disabled。新控制与日志前缀为 `resume-ply300-nextpub-20261020T112514Z`，旧10次发布自动监控已替换为新run的一次发布复核。8767当前进程及对局保持运行，仍引用旧目录和旧模型，其显示不能作为新训练版本的证据。本节记录迁移及启动安排，完成结果由后续复核追加，不能提前宣称456000已保存。未提交、推送或合并。

### 38.1 456000步发布与正常停止复核（2026-10-20）

本次session为 `0f55418e3d4744379e8cfeda71d3230d`。supervisor记录北京时间19:25:40.793205启动、20:39:29.161870退出，单调计时4428.369秒（1小时13分48.369秒）。stdout仅出现预期的新发布 `step=456000`，随后输出最终456000步、generation444、62211局面；pipeline退出码0、stderr为空，控制文件phase=completed。核对boot ID及原PID76765/start ticks3839592后，确认pipeline、推理76800、supervisor76763均已退出，/proc没有session ID76765的残留子进程。未额外发送停止信号或强制终止。

CPU实际加载learner.pt和latest.pt，二者均为456000步、同新run ID及300步指纹，模型逐张量相等，learner模型与优化器张量全部有限；配置仍为300步上限、400次搜索、8000步发布、100步恢复、100000局面上限。完成448800→456000共7200次更新，全部保存，没有已完成而未保存的尾部更新。learner的Replay恢复标记为新run/generation444；只读SQLite确认444盘/62211局面、max_actor_step448800≤456000，与恢复校验相容。

迁入396盘/53819局面之后，本次新增48盘/8392局面。直接按新插入game ID查询：max_ply判和9盘，均恰好300步（合计2700局面）；no_legal_move终局36盘，合计5110局面、最长287步；repetition判和3盘，合计582局面、最长225步。新完整对局均未超过300步，为实际训练数据中的限步行为证据。456000模型在训练结束前刚发布，不能声称它已经生成这些48盘。

现有分析器输出为run目录 `resume-ply300-nextpub-20261020T112514Z-analysis.json`；实际差分窗口63.690—4426.267秒，共4362.577秒，448878—456000步，包含7122次完成更新。窗口基线恰跨一次更新，采样/传输计数7121，不能当作与7122次计算严格相同的样本集合。

| 指标 | 窗口实测 | 样本/解释 |
|---|---:|---|
| 每手400次搜索 | 11.069秒 | 12630个完成整手 |
| 同步神经调用 / 本地墙钟估算 | 10.551 / 0.518秒每手 | 本地值不是精确CPU执行时间 |
| 完成模拟 / 推理输入 | 1158.03 / 1148.10每秒 | 分别为完成整手模拟与实际输入 |
| 推理实际平均Batch / 服务 / 请求RTT | 9.236 / 3.420毫秒 / 25.915毫秒 | 542298个执行批次；RTT含推理与排队，非纯IPC |
| SGD完整更新 | 605.426毫秒 | 7122次更新 |
| Replay采样 / 主机侧传输 / 计算更新 | 173.875 / 25.757 / 405.478毫秒 | 各自墙钟，不能忽略并发与同步边界 |
| 抽样CUDA传输 / 计算 | 28.473 / 399.784毫秒 | 各111份CUDA Event样本 |
| Replay写入 / 恢复存档 / 新发布 | 105.248 / 565.712 / 291.203毫秒 | 分别48、72、1次调用 |
| 新局面 / 完整对局 | 6925.08 / 39.61每小时 | 本次窗口产出，不是棋力指标 |
| CPU / WSL内存 | 24.27% / 8.369GiB均值 | 872份快照，内存范围7.852—8.556GiB |

runtime-failures为空，stdout/stderr没有新的CUDA/worker错误或15秒推理阶段告警。885份独立阶段快照没有≥30秒活动阶段；完整阶段计时另记录模型创建2次、最大0.302秒，Graph预热/捕获各6次、最大0.542/0.032秒，包含冷执行的整次Graph准备最大0.894秒。窗口内新增Graph预热/捕获计数均为0，主要覆盖启动448800模型的6次形状准备；456000版本刚发布即停止，未充分覆盖其冷图执行或长时多版本驻留。不能把这次搜索均值较第37节低归为300步设置的独立加速效果，Replay集合、输入归一化和版本负载都已改变。

GPU管理采样保持disabled，本轮只使用既有遥测和CPU侧检查，没有运行GPU测试或查询nvidia-smi/NVML；上述结果不构成驱动问题根治、显卡硬件稳定性或远程软件连接的验收。自动复核已暂停，训练保持停止。8767原进程438仍指向旧 `xiangqi-rules-20261016/latest.pt`，未重启或重置用户对局，因此不会自动载入新目录的456000步模型。本轮只补充本报告及运行复核记录，没有修改算法/参数，也没有提交、推送或合并。

## 39. 参数直接配置与同目录续训（2026-10-20）

用户要求将运行参数调整归入main，不再为每次修改搜索次数或发布间隔新建分支、复制运行目录。main已包含原 `codex/training-search400-publish32000` 与性能监控提交。卷积50行动平面仍独立保留在 `alphazero_Policy`；左右镜像/5万局面方案仍留在未启用的独立工作树，不混入本次运行。

### 39.1 参数入口和兼容处理

- `scripts/training.ps1`统一读取配置的checkpoint_dir与telemetry_port。支持Simulations、PublishEvery、ReplayPositions、MaxGamePly、Publications直接覆盖启动参数；YAML持久修改与单次命令行覆盖均可使用。普通参数变化不再要求Git分支或新目录。运行中的进程不热改参数，需先正常停止再Resume。
- 搜索次数、模型发布间隔、回放局面上限在Resume时保留权重、SGD动量、全局步数和RNG。发布仍按全局步数的整间隔边界触发；Publications只计未来的新发布，启动重导出不计入，超出剩余更新预算时拒绝启动。调整记录追加到原有session_start/config_changes。
- max_game_ply影响输入归一化，不能仅绕过指纹检查。新增 `AlphaZero/train/resume.py`在原run锁保护下、CUDA加载前完成CPU预检和转换：验证保留棋谱的旧输入、合法策略和结局，重建步数归一化再验证。缩小时剔除超过新上限的整局；扩大时剔除旧上限强制判和的整局。历史MCTS访问数未重算，明确记录此限制。
- 同目录转换保留run ID、全局步数、模型/优化器状态、RNG、累计产出及worker序号。游戏generation采用SQLite自增水位，避免过滤末尾游戏后generation倒退或复用提交编号。转换前保留原目录中的 `.resume-backup-*` 文件；多文件替换使用恢复日志，中途失败回滚，模拟掉电后下次Resume可恢复。转换过程记录parameter-changes.jsonl，不创建新的永久训练目录。备份保留会增加磁盘占用。
- 人机对弈服务的新局重载允许单独的max_game_ply变化；网络、规则及其他规格不兼容或版本倒退仍拒绝。当前已加载的旧服务代码不会凭文件修改热更新，本次没有重启或重置用户棋局。

示例（本轮未对正在运行的训练执行此命令）：

```powershell
.\scripts\training.ps1 -Action Resume -Simulations 400 -PublishEvery 8000 -ReplayPositions 100000 -MaxGamePly 300 -Publications 10 -Performance -RuntimeDiagnostics
```

### 39.2 轻量验证和运行边界

主分支定向CPU回归：test_resume_parameters、test_search_budget_resume、test_replay_compact、test_ply_limit_migration、test_rule_migration、test_runtime_diagnostics共53项通过；test_performance与test_play另9项通过、1项CUDA测试因主动隐藏GPU而跳过。Node网页性能区测试1项通过。新增同目录调整测试覆盖上调/下调步数上限、模型/动量/RNG保持、历史回放转换、坏数据拒绝、安装中断即时回滚、掉电后恢复，以及启动重导出不计新发布。微型CPU管线验证实际能从原目录继续更新。

独立Policy工作树的test_policy_planes排除real_800长检查后25项通过，提交ada0f24保留50平面编码及卷积策略头。没有启动新的GPU测试负载、调用nvidia-smi/NVML，或把CPU通过等同于GPU稳定性认证。

实际训练仍为run `5f1600f2-b1be-4c4e-ad9e-b95e2b6120b8` / session `1ed1de4922f640f39595fd2e10ec4ff0`，进程89201未停止；只读Status在本轮观察到471267步。当前运行仍使用300步上限、400次搜索、8000步发布、至多100000回放局面，目标536000不变；本次代码修改在下次启动使用，没有对其Replay或Checkpoint执行转换。用户此前保留的旧run备份与独立工作树不删除、不改名。

## 40. 用户暂停并要求清理诊断与旧入口（2026-10-20至21）

本节更新当前状态，前文阶段诊断、旧后端和运行目标描述仅为历史记录。用户在北京时间10月20日23:48要求暂停：先校验boot ID、PID/start ticks、命令、工作目录及session，再于23:48:51.008向唯一匹配的pipeline89201发送SIGINT。supervisor记录23:48:56.860退出、exit_code=-2；这是主动暂停，不是推理故障。最后日志471807步，CPU实际加载的learner为471800步、latest为464000步，同run ID不变；7次已记录更新未保存。训练、推理和该session子进程全部退出，人机对弈服务保留。原536000/十次发布目标未完成，自动复核已暂停，不再自动追赶该目标。

### 40.1 main清理范围

- 删除独立推理阶段共享状态、定时采样、CUDA Graph阶段回调、分配器读数、阶段告警和诊断文件输出，以及相关启动参数和PowerShell开关。前端推理状态长行、显存内存池显示、无数据的GPU栏同步删除，没有只隐藏页面而保留后台采样。
- 保留用户此前要求的可关闭五模块粗粒度性能计时和正常训练指标；正常异常传播、进程退出检查及300秒请求超时属于故障处理，继续保留，异常直接进入标准错误。不会另建阶段诊断日志，也不查询GPU管理接口。
- 删除旧独立Web后端、配套页面/脚本/样式、数据库适配代码、旧接口测试、Web专用依赖和过时操作说明。保留8765训练观战与8767人机对弈，两者继续使用同一规则引擎。原Web测试中仍适用的颜色转换检查移至独立引擎回归，不保留失效后端导入。
- 删除未使用的计时Worker兼容模块、旧WDL模型别名和闲置单状态推理包装；神经评估只返回当前策略和标量价值。可选Arena模型加载也改用已有ModelPlayer，不再用旧模型尺寸默认值或旧搜索参数兜底；重复的benchmark命令行入口删除，独立评估使用evaluate_model。
- 新增统一路径解析：训练、对弈和评估默认读取config/local-train.yaml及其中checkpoint_dir；对弈默认选该目录latest.pt并读取配置搜索次数，评估默认将结果写在所选模型旁。允许显式指定其他配置、目录或快照。运行入口和README不再硬编码历史模型/数据目录；论文参考配置的空实验默认目录统一为AlphaZero/runs/training。当前本机checkpoint_dir不改名、不移动数据。
- 历史审查和测量证据保留当时路径用于追溯，不作为现行启动指令；用户未跟踪论文、旧模型/棋谱/数据库、独立Policy与镜像增强工作树均保留。

### 40.2 验证与边界

最终全套CPU回归128项通过、1项CUDA检查因主动隐藏GPU跳过（45.83秒）；网页性能区Node检查1项通过，两份现行前端JavaScript语法检查通过。回归覆盖真实小模型续训、同目录参数转换、Replay保留、worker异常传播、请求超时、合法走法和颜色归一化。

另用统一配置路径在CPU实际加载当前464000步actor，按300步规则上限完成一次单模拟搜索并返回合法走法；CPU读取learner确认471800步。该检查只验证现有文件与清理后推理入口兼容，不证明棋力或GPU稳定性，也不改变实际训练400次搜索配置。8767原服务仍响应，未为本轮清理重启、重置用户对局或启动新训练。本轮没有执行GPU测试或管理查询，真实CUDA清理后启动路径尚未重新运行。

## 41. 合并左右镜像增强（2026-10-21）

用户授权将codex/mirror-augmentation合并到main并删除该分支。分支工作区此前包含旧运行基线及未提交增强；先提交其完整状态以保留来源，再以main当前代码为基准解决合并冲突，只接入镜像增强相关内容。第40节删除的诊断、旧Web服务和旧路径均不恢复，main的统一入口、同目录参数调整和Replay generation水位处理保留。

增强在learner取样时对每个槽独立以50%概率选择左右反射，同步变换完整8帧历史和策略动作的起点/终点列；标量价值、行棋方与全局状态特征保持不变。回放库只存原始棋谱，不复制数据库局面；完整SGD batch仍为4096。该实现均匀取样原始加镜像的虚拟池，不保证每批成对，也不等于新增独立实战数据。仅用于learner，不修改MCTS或自博弈执行。

本机YAML开启mirror_augmentation，并将replay_window_positions从100000调整到50000，实现用户要求的“5万原局面×两种呈现”。默认配置开关仍为false；既有learner缺失该字段时视为关闭，恢复时记录false→true，保留模型、SGD动量、步数和RNG。开关不改变模型/规则指纹及Replay格式，导出的actor省略该learner专用字段，避免已运行的对弈服务因未知配置键失败。关闭增强时保留原先随机数消费。

沿用分支已有的test_mirror_augmentation检查用例，覆盖动作镜像双射、红黑完整历史、合法策略对应、均匀虚拟池、关闭时RNG一致、存档恢复和actor兼容。本轮执行合并与差异检查，没有重新运行测试或训练；不将此前分支的检查结果冒称合并后的新验收。训练此前两次新发布已完成，现有learner/latest为480000步；本次不启动训练、不改写存档或裁剪现有Replay。新5万局面上限及镜像开关在用户下次Resume时应用，8767对弈保留。
