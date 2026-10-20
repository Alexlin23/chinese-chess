# 中国象棋 AlphaZero

目标是在中国象棋上复现 AlphaZero 的学习与搜索方法，只缩减网络规模，并做硬件执行优化。**当前是重构中的开发版，还不能称为 100% 复现完成。** 主要缺口是完整中国象棋长捉责任裁定、独立规则验收及长期学习验证。唯一主审查与修复记录见 [ALPHAZERO_PAPER_REVIEW.md](ALPHAZERO_PAPER_REVIEW.md)，第 20 节记录收到 Data S1 后的核对与修正。

## 可关闭的性能监控

`codex/performance-monitoring` 基于main添加五模块粗粒度计时，实际短时观测与分析统一记录在主审查文档第25节。本轮仅增加观测，保留现有规则、MCTS/800次预算、模型、损失、随机数调用及checkpoint规格。

在本分支PowerShell中，使用 `./scripts/training.ps1 -Action Resume -Performance` 开启；去掉 `-Performance` 则关闭，仍可正常恢复已有存档。WSL直接启动可使用：

~~~bash
bash scripts/train.sh --config config/local-train.yaml \
  --output-dir AlphaZero/runs/xiangqi-rules-20261016 \
  --resume AlphaZero/runs/xiangqi-rules-20261016/learner.pt --performance
# 关闭：使用 --no-performance（默认关闭）。
~~~

默认每5秒聚合，GPU每64次实际批次/更新抽样一次；可用 `--performance-interval` 和 `--performance-gpu-every` 调整，开关不是训练配置字段，不写入checkpoint。运行中的8765页面显示MCTS、推理、请求往返、SGD、存储与产出速率；展开可查看样本数、累计值和近似P95上界。原始汇总追加到run目录的 `performance.jsonl`，最近一次开启监控的会话摘要为 `performance.json`，各自包含session_id；关闭后的新会话不会覆盖旧性能文件。

MCTS本地时间是整手墙钟减同步神经调用时间的估算，RTT包含推理及排队，不能当作纯IPC。CUDA Event复用已有同步点读取，不新增synchronize；传输主机准备时间不等于DMA完成时间，CUDA区间包含流竞争/主机间隙。并发模块耗时相互重叠，不能相加当总周期。P95使用固定直方图近似，不取各worker的P95平均值。

本次实际从138800步短时续训约300秒后停止，存档139200步；排除首60秒后每手800搜索约25.50秒，其中同步神经调用24.35秒，本地估算1.15秒。每次SGD约639ms，Replay采样179ms，计算更新432ms。期间没有新整局完成，小时产出和Replay写入不能作为稳定吞吐结论。见 `references/performance_observation_20261017.json`。当前训练保持停止，未进行后续优化。

训练停止时现有监控服务随进程关闭，已连接页面可保留最后的性能数值，刷新后需等下次启动；离线分析读取上述文件。本轮浏览器策略拒绝读取先前的错误页面，页面内容通过本机HTTP及脚本检查，未完成真实浏览器截图验收。

## 当前训练路径

- 纯神经网络 MCTS 自我对弈；每手新建搜索树、逐次完成 800 次模拟，根 Dirichlet 噪声；前 30 ply 按访问数采样，之后选择访问数最多的动作，训练目标始终保留归一化访问数；没有启发式监督、吃子权重或 Arena 回滚。
- 连续 SGD learner 与并行 actor 同时运行。每局固定一个已发布网络版本，双方使用同一版本；中心服务按版本合批，保存每手模型版本。
- 按 Data S1 异步运行，不按固定对局数限制每次更新；每 1000 次更新发布网络。4400 万局是原国际象棋实验的数据量，不是伪代码规定的每步配额。默认 8 actor 与原实验 5000 actor 的数据生成速度不同，日志记录生成量和数据复用比；尚未证明同等学习效果。
- 8 帧历史及重复特征，131 输入通道；当前玩家视角下同时变换棋盘与动作。
- Policy CE + 标量 tanh value 的最终结果 MSE。完整 batch=4096；SGD momentum=0.9，初始 LR=0.2，100k/300k/500k 更新后各降 10 倍，总预算 700k 更新。
- 已取得用户提供的官方 Data S1：正则项采用 c/2 × 参数平方和，SGD weight_decay=1e-4。**损失归一化仍有来源差异**：默认按局面取 batch 均值；Data S1 字面求和可用 loss_reduction: sum 核对，但完整 4096 batch、LR=0.2 的短检查出现非有限推理。详情、复现命令及失败证据见主报告第 20 节，不能称已完全复现原优化器。
- 保留论文的 19 个残差块，通道缩为 128，约 725 万参数；完整 4096 batch 的 CUDA 实测预留峰值约 7.97GB。平铺 8100 动作头属于架构选择。BF16、channels-last、有界队列、跨对局合批。默认单树 batch=1、virtual_loss=0、reuse_tree=false；单树批量搜索和复用是可选实验，与默认参考搜索分开。
- 最近 100 万**完整对局**存入 SQLite，均匀抽局面；不将窗口偷偷缩为 100 万局面。SQLite 内存缓存默认 256MB，数据保存在磁盘，磁盘需求随对局长度增长。
- 原子 learner checkpoint 与 actor 发布；回放每局事务提交，记录 run ID、规则/输入/动作/目标版本及 worker 序号。恢复拒绝跨 run、缺数据和旧 WDL/18 通道状态。

中国象棋规则和动作编码是论文没有给出的适配。用户已确认以中国象棋协会审定的《象棋竞赛规则（2020版）》为目标。当前 rule_version 仍明确标为开发版：已处理常见合法走法、将死/困毙优先级、三次重复中的长将和 120-ply 无吃子限着；**尚未完成国规的完整长捉、例外及特殊计数，不能把这些简化条款称作国规完整实现。** 暂不混用开发版数据与未来正式规则实验。

## 在本机启动

已在 WSL 的 /home/lin/.venvs/xiangqi-alphazero 建立独立环境：Python 3.10、PyTorch 2.8.0+cu128，实际识别 RTX 5070。不会使用系统 Python 的旧 CPU 环境。

在项目目录的 PowerShell 中启动：

~~~powershell
wsl --cd D:\project\Alphazero --exec bash scripts/train.sh --config config/default.yaml
~~~

训练运行时打开 [训练实况](http://localhost:8765)。这会运行完整默认预算；初次收集完整自对弈后开始优化。Ctrl+C 停止，保留最近一次完整 checkpoint 和已提交的对局。

建议把大型回放放在 WSL 的 Linux 文件系统。在 WSL 终端中：

~~~bash
cd /mnt/d/project/Alphazero
bash scripts/train.sh --config config/default.yaml --output-dir "$HOME/alphazero-runs/paper-v3"
~~~

恢复必须指向同一个 run 的 learner 和 replay：

~~~bash
bash scripts/train.sh --config config/default.yaml \
  --output-dir "$HOME/alphazero-runs/paper-v3" \
  --resume "$HOME/alphazero-runs/paper-v3/learner.pt"
~~~

默认不会静默自动恢复，也不会把不完整权重文件当完整训练状态。请给新实验使用独立输出目录。

若重建环境，运行 bash scripts/setup_wsl.sh。CUDA 安装方式依据 [PyTorch 官方版本页](https://pytorch.org/get-started/previous-versions/)；不要给 RTX 50 系列照搬旧文档的 cu118 命令。

## 浏览器训练实况

当前训练目录：`AlphaZero/runs/xiangqi-rules-20261016`（本轮短时观测后保持停止）。在PowerShell可管理该run：

~~~powershell
.\scripts\training.ps1 -Action Status
.\scripts\training.ps1 -Action Stop
.\scripts\training.ps1 -Action Resume
~~~

Status 显示实际进程和最近更新。Stop 只中断对应 run；Resume 保留完整回放并从 learner checkpoint 继续。训练日志位于该目录。新代码每 100 步写恢复状态，每 1000 步向自对弈发布模型，两者分别控制。

停止后，可以修改 `config/local-train.yaml` 中的 `num_simulations`（每手搜索次数，例如 800 改为 400），再执行 Resume。模型、SGD 动量、累计训练步数和已有回放继续保留；新启动的自对弈使用修改后的次数，尚未完成的旧对局不恢复。参数必须为正整数。

`learner.pt` 保存最近恢复点的训练状态和配置，`config.json` 保存本次启动配置；`metrics.jsonl` 追加保存每次启动的完整配置、恢复步数与 `config_changes`（旧值/新值），不会用新配置覆盖历史记录。新生成棋谱的 trace 同样记录搜索次数。学习率、模型结构等其他续训限制保持原有检查。

### 左右镜像增强（codex/mirror-augmentation 分支）

此分支的本机配置使用 `replay_window_positions: 50000` 和 `mirror_augmentation: true`。
回放库最多保留最近50000个原始局面，按完整对局淘汰，实际数量可能略少；采样时对每个样本独立以50%概率选择左右镜像，相当于均匀抽取“原局面＋镜像”的虚拟池，候选呈现最多100000个。对称局面的镜像可能与自身相同，这不等于100000个独立原始局面。每次SGD仍采样4096项，不强制同一批中成对出现，也不把Batch增为8192；数据库只保存原始棋局。

镜像同步反射8帧历史棋盘和策略中的起点/终点列，保持行棋方、步数、重复计数及胜负价值不变。它只用于learner取样，不在自博弈、MCTS或推理路径额外变换棋盘；属于针对中国象棋的训练扩展，原版AlphaZero没有使用对称增强。关闭开关后不额外消耗镜像随机数。启用后，镜像随机状态随现有NumPy RNG存档，续训可重现后续抽样。

现有相同模型/规则的learner可以恢复，保留权重、SGD动量和累计步数；`config_changes`记录镜像开关及回放上限变化，下次启动时才按新上限裁剪真实回放。actor导出省略仅learner使用的新增配置键，使已经运行的对弈服务可以继续读取新actor快照。无需为这个开关另建训练目录或迁移棋局。默认开关为false，本分支的local配置显式开启；本次只在独立工作区实现，正在运行的D盘工作区继续使用原来的配置。

该 run 目前使用 `config/local-train.yaml`：32盘独立对局分配到8个CPU工作进程，每盘仍顺序执行800次搜索。Resume默认沿用该配置；可用 `-Config config/default.yaml` 明确选择8盘配置。已保留1000、2000、3000步模型；它们仍是早期模型，独立评估尚未达到成熟棋力。

无需另启 FastAPI，训练进程自带只读页面：

- 活跃对局自动增加/移除卡片，**有几盘同时下就显示几盘**，没有固定棋盘数量和分页上限。
- 每盘显示棋盘、行棋方、手数、耗时、模型版本及搜索次数。
- 总览显示完成局数、回放局面数、优化步、策略/价值损失及 CPU/GPU/显存/内存。
- 页面自动重连；窗口大小和棋盘密度自适应。通过 SSE 发送最新观测，慢客户端不阻塞训练。
- 观测事件可丢帧，训练对局和标签走独立可靠队列，完整局提交后才进入回放。
- 监听默认仅本机。修改 telemetry_port 可同时观察不同 run；每个 run 使用不同端口和输出目录。

旧 FastAPI 对战演示仍是独立功能；它的完整状态、请求版本与数据库事务问题尚未在本轮全部迁移。**新训练观战页不走旧对战接口，也不从演示棋局伪造训练画面。**

## 使用已训练模型下棋

新的对弈入口直接使用完整规则状态和本轮标量模型，不依赖旧对战接口。在 WSL 项目目录运行：

~~~bash
$HOME/.venvs/xiangqi-alphazero/bin/python -m AlphaZero.scripts.play_server \
  --checkpoint AlphaZero/runs/xiangqi-v3-20261015/latest.pt --simulations 800
~~~

打开 [与模型对弈](http://localhost:8767)，点击棋子及目标位置即可，可选择红黑方。默认对弈使用 CPU，不争用训练显存；可明确加 `--device cuda`，当前已运行的页面使用CUDA。每局固定一个快照；指向latest.pt时，新对局会载入最新已发布模型。用 `--fixed-snapshot` 可固定实验快照。第 0 步随机模型会被拒绝，不会伪装成已训练成果。

独立评估示例：

~~~bash
$HOME/.venvs/xiangqi-alphazero/bin/python -m AlphaZero.scripts.evaluate_model \
  --checkpoint AlphaZero/runs/xiangqi-v3-20261015/latest.pt \
  --output AlphaZero/runs/xiangqi-v3-20261015/evaluation-random.json \
  --baseline random --games 8 --simulations 64
~~~

评估记录载入的确切更新步数、红黑胜负、和棋与逐局结果。64 次搜索只用于这项短评估，不改变训练的 800 次搜索。少量对随机走法的结果不能作为强棋力或完整国规认证。

## 规模与检查

config/default.yaml 是目标训练参数；config/test.yaml 与 config/cuda-smoke.yaml **仅用于短链路检查**，缩减搜索/批大小的结果不能算论文基线或棋力成果。

~~~bash
bash scripts/train.sh --config config/test.yaml --output-dir /tmp/az-cpu-check
bash scripts/train.sh --config config/cuda-smoke.yaml --output-dir /tmp/az-cuda-check
$HOME/.venvs/xiangqi-alphazero/bin/python -m AlphaZero.scripts.runtime_check --config config/default.yaml
~~~

本次实际完成：CPU 端到端；CUDA 并发自对弈与 100 步小预算训练；默认 800 搜索、8 actor、4096 batch 的短闭环；默认 batch 的 CUDA checkpoint 恢复；浏览器实时多盘检查。上述为早期验收；Data S1 修正后的新增检查与边界在主报告第 20 节。新 run 默认位于 AlphaZero/runs/paper-v3，拒绝混用旧算法数据。

compile_model 默认关闭；开启前应在本机对比编译开销与稳态吞吐。GPU 利用率不是正确性或棋力证据，扩大 worker 数前应观察 CPU、排队和实际合批分布。WSL 当前可见约 15.2GB 内存，是主机 32GB 的默认分配附近值；本轮未改 WSL 系统资源限制。

## 目录

~~~text
AlphaZero/engine       规则状态与观测编码（完整国规仍待完成）
AlphaZero/model        Policy/value 网络及坐标映射
AlphaZero/search       参考串行 PUCT；可选批量/复用实验
AlphaZero/train        learner、actor、磁盘回放、推理与只读观测
AlphaZero/scripts      训练与本机硬件检查入口
frontend/training.*    实时训练观战页面
config                 目标配置与明确隔离的冒烟配置
scripts                WSL 环境与启动脚本
~~~

修复进展统一维护在主审查文档；仓库原有 CODE_REVIEW/PLAN 等是历史材料，不能作为新训练路径已经验收的证明。

## 规则状态统一分支

`codex/rule-state-consistency` 先统一现有开发版规则状态；完整国规2020的长捉、循环责任和自然限着例外仍待实现。行动编码和模型架构沿用当前版本。

底层颜色统一通过 `AlphaZero.engine.player_id()` 转换：红方支持 `True/1/"r"`，黑方支持 `False/-1/"b"`，包括 NumPy 布尔和整数；其他输入明确报错。

旧 FastAPI Web 接口现在从服务端保存的起始局面和完整棋谱恢复 `GameState`，与训练及搜索使用同一状态转换、编码和终局逻辑。走棋、悔棋在单个数据库事务内完成；`revision` 在走棋和悔棋后均递增。数据库兼容添加起始局面和版本字段；旧棋谱仅在完整重放后与保存棋盘一致时恢复，不能补造缺失历史。每次读取重放完整棋谱，以正确性为先，后续可用版本化缓存优化。

接口有明确兼容变化：`/api/valid-moves`、`/api/move`、`/api/check-win`、`/api/ai-move`、`/api/heuristic-move` 均要求 `game_id`、`revision`、`board`、`turn`。客户端须先创建或获取对局，再携带返回的版本号。旧的只传棋盘调用返回422，棋盘或版本过期返回409；不会静默清空历史。悔棋请求体为 `{"revision": 当前版本号}`。前端 `chess.js` 已同步修改，须和后端一起部署。

旧 Web 服务可通过 `XIANGQI_MODEL` 明确指定本分支兼容的已训练模型，神经搜索使用完整服务端规则状态，并关闭探索噪声。Web 依赖在 `requirements-web.txt`，开发检查依赖在 `requirements-dev.txt`。现有8765训练监控和8767对弈服务不需为本分支重启。

### 自然限着将军计数

本分支规则版本为 `xiangqi-development-v3-natural-limit`。按2020版第4.2.4和22.2条，在最后一次吃子之后分别累计红黑将军着数；提出方最多计入10次将军，超出的部分从该方有效着数扣除。完整回合也必须足够：有效回合 = min(未吃子半回合数整除2, 提出方未吃子着数 - max(提出方将军次数 - 10, 0))。吃子后双方将军计数同时清零，吃子兼将军的这一着不计入新一段未吃子阶段。

自对弈自动裁定约定：每个局面由轮到行棋的一方作为自动提出方，有效回合达到60则判和；不能把双方超额将军合并扣除，也不要求双方同时达到各自申请条件。将死、困毙及当前已实现的重复责任判断优先。该约定不模拟申请、犯规扣时和裁判流程。50回合快棋及其他国规缺项仍未实现。

训练监控事件、对弈API及旧Web响应的 `natural_limit` 给出原始半回合数、双方将军数、双方有效回合和当前提出方是否达标。历史重放、悔棋和MCTS缓存区分这些计数。131通道网络输入暂不扩展，现有无吃子输入平面仍记录原始计数；双方将军计数尚未各自增加独立输入平面，后续完整规则观测设计仍需补齐。

规则变更会改变终局标签，配置指纹和回放版本已升级；旧开发版规则的learner/replay不会直接恢复为此版本。显式迁移工具只接受已知的v2规则及相同模型/算法规格，创建全新运行目录，保留模型权重、SGD动量、累计步数、学习率日程和随机状态。逐盘重放旧棋谱，核对输入、策略合法性与终局价值；终局发生变化的整盘剔除，数据不完整或损坏则终止迁移。保留策略目标仍是旧规则下生成的MCTS访问数，不会假称已重新搜索，来源写入棋谱及迁移报告。

~~~bash
python -m AlphaZero.scripts.migrate_natural_limit --source AlphaZero/runs/xiangqi-v3-20261015 --destination AlphaZero/runs/xiangqi-rules-20261016
~~~

工具必须在旧训练停止后执行，目标目录必须尚不存在。此次已迁移81300步存档，145盘、18517局面全部保持相同终局；旧目录保留作为回退点。`scripts/training.ps1` 的默认Status/Stop/Resume现在对应 `xiangqi-rules-20261016`，旧目录可通过显式 `-RunName xiangqi-v3-20261015` 查询。
