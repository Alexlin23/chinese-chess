# 中国象棋 AlphaZero

目标是在中国象棋上复现 AlphaZero 的学习与搜索方法，只缩减网络规模，并做硬件执行优化。**当前是重构中的开发版，还不能称为 100% 复现完成。** 主要缺口是完整中国象棋长捉责任裁定、独立规则验收及长期学习验证。唯一主审查与修复记录见 [ALPHAZERO_PAPER_REVIEW.md](ALPHAZERO_PAPER_REVIEW.md)，第 20 节记录收到 Data S1 后的核对与修正。

本地 `alphazero_Policy` 分支将策略头改为两层卷积输出50个动作平面，完整说明及验证记录见主报告第25节。本分支和main中正在运行的8100输出模型具有不同规格，旧learner、actor和回放不可直接resume；当前main训练继续使用原模型。本分支尚未启动长期训练。

## 当前训练路径

- 纯神经网络 MCTS 自我对弈；每手新建搜索树、逐次完成 800 次模拟，根 Dirichlet 噪声；前 30 ply 按访问数采样，之后选择访问数最多的动作，训练目标始终保留归一化访问数；没有启发式监督、吃子权重或 Arena 回滚。
- 连续 SGD learner 与并行 actor 同时运行。每局固定一个已发布网络版本，双方使用同一版本；中心服务按版本合批，保存每手模型版本。
- 按 Data S1 异步运行，不按固定对局数限制每次更新；每 1000 次更新发布网络。4400 万局是原国际象棋实验的数据量，不是伪代码规定的每步配额。默认 8 actor 与原实验 5000 actor 的数据生成速度不同，日志记录生成量和数据复用比；尚未证明同等学习效果。
- 8 帧历史及重复特征，131 输入通道；当前玩家视角下同时变换棋盘与动作。
- Policy CE + 标量 tanh value 的最终结果 MSE。完整 batch=4096；SGD momentum=0.9，初始 LR=0.2，100k/300k/500k 更新后各降 10 倍，总预算 700k 更新。
- 已取得用户提供的官方 Data S1：正则项采用 c/2 × 参数平方和，SGD weight_decay=1e-4。**损失归一化仍有来源差异**：默认按局面取 batch 均值；Data S1 字面求和可用 loss_reduction: sum 核对，但完整 4096 batch、LR=0.2 的短检查出现非有限推理。详情、复现命令及失败证据见主报告第 20 节，不能称已完全复现原优化器。
- 19个残差块、128通道，策略头为3×3卷积→BN→ReLU→3×3卷积，直接输出50×10×9 logits，按平面、起点顺序展平为4500；全网5,993,269个参数。BF16、channels-last、有界队列、跨对局合批。默认单树batch=1、virtual_loss=0、reuse_tree=false；单树批量搜索和复用是可选实验。原8100头的CUDA显存与速度测量不代表本分支，新头尚未单独测量GPU峰值及速度。
- 最近 100 万**完整对局**存入 SQLite，均匀抽局面；不将窗口偷偷缩为 100 万局面。SQLite 内存缓存默认 256MB，数据保存在磁盘，磁盘需求随对局长度增长。
- 原子 learner checkpoint 与 actor 发布；回放每局事务提交，记录 run ID、规则/输入/动作/目标版本及 worker 序号。恢复拒绝跨 run、缺数据和旧 WDL/18 通道状态。

中国象棋规则和动作编码是论文没有给出的适配。用户已确认以中国象棋协会审定的《象棋竞赛规则（2020版）》为目标。当前rule_version仍明确标为开发版：已处理常见合法走法、将死/困毙优先级、三次重复中的单方长将及自然限着提出方将军10着上限。完整长捉、循环责任和例外仍未完成，不能称为完整国规；双方将军计数尚未各自增加独立观测平面。

## 在本机启动

已在 WSL 的 /home/lin/.venvs/xiangqi-alphazero 建立独立环境：Python 3.10、PyTorch 2.8.0+cu128，实际识别 RTX 5070。不会使用系统 Python 的旧 CPU 环境。

在项目目录的 PowerShell 中启动：

~~~powershell
wsl --cd "$($PWD.Path)" --exec bash scripts/train.sh --config config/default.yaml
~~~

本分支默认监控端口为[8768](http://localhost:8768)，与main正在使用的8765区分。启动后初次收集完整自对弈再开始优化。Ctrl+C停止，保留最近一次完整checkpoint和已提交对局。不同端口不代表显存足够同时运行两个完整训练，本轮没有启动第二个GPU任务。

建议把大型回放放在 WSL 的 Linux 文件系统。在 WSL 终端中：

~~~bash
# 在alphazero_Policy工作树根目录执行。
bash scripts/train.sh --config config/default.yaml --output-dir "$HOME/alphazero-runs/policy-planes-v1"
~~~

恢复必须指向同一个 run 的 learner 和 replay：

~~~bash
bash scripts/train.sh --config config/default.yaml \
  --output-dir "$HOME/alphazero-runs/policy-planes-v1" \
  --resume "$HOME/alphazero-runs/policy-planes-v1/learner.pt"
~~~

默认不会静默自动恢复，也不会把不完整权重文件当完整训练状态。请给新实验使用独立输出目录。

若重建环境，运行 bash scripts/setup_wsl.sh。CUDA 安装方式依据 [PyTorch 官方版本页](https://pytorch.org/get-started/previous-versions/)；不要给 RTX 50 系列照搬旧文档的 cu118 命令。

## 浏览器训练实况

本分支本机配置使用新的 `AlphaZero/runs/xiangqi-policy-20261016`，与main现有训练目录隔离；本轮只完成临时目录检查，尚无该目录的长期训练。以后创建该运行后，可在本分支目录的PowerShell管理：

~~~powershell
.\scripts\training.ps1 -Action Status
.\scripts\training.ps1 -Action Stop
.\scripts\training.ps1 -Action Resume
~~~

Status 显示实际进程和最近更新。Stop 只中断对应 run；Resume 保留完整回放并从 learner checkpoint 继续。训练日志位于该目录。新代码每 100 步写恢复状态，每 1000 步向自对弈发布模型，两者分别控制。

停止后，可以修改 `config/local-train.yaml` 中的 `num_simulations`（每手搜索次数，例如 800 改为 400），再执行 Resume。模型、SGD 动量、累计训练步数和已有回放继续保留；新启动的自对弈使用修改后的次数，尚未完成的旧对局不恢复。参数必须为正整数。

`learner.pt` 保存最近恢复点的训练状态和配置，`config.json` 保存本次启动配置；`metrics.jsonl` 追加保存每次启动的完整配置、恢复步数与 `config_changes`（旧值/新值），不会用新配置覆盖历史记录。新生成棋谱的 trace 同样记录搜索次数。学习率、模型结构等其他续训限制保持原有检查。

`config/local-train.yaml` 配置32盘独立对局、8个CPU工作进程，每盘顺序执行800次搜索。Resume默认使用该配置；可用 `-Config config/default.yaml` 明确选择8盘配置。旧目录1000、2000、3000等快照使用旧策略头，不能在本分支直接加载。

无需另启 FastAPI，训练进程自带只读页面：

- 活跃对局自动增加/移除卡片，**有几盘同时下就显示几盘**，没有固定棋盘数量和分页上限。
- 每盘显示棋盘、行棋方、手数、耗时、模型版本及搜索次数。
- 总览显示完成局数、回放局面数、优化步、策略/价值损失及 CPU/GPU/显存/内存。
- 页面自动重连；窗口大小和棋盘密度自适应。通过 SSE 发送最新观测，慢客户端不阻塞训练。
- 观测事件可丢帧，训练对局和标签走独立可靠队列，完整局提交后才进入回放。
- 监听默认仅本机。修改 telemetry_port 可同时观察不同 run；每个 run 使用不同端口和输出目录。

旧FastAPI对战演示和独立PlaySession均通过完整GameState搜索；旧Web已加入服务端棋谱重放、事务和请求版本。训练观战页接收真实自对弈观测事件。

## 使用已训练模型下棋

新的对弈入口直接使用完整规则状态和本轮标量模型，不依赖旧对战接口。在 WSL 项目目录运行：

~~~bash
$HOME/.venvs/xiangqi-alphazero/bin/python -m AlphaZero.scripts.play_server \
  --checkpoint AlphaZero/runs/xiangqi-policy-20261016/latest.pt --simulations 800
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

本分支继承main中已合并的规则状态修正；完整国规2020的长捉、循环责任和自然限着例外仍待实现。策略头与行动编码的变更见下文。

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

上述迁移是main上同一旧策略头的历史记录：81300步存档、145盘及18517局面保持相同终局。自然限着迁移工具要求模型/动作规格相同，不负责将main的全连接头变成当前卷积头；本分支会拒绝此类跨架构来源。本分支管理脚本默认指向新的 `xiangqi-policy-20261016`，不会控制main正在运行的任务。

### 50平面卷积策略头

按照[AlphaZero Science论文的棋类卷积策略头](https://www.davidsilver.uk/wp-content/uploads/2020/03/alphazero-science_compressed.pdf)构建两层卷积，保留本机128通道的规模让步。论文未定义中国象棋；这里的50平面是本项目的中国象棋适配。

| 平面索引 | 位移 `(行变化, 列变化)` | 数量 |
|---|---|---|
| 0—17 | 向上1—9格，再向下1—9格 | 18 |
| 18—33 | 向左1—8格，再向右1—8格 | 16 |
| 34—41 | `(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)` | 8 |
| 42—45 | `(-2,-2),(-2,2),(2,-2),(2,2)` | 4 |
| 46—49 | `(-1,-1),(-1,1),(1,-1),(1,1)` | 4 |

动作索引为 `平面×90 + 起点行×9 + 起点列`，与PyTorch逻辑NCHW展平一致。将帅、兵卒共用直线平面；车炮同位移共用编码，是否隔子、塞马腿、塞象眼、越河、出宫及将帅照面由规则引擎判定。4500个槽包含越界填充槽，合法掩码在推理softmax前排除，搜索只展开合法动作，回放拒绝非法目标；训练交叉熵仍使用完整4500 logits。

黑方规范朝向将起点旋转180度，并将位移映射到反方向平面。该置换是自身的逆，输入规范化、神经推理输出还原、独立推理进程和回放标签使用同一映射；不能直接倒序4500个元素。

新动作版本为 `absolute-displacement-50x10x9-v1`，配置指纹另含策略头结构与50平面信息，旧8100 checkpoint和回放明确不兼容。此次未实现旧塔/value的显式权重导入或旧回放重编码；不能从旧learner直接完整续训。参数量减少不等于已经实测训练加速，新头的GPU实测留待独立运行。
