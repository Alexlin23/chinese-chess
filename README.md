# 中国象棋 AlphaZero

目标是在中国象棋上复现 AlphaZero 的学习与搜索方法，只缩减网络规模，并做硬件执行优化。**当前是重构中的开发版，还不能称为 100% 复现完成。** 主要缺口是完整中国象棋长捉责任裁定、独立规则验收及长期学习验证。唯一主审查与修复记录见 [ALPHAZERO_PAPER_REVIEW.md](ALPHAZERO_PAPER_REVIEW.md)，第 20 节记录收到 Data S1 后的核对与修正。

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

当前持续训练目录：`AlphaZero/runs/xiangqi-v3-20261015`。在 PowerShell 可管理该 run：

~~~powershell
.\scripts\training.ps1 -Action Status
.\scripts\training.ps1 -Action Stop
.\scripts\training.ps1 -Action Resume
~~~

Status 显示实际进程和最近更新。Stop 只中断对应 run；Resume 保留完整回放并从 learner checkpoint 继续。训练日志位于该目录。新代码每 100 步写恢复状态，每 1000 步向自对弈发布模型，两者分别控制。

停止后，可以修改 `config/local-train.yaml` 中的 `num_simulations`（每手搜索次数，例如 800 改为 400），再执行 Resume。模型、SGD 动量、累计训练步数和已有回放继续保留；新启动的自对弈使用修改后的次数，尚未完成的旧对局不恢复。参数必须为正整数。

`learner.pt` 保存最近恢复点的训练状态和配置，`config.json` 保存本次启动配置；`metrics.jsonl` 追加保存每次启动的完整配置、恢复步数与 `config_changes`（旧值/新值），不会用新配置覆盖历史记录。新生成棋谱的 trace 同样记录搜索次数。学习率、模型结构等其他续训限制保持原有检查。

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
