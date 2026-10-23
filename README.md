# 中国象棋 AlphaZero

当前为 `alphazero_Policy` 分支：策略头输出50个动作平面（4500 logits）。已同步main的规则、镜像增强、按新增局面更新、SGD计数调整、性能区及人机噪声设置。模型和回放规格与main的8100头不同，不能直接混用其learner、actor或Replay。本分支配置继续使用原有独立目录 `AlphaZero/runs/xiangqi-policy-20261016` 和监控8768；完整实现说明见审查报告第42节。

训练、搜索和浏览器对弈共用 `AlphaZero.engine.GameState`。当前是开发版：完整国规长捉责任、例外裁定及长期棋力验证尚未完成。唯一审查和历史测量记录见 [ALPHAZERO_PAPER_REVIEW.md](ALPHAZERO_PAPER_REVIEW.md)。

## 统一配置与数据目录

日常使用 `config/local-train.yaml`。它的 `checkpoint_dir` 是训练、对弈和评估共同使用的目录来源，不再在不同脚本里写死历史模型或数据路径。

| 文件 | 用途 |
|---|---|
| learner.pt | 最近恢复点：模型、SGD动量、全局步数和随机状态 |
| latest.pt | 最近发布给自博弈和人机对弈的模型 |
| replay.sqlite | 最近的完整自博弈及训练局面 |
| config.json / metrics.jsonl | 本次配置、训练指标和参数变化历史 |
| performance.json / performance.jsonl | 仅开启性能计时时产生的汇总 |

正常训练覆盖learner和latest，不为每次发布无限归档。旧存档与棋谱文件不会因代码清理而删除，也不会被自动选中。需要使用其他实验时，显式选择配置或 `--run-dir` / `--checkpoint`。

## 启动、暂停与续训

本机WSL环境：`/home/lin/.venvs/xiangqi-alphazero`。在项目根目录PowerShell中：

```powershell
.\scripts\training.ps1 -Action Status
.\scripts\training.ps1 -Action Stop
.\scripts\training.ps1 -Action Resume
```

首次空目录使用Start；已有learner必须使用Resume。Stop只中断对应训练进程，不停止人机对弈服务。恢复使用最近一次完整保存的learner；停止前尚未保存的尾部更新不会保留。

停止后直接修改配置，再Resume，不需要新Git分支或新运行目录。也可临时覆盖参数：

```powershell
    .\scripts\training.ps1 -Action Resume -Simulations 400 -PublishEvery 4000 -ReplayPositions 50000 -MaxGamePly 180 -NoiseDepth 3 -Publications 10
```

| 参数 / YAML字段 | 含义 |
|---|---|
| Simulations / num_simulations | 每手MCTS模拟次数 |
| PublishEvery / checkpoint_interval | 每多少次SGD更新发布模型 |
| ReplayPositions / replay_window_positions | 保留的最近局面上限，按整局淘汰；0表示只使用对局数窗口 |
| mirror_augmentation（YAML开关） | 训练取样时随机左右镜像，默认关闭，本机配置开启 |
| MaxGamePly / max_game_ply | 双方合计的强制和棋步数上限 |
| NoiseDepth / search_noise_depth | 自博弈搜索中加噪声的决策层数，包含根节点；0关闭，1为论文基线 |
| NewPositionsPerUpdate / new_positions_per_update | 每新增多少原始局面允许一次SGD更新；0表示原来的连续更新 |
| Publications | 本次继续完成多少次新发布后停止；启动重导出不计数 |

YAML修改持久有效，命令行覆盖只影响本次启动。运行中的参数不热改。发布按全局SGD步数的间隔边界触发；剩余更新预算不足时拒绝启动。

搜索次数、发布间隔和回放上限调整保留模型、SGD动量、步数、随机状态及兼容回放。改变强制和棋上限时自动校验棋谱、重建输入的步数归一化：缩小时从数据库删除超过新上限的整局，扩大时删除旧上限强制判和的整局。历史策略标签仍是当时的访问数，不会重新搜索。转换仅在执行期间保留同目录的临时回滚文件，成功提交或完成回滚后自动删除；转换记录保留在 `parameter-changes.jsonl`，不长期备份被淘汰的棋局。网络结构或规则版本不兼容时仍拒绝直接混用。

`search_noise_depth: 3` 在每次自博弈搜索的根节点、走完一手后的节点、走完两手后的节点混入Dirichlet噪声，之后的层不加。每个已展开节点在一次搜索内只混入一次噪声，不随访问次数叠加；复用子树时从原网络先验重新混合。它指搜索树的前三层，不是整盘棋开头三手。alpha=0.3、epsilon=0.25及前30手温度采样保持独立。此项是用户指定的实验扩展，论文参考配置仍设为1。旧存档缺少该字段时按1恢复，改变层数会记录在续训配置变化中；不清空Replay或重置SGD步数、动量和学习率。人机对弈默认关闭噪声；通过服务参数显式开启时，可共用此层数设置。

### 左右镜像增强

本机配置为 `replay_window_positions: 50000`、`mirror_augmentation: true`。回放只保存最近至多5万个原始局面，按完整对局淘汰；每个SGD样本以50%概率选择原局面或左右镜像，相当于从至多“5万原局面＋5万镜像”的虚拟池取样。对称局面可能与自身相同，因此不是10万个独立原始局面。

镜像同时反射8帧历史和策略动作的起点、终点列，胜负价值及全局状态特征不变。每次SGD仍抽4096项，不强制原图和镜像成对，不增加数据库副本。它是中国象棋训练扩展；关闭开关不额外消耗随机数。启用后的抽样随机状态由现有NumPy存档恢复。

已有相同模型/规则的learner可以在原目录续训，参数变化记录到config_changes。新回放上限在下一次Resume时生效，合并代码本身不裁剪现有数据或启动训练。actor发布省略learner专用开关，保持现有对弈服务可读取。

### 按新增数据触发更新

本机配置为 `new_positions_per_update: 4096`：每入库4096个新的原始自博弈局面，允许一次SGD更新，每次仍从整个当前Replay随机抽4096项并按开关做镜像。并非只训练刚产生的4096项，也不保证每个局面恰好抽中一次。完整对局入库时累计新增量，镜像、重复提交和抽样不增加计数；FIFO淘汰旧局面不会扣减累计量。

首次启用或改变此间隔，从本次启动时的累计入库数开始计算，既有Replay不授予更新额度。间隔不变时，剩余额度随learner的 `extra.update_schedule` 保存；Resume保留余数，也计入检查点之后已提交的对局。未保存的SGD尾部会同模型一起回退。该字段兼容缺少此元数据的旧learner，不改变模型结构、动作标签或Replay格式；actor不携带此训练参数。设为0恢复连续SGD。

新增数据不足时继续自博弈，learner等待。发布间隔独立配置；本机暂设 `checkpoint_interval: 1`，即每次SGD更新后保存并发布，对应每新增4096个原始局面发布一次。若改为4000更新/发布，则对应16384000个新增局面/发布。`max_optimizer_steps: 700000`仍是SGD预算；调整发布间隔时，`max_iterations * checkpoint_interval`也必须覆盖该预算。本次频率限制是本机实验配置，论文参考配置仍使用0。

停止训练后，可在原目录调整SGD日程计数，例如 `wsl --cd . --exec /home/lin/.venvs/xiangqi-alphazero/bin/python -m AlphaZero.scripts.reset_optimizer_step --step 400000`。该操作保留模型权重、SGD动量、RNG、数据和发布快照，只调整learner的日程计数、对应学习率与迭代数，变更记录在learner中。发布模型的版本通过存档内的偏移量保持递增，历史棋谱版本不改写；例如SGD从700000调到400000，模型版本仍为700000，再更新4000次后为704000。学习率和停止预算按调整后的SGD计数计算，因此40万步对应本机现有日程0.002，到50万步后再降至0.0002。此操作不会启动训练，也不会重置运行中的人机对局。

## 浏览器

- [8768训练观战](http://localhost:8768)：训练进程提供，有多少盘同时下就显示多少盘。显示训练进度、损失、CPU和内存，保留原有样式；训练停止后服务随之关闭。
- [8770动作平面对弈](http://localhost:8770)：独立运行，训练暂停后仍能下棋。每局固定模型快照，开始新局时检查最新发布模型。

启动人机对弈，默认读取同一份本机配置及其latest.pt：

```powershell
wsl --cd . --exec /home/lin/.venvs/xiangqi-alphazero/bin/python -m AlphaZero.scripts.play_server --device cuda --port 8770
```

默认CPU；明确指定 `--device cuda` 才使用GPU。可用 `--config` 选择其他配置，`--simulations` 临时覆盖搜索次数，`--fixed-snapshot` 禁止新局重载。没有旧版独立Web后端或第二套规则数据库。

人机对弈默认关闭探索噪声；启动加上 `--exploration-noise` 开启，`--search-noise-depth` 控制搜索树加噪层数，未指定时使用本机配置（当前3层）。α和混合比例取模型配置（当前0.3和0.25）。开启后，新对局重载模型仍保留这些服务参数；选棋温度仍为0，按搜索访问数最大者落子，噪声不保证每次都会走不同的棋。

## 可关闭的性能计时

保留此前要求的五模块粗粒度性能区，不包含推理阶段、内存池、停顿诊断或GPU管理查询。

```powershell
.\scripts\training.ps1 -Action Resume -Performance
```

不传Performance则关闭。默认每5秒汇总、每64批/更新抽样CUDA Event，复用已有同步点，不逐模拟写日志。MCTS本地时间为整手墙钟减同步神经调用的估算；请求RTT包括推理、排队和进程通信。并发模块时间不能直接相加，近似P95不能冒充精确分位数。

保留正常异常传播与请求超时保护：推理失败会终止训练并在标准错误中报告，不用伪造结果继续，不另建阶段诊断日志。界面不展示未实际采集的GPU利用率和整卡显存。

## 模型与训练设置

本机当前配置：19个残差块、128通道、131输入通道、平铺8100动作头，32盘/8个CPU工作进程，BF16、channels-last、跨对局合批及CUDA Graph。400次搜索、4000步发布、180步上限、搜索前三层噪声、至多5万原始回放局面并开启左右镜像增强；每次SGD抽4096项，可重复抽样。论文参考参数位于 `config/default.yaml`，测试配置与实际训练目录独立。

Policy交叉熵加标量tanh value的结果MSE；SGD momentum=0.9、初始LR=0.2，在100k/300k/500k更新后各降10倍，预算700k。Data S1正则项为c/2乘参数平方和，weight_decay=1e-4。默认损失按batch均值；Data S1字面求和存在尚未解决的数值/来源差异，不能声称已100%复现，详情见主审查报告。

底层颜色由player_id统一转换。历史重复、自然限着将军计数等状态由同一GameState传递给训练、搜索和人机对弈。自然限着按轮到行棋的一方自动提出处理；将死、困毙及已实现的重复裁定可更早终局，整局上限与连续不吃子上限是两回事。完整国规仍待验收。

## 评估与开发

评估默认选择当前配置目录中的latest.pt，结果写在同一目录：

```powershell
wsl --cd . --exec /home/lin/.venvs/xiangqi-alphazero/bin/python -m AlphaZero.scripts.evaluate_model --baseline random --games 8 --simulations 64
```

少量对随机走法的结果不证明强棋力。测试使用 `requirements-dev.txt`，运行环境依赖为 `requirements.txt`；重建WSL环境用 `scripts/setup_wsl.sh`。诊断、旧后端和对应依赖已移除，历史测量资料只作证据，不作为启动入口。

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
