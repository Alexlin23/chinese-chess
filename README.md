# 中国象棋 AlphaZero

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
.\scripts\training.ps1 -Action Resume -Simulations 400 -PublishEvery 8000 -ReplayPositions 100000 -MaxGamePly 300 -Publications 10
```

| 参数 / YAML字段 | 含义 |
|---|---|
| Simulations / num_simulations | 每手MCTS模拟次数 |
| PublishEvery / checkpoint_interval | 每多少次SGD更新发布模型 |
| ReplayPositions / replay_window_positions | 保留的最近局面上限，按整局淘汰；0表示只使用对局数窗口 |
| mirror_augmentation（YAML开关） | 训练取样时随机左右镜像，默认关闭，本机配置开启 |
| MaxGamePly / max_game_ply | 双方合计的强制和棋步数上限 |
| Publications | 本次继续完成多少次新发布后停止；启动重导出不计数 |

YAML修改持久有效，命令行覆盖只影响本次启动。运行中的参数不热改。发布按全局SGD步数的间隔边界触发；剩余更新预算不足时拒绝启动。

搜索次数、发布间隔和回放上限调整保留模型、SGD动量、步数、随机状态及兼容回放。改变强制和棋上限时自动校验棋谱、重建输入的步数归一化：缩小时剔除超过新上限的整局，扩大时剔除旧上限强制判和的整局。历史策略标签仍是当时的访问数，不会重新搜索。转换在原目录保留 `.resume-backup-*` 备份及恢复日志，失败可回滚；备份会占空间，不会自动删除。网络结构或规则版本不兼容时仍拒绝直接混用。

### 左右镜像增强

本机配置为 `replay_window_positions: 50000`、`mirror_augmentation: true`。回放只保存最近至多5万个原始局面，按完整对局淘汰；每个SGD样本以50%概率选择原局面或左右镜像，相当于从至多“5万原局面＋5万镜像”的虚拟池取样。对称局面可能与自身相同，因此不是10万个独立原始局面。

镜像同时反射8帧历史和策略动作的起点、终点列，胜负价值及全局状态特征不变。每次SGD仍抽4096项，不强制原图和镜像成对，不增加数据库副本。它是中国象棋训练扩展；关闭开关不额外消耗随机数。启用后的抽样随机状态由现有NumPy存档恢复。

已有相同模型/规则的learner可以在原目录续训，参数变化记录到config_changes。新回放上限在下一次Resume时生效，合并代码本身不裁剪现有数据或启动训练。actor发布省略learner专用开关，保持现有对弈服务可读取。

## 浏览器

- [8765训练观战](http://localhost:8765)：训练进程提供，有多少盘同时下就显示多少盘。显示训练进度、损失、CPU和内存，保留原有样式；训练停止后服务随之关闭。
- [8767人机对弈](http://localhost:8767)：独立运行，训练暂停后仍能下棋。每局固定模型快照，开始新局时检查最新发布模型。

启动人机对弈，默认读取同一份本机配置及其latest.pt：

```powershell
wsl --cd D:\project\Alphazero --exec /home/lin/.venvs/xiangqi-alphazero/bin/python -m AlphaZero.scripts.play_server --device cuda
```

默认CPU；明确指定 `--device cuda` 才使用GPU。可用 `--config` 选择其他配置，`--simulations` 临时覆盖搜索次数，`--fixed-snapshot` 禁止新局重载。没有旧版独立Web后端或第二套规则数据库。

## 可关闭的性能计时

保留此前要求的五模块粗粒度性能区，不包含推理阶段、内存池、停顿诊断或GPU管理查询。

```powershell
.\scripts\training.ps1 -Action Resume -Performance
```

不传Performance则关闭。默认每5秒汇总、每64批/更新抽样CUDA Event，复用已有同步点，不逐模拟写日志。MCTS本地时间为整手墙钟减同步神经调用的估算；请求RTT包括推理、排队和进程通信。并发模块时间不能直接相加，近似P95不能冒充精确分位数。

保留正常异常传播与请求超时保护：推理失败会终止训练并在标准错误中报告，不用伪造结果继续，不另建阶段诊断日志。界面不展示未实际采集的GPU利用率和整卡显存。

## 模型与训练设置

本机当前配置：19个残差块、128通道、131输入通道、平铺8100动作头，32盘/8个CPU工作进程，BF16、channels-last、跨对局合批及CUDA Graph。400次搜索、8000步发布、300步上限、至多5万原始回放局面并开启左右镜像增强；每次SGD抽4096项，可重复抽样。论文参考参数位于 `config/default.yaml`，测试配置与实际训练目录独立。

Policy交叉熵加标量tanh value的结果MSE；SGD momentum=0.9、初始LR=0.2，在100k/300k/500k更新后各降10倍，预算700k。Data S1正则项为c/2乘参数平方和，weight_decay=1e-4。默认损失按batch均值；Data S1字面求和存在尚未解决的数值/来源差异，不能声称已100%复现，详情见主审查报告。

底层颜色由player_id统一转换。历史重复、自然限着将军计数等状态由同一GameState传递给训练、搜索和人机对弈。自然限着按轮到行棋的一方自动提出处理；将死、困毙及已实现的重复裁定可更早终局，整局上限与连续不吃子上限是两回事。完整国规仍待验收。

## 评估与开发

评估默认选择当前配置目录中的latest.pt，结果写在同一目录：

```powershell
wsl --cd D:\project\Alphazero --exec /home/lin/.venvs/xiangqi-alphazero/bin/python -m AlphaZero.scripts.evaluate_model --baseline random --games 8 --simulations 64
```

少量对随机走法的结果不证明强棋力。测试使用 `requirements-dev.txt`，运行环境依赖为 `requirements.txt`；重建WSL环境用 `scripts/setup_wsl.sh`。诊断、旧后端和对应依赖已移除，历史测量资料只作证据，不作为启动入口。
