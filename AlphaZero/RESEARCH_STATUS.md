# AlphaZero 改造与验收状态

基线：`main` 的 `c555249`。本文件记录已经得到的证据，也列出尚未验证的目标。配置中的 4×64、4 个 Worker、64 次模拟和 batch 64 是起始值，不代表最佳值。

| 阶段 | 当前状态 | 已有证据 | 尚缺的验收 |
| --- | --- | --- | --- |
| S0 正确性 | 部分完成 | MCTS 视角、虚拟损失、重复叶节点和无噪声评估的固定测试；20 个可达局面的 Web/训练走法对照 | 完整长将长捉规则版本、更多固定残局和独立规则实现 |
| S1 内存与故障 | 部分完成 | 90 字节棋盘、稀疏策略、有界队列、经验池容量和存取往返测试；推理失败不再伪造策略 | 多进程崩溃注入与长时间内存峰值测量 |
| S2 性能 | 未验收 | 小模型起始配置、跨 Worker 合批和逐批训练代码 | CPU/GPU 分项测时、吞吐量比较、24 小时无 OOM/卡死测试 |
| S3 恢复 | 部分完成 | 训练检查点包括模型、优化器、调度器、轮次和随机状态；对局数据在训练前原子落盘 | 训练中断与重新启动的一致性实测 |
| S4 棋力 | 部分完成 | 交替执红黑的随机/启发式对手评估工具，报告胜负和描述性区间 | 实际模型比赛、皮卡鱼对照、足够样本的结论 |
| S5 模型研究 | 未开始 | 4×64 起始配置 | 8×128、轻量策略头、关系特征模型的等预算实验 |

CPU 测试：`python -m pytest AlphaZero/tests -q`。语法检查：`python -m compileall -q AlphaZero`。

示例棋力基准（在安装 PyTorch 并具备模型检查点的环境运行）：

```bash
python -m AlphaZero.train.benchmark --checkpoint AlphaZero/checkpoints/best.pt --baseline random --games 100 --device cuda
python -m AlphaZero.train.benchmark --checkpoint AlphaZero/checkpoints/best.pt --baseline heuristic --games 100 --device cuda
```

请分别记录设备、配置、随机种子、模型提交号、每局思考预算、总耗时及资源峰值。置信区间跨越 50% 时，不宣称模型更强。

本次执行环境未安装 PyTorch，Windows/Ubuntu WSL 未检测到可用 GPU 工具。因此尚未进行训练集成、GPU 吞吐或 24 小时压力测试。
