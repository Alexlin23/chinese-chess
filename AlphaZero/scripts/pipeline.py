"""AlphaZero 训练管道 — 全并行 + 实时监控

用法:
    python -m AlphaZero.scripts.pipeline --config config/default.yaml
    python -m AlphaZero.scripts.pipeline --config config/default.yaml --workers 16
"""
import sys
import os
import time
import argparse
from pathlib import Path
from datetime import datetime
import multiprocessing as mp
from multiprocessing import Process, Queue, Event

import numpy as np
import torch

_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))

from AlphaZero.model import PolicyWDLEncoder, NeuralEvaluator
from AlphaZero.search import MCTS
from AlphaZero.train import AlphaZeroConfig, ReplayBuffer, Trainer
from AlphaZero.train.arena import Arena
from AlphaZero.train.inference_server import InferenceServer
from AlphaZero.train.self_play_worker import self_play_worker_fn
from AlphaZero.train.monitor import monitor
from AlphaZero.engine import GameState, RED, BLACK


def parallel_self_play(config: AlphaZeroConfig,
                       model_path: str,
                       num_workers: int,
                       games_per_worker: int,
                       device: str = 'cuda',
                       verbose: bool = True) -> ReplayBuffer:
    """多进程并行自对弈"""
    server = InferenceServer(
        model_path=model_path,
        device=device,
        batch_size=config.inference_batch_size,
        max_wait_ms=config.inference_wait_ms,
        num_workers=num_workers,
    )
    server_pid = server.start()
    if verbose:
        print(f"  InferenceServer 已启动 (PID={server_pid}, device={device})")

    result_queue = Queue(maxsize=200000)
    stop_event = Event()
    processes = []

    for i in range(num_workers):
        p = Process(
            target=self_play_worker_fn,
            args=(i, config, games_per_worker,
                  server.request_queue, server.response_queues[i],
                  result_queue, stop_event),
            daemon=True,
        )
        p.start()
        processes.append(p)

    if verbose:
        print(f"  {num_workers} 个 SelfPlayWorker 已启动")

    replay = ReplayBuffer(max_size=config.replay_buffer_size)
    workers_done = 0
    total_stats = {'red': 0, 'black': 0, 'draw': 0, 'moves': 0, 'games': 0}
    t_start = time.perf_counter()

    while workers_done < num_workers:
        try:
            item = result_queue.get(timeout=300)  # 5分钟超时
        except:
            # 检查worker是否还活着
            alive = sum(1 for p in processes if p.is_alive())
            if alive == 0 and workers_done < num_workers:
                print(f"  ⚠ 所有Worker已退出，但只有 {workers_done}/{num_workers} 报告完成")
                break
            continue

        if (isinstance(item, tuple) and len(item) == 3
                and isinstance(item[0], str) and item[0] == 'DONE'):
            _, stats, _ = item
            workers_done += 1
            for k in ['red', 'black', 'draw', 'moves', 'games']:
                total_stats[k] += stats[k]
            if verbose:
                elapsed = time.perf_counter() - t_start
                print(f"  Worker {stats['worker_id']} 完成: "
                      f"{stats['games']}局 {stats['moves']}步 "
                      f"({stats['elapsed']:.0f}s) "
                      f"[{workers_done}/{num_workers}]")
        else:
            state_enc, policy, wdl = item
            replay.add(state_enc, policy, wdl)

        # 更新监控
        monitor.update(
            phase='self_play',
            workers=num_workers,
            games_done=total_stats['games'],
            games_total=games_per_worker * num_workers,
            samples=len(replay),
        )

    stop_event.set()
    server.stop()
    for p in processes:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    elapsed = time.perf_counter() - t_start
    if verbose:
        total_games = total_stats['games']
        total_moves = total_stats['moves']
        print(f"  自对弈完成: {total_games}局 {total_moves}步 "
              f"{elapsed:.0f}s ({elapsed/max(total_games,1):.1f}s/局)")
        print(f"  R={total_stats['red']} B={total_stats['black']} "
              f"D={total_stats['draw']} | {len(replay)} 样本")

    return replay


def pipeline(config: AlphaZeroConfig,
             max_iterations: int = None,
             checkpoint: str = None,
             resume: str = None,
             output_dir: str = None,
             enable_monitor: bool = True):
    """完整训练管道（全并行）"""
    # 从config读取max_iterations，或使用默认值100
    if max_iterations is None:
        # 尝试从config获取（需要在config中添加这个字段）
        max_iterations = getattr(config, 'max_iterations', 100)
    output_dir = output_dir or config.checkpoint_dir
    num_workers = config.num_workers or os.cpu_count() or 4

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = PolicyWDLEncoder(
        num_blocks=config.num_blocks,
        num_filters=config.num_filters,
    ).to(device)
    start = 0

    if resume:
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        start = ck.get('iteration', 0) + 1
        print(f"恢复训练: {resume} (iteration {start})")
    elif checkpoint:
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        print(f"加载 checkpoint: {checkpoint}")

    replay = ReplayBuffer(max_size=config.replay_buffer_size)
    trainer = Trainer(config, device)
    if resume or checkpoint:
        trainer.model.load_state_dict(model.state_dict())

    arena = Arena(config, device)
    best_model_path = str(out / "best.pt")

    if not Path(best_model_path).exists():
        torch.save({'model_state_dict': model.state_dict(),
                    'config': config.to_dict()}, best_model_path)

    # 启动监控
    if enable_monitor:
        monitor.interval = config.monitor_interval
        monitor.update(
            iteration=start,
            max_iterations=start + max_iterations,
            workers=num_workers,
            promoted=0,
        )
        monitor.start()

    print(f"\n{'='*60}")
    print(f"AlphaZero 并行训练管道")
    print(f"{'='*60}")
    print(f"模型: {model.count_parameters():,} 参数")
    print(f"设备: {device}")
    print(f"Workers: {num_workers}")
    print(f"Iterations: {max_iterations}")
    print(f"每轮: {config.games_per_iteration} 局自对弈 + {config.epochs_per_iteration} epochs")
    print(f"Arena: {config.arena_games} 局, 晋升阈值 {config.promotion_score_rate}")
    print(f"MCTS: {config.num_simulations} 模拟, c_puct={config.c_puct}")
    print(f"训练: batch={config.batch_size}, lr={config.learning_rate}")
    print(f"监控: {'开启' if enable_monitor else '关闭'}")
    print(f"{'='*60}\n")

    total_start = time.perf_counter()
    history = []
    total_promoted = 0

    for it in range(start, start + max_iterations):
        ts = datetime.now().strftime("%H:%M:%S")
        model_path = str(out / "latest.pt")

        # 保存当前模型供 self-play 使用
        torch.save({'model_state_dict': model.state_dict(),
                    'config': config.to_dict()}, model_path)

        print(f"\n[{ts}] Iteration {it+1}/{start+max_iterations}")
        print(f"{'─'*40}")

        # 保存旧模型
        model_old_path = str(out / f"iter{it:03d}_old.pt")
        torch.save({'model_state_dict': model.state_dict()}, model_old_path)

        # 更新监控
        monitor.update(iteration=it+1, phase='self_play')

        # 并行自对弈
        games_per_worker = max(1, config.games_per_iteration // num_workers)
        replay = parallel_self_play(
            config=config,
            model_path=model_path,
            num_workers=num_workers,
            games_per_worker=games_per_worker,
            device=device,
        )

        # 更新监控
        monitor.update(phase='training', loss=None)

        # 训练
        stats = trainer.train(replay, epochs=config.epochs_per_iteration)
        model.load_state_dict(trainer.model.state_dict())

        # 更新监控
        monitor.update(loss=stats.get('avg_loss', 0))

        # 保存候选模型
        candidate_path = str(out / f"iter{it:03d}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'iteration': it, 'config': config.to_dict(),
            'stats': stats,
        }, candidate_path)

        # Arena 评估（第一轮直接晋升）
        if it == start:
            # 第一轮直接晋升，不做 Arena
            print(f"  ✓ 第一轮直接晋升")
            total_promoted += 1
            arena_result = {'wins': 0, 'losses': 0, 'draws': 0,
                           'score_rate': 1.0, 'should_promote': True}
            torch.save({'model_state_dict': model.state_dict(),
                        'iteration': it, 'config': config.to_dict()},
                       best_model_path)
        else:
            print(f"  Arena: {config.arena_games} 局 vs 旧模型...")
            monitor.update(phase='arena', arena_w=0, arena_l=0, arena_d=0)

            arena_result = arena.evaluate(candidate_path, model_old_path)
            print(f"  结果: W={arena_result['wins']} L={arena_result['losses']} "
                  f"D={arena_result['draws']} score={arena_result['score_rate']:.1%}")

            # 更新监控
            monitor.update(
                arena_w=arena_result['wins'],
                arena_l=arena_result['losses'],
                arena_d=arena_result['draws'],
            )

            # 晋升条件：score >= 阈值 或 全平局（新模型不比旧模型差）
            should_promote = (
                arena_result['should_promote'] or
                (arena_result['losses'] == 0 and arena_result['wins'] == 0)
            )
            if should_promote:
                print(f"  ✓ 晋升")
                total_promoted += 1
                torch.save({'model_state_dict': model.state_dict(),
                            'iteration': it, 'config': config.to_dict()},
                           best_model_path)
            else:
                print(f"  ✗ 回滚")
                ck = torch.load(model_old_path, map_location=device)
                model.load_state_dict(ck['model_state_dict'])
                trainer.model.load_state_dict(ck['model_state_dict'])

        # 更新监控
        monitor.update(promoted=total_promoted)

        Path(model_old_path).unlink(missing_ok=True)

        history.append({
            'iteration': it, 'buffer_size': len(replay),
            'arena': arena_result, **stats
        })

        torch.save({'model_state_dict': model.state_dict(),
                    'iteration': it, 'config': config.to_dict()},
                   str(out / "latest.pt"))

    # 停止监控
    monitor.stop()

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'='*60}")
    print(f"训练完成: {len(history)} 轮, {total_elapsed/3600:.1f} 小时")
    print(f"晋升: {total_promoted}/{len(history)}")
    print(f"最佳模型: {best_model_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    p = argparse.ArgumentParser(description="AlphaZero 并行训练管道")
    p.add_argument("--config", type=str, default="config/default.yaml",
                   help="YAML 配置文件路径")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--workers", type=int, default=None, help="覆盖配置文件中的 workers")
    p.add_argument("--no-monitor", action="store_true", help="禁用实时监控")
    args = p.parse_args()

    # 从 YAML 加载配置
    config = AlphaZeroConfig.from_yaml(args.config)

    # 命令行覆盖
    if args.workers is not None:
        config.num_workers = args.workers

    print(f"配置文件: {args.config}")

    pipeline(config=config, max_iterations=args.iterations,
             checkpoint=args.checkpoint, resume=args.resume,
             output_dir=args.output,
             enable_monitor=not args.no_monitor)
