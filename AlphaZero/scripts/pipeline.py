"""AlphaZero 训练管道 — 两阶段冷启动 -> 自举循环。

Phase 1 (warmup): MCTS + HeuristicEvaluator -> 数据 -> 训练网络 value
Phase 2 (main):   MCTS + NeuralEvaluator  -> 数据 -> 训练 -> 循环

用法:
    python -m AlphaZero.scripts.pipeline --warmup 10 --iterations 50
"""
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))

from AlphaZero.model import AlphaZeroNet, NeuralEvaluator
from AlphaZero.search import MCTS, HeuristicEvaluator
from AlphaZero.train import AlphaZeroConfig, ReplayBuffer, Trainer
from AlphaZero.train.self_play import SelfPlayGame


def run_iteration(model, config, replay, iteration, phase='neural'):
    """一次迭代: 自对弈 + 训练。

    Args:
        model:  AlphaZeroNet
        config: AlphaZeroConfig
        replay: ReplayBuffer
        iteration: int
        phase: 'heuristic' | 'neural'
    """
    use_heuristic = (phase == 'heuristic')
    tag = 'HEURISTIC' if use_heuristic else 'NEURAL'

    # -- Self-play --
    print(f"\n{'='*60}")
    print(f"Iter {iteration} [{tag}] {config.games_per_iteration} games x {config.num_simulations} sims")
    print(f"{'='*60}")

    if use_heuristic:
        evaluator = HeuristicEvaluator(seed=42 + iteration)
    else:
        evaluator = NeuralEvaluator(model)

    mcts = MCTS(evaluator=evaluator, num_simulations=config.num_simulations,
                c_puct=config.c_puct, dirichlet_alpha=config.dirichlet_alpha,
                dirichlet_epsilon=config.dirichlet_epsilon)

    red = black = draw = 0
    total_moves = new_samples = 0
    t0 = time.perf_counter()

    for i in range(config.games_per_iteration):
        game = SelfPlayGame(mcts, config, game_id=iteration * 1000 + i)
        while not game.is_terminal():
            game.step()
        r = game.result()
        if r > 0.5: red += 1
        elif r < -0.5: black += 1
        else: draw += 1
        total_moves += game.state.move_count
        for s, p, v in game.get_training_data():
            replay.add(s, p, v)
        new_samples += game.state.move_count

        if (i + 1) % max(1, config.games_per_iteration // 5) == 0:
            t = time.perf_counter() - t0
            term = red + black
            print(f"  G{i+1:3d}/{config.games_per_iteration}: "
                  f"R={red} B={black} D={draw} ({term}/{i+1} term) [{t:.0f}s]")

    tsp = time.perf_counter() - t0
    tr = (red + black) / config.games_per_iteration * 100
    print(f"  Done: {tsp:.0f}s ({tsp/config.games_per_iteration:.1f}s/game) | "
          f"R={red} B={black} D={draw} term={tr:.0f}% | +{new_samples} samples")

    # -- Train --
    print(f"  Train {config.epochs_per_iteration} epochs...")
    trainer = Trainer(config)
    trainer.model.load_state_dict(model.state_dict())
    stats = trainer.train(replay, epochs=config.epochs_per_iteration)
    model.load_state_dict(trainer.model.state_dict())

    return {
        'iteration': iteration, 'phase': tag,
        'red': red, 'black': black, 'draw': draw,
        'term_rate': tr, 'total_moves': total_moves,
        'new_samples': new_samples, 'buffer_size': len(replay),
        'elapsed_sp': tsp, **stats,
    }


def pipeline(config, max_iterations=50, warmup_iterations=10,
             checkpoint=None, resume=None, output_dir="AlphaZero/checkpoints"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = AlphaZeroNet(num_blocks=config.num_blocks,
                          num_filters=config.num_filters)
    start = 0

    if resume:
        ck = torch.load(resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        start = ck.get('iteration', 0) + 1
        print(f"Resumed: {resume} (iteration {ck.get('iteration','?')})")
    elif checkpoint:
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        print(f"Loaded: {checkpoint}")

    replay = ReplayBuffer(max_size=config.replay_buffer_size)
    history = []

    print(f"\nModel: {model.count_parameters():,} params")
    print(f"Warmup: {warmup_iterations} iters (heuristic), "
          f"Main: {max_iterations - warmup_iterations} iters (neural)")
    print(f"Total:  {config.games_per_iteration} games x {config.num_simulations} sims x {max_iterations} iters")

    total_start = time.perf_counter()

    for it in range(start, start + max_iterations):
        ts = datetime.now().strftime("%H:%M:%S")
        phase = 'heuristic' if it < warmup_iterations else 'neural'

        print(f"\n{'#'*60}")
        print(f"# Iteration {it+1}/{start+max_iterations} [{phase.upper()}] [{ts}]")
        print(f"{'#'*60}")

        st = run_iteration(model, config, replay, it, phase=phase)
        st['timestamp'] = ts
        history.append(st)

        # Save checkpoint
        ckpt_path = out / f"iter{it:03d}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'iteration': it, 'config': config.to_dict(),
            'stats': st, 'history': history,
        }, str(ckpt_path))

        # Latest
        torch.save({
            'model_state_dict': model.state_dict(),
            'iteration': it, 'config': config.to_dict(),
            'stats': st,
        }, str(out / "latest.pt"))

        # Trend
        if len(history) >= 2:
            prev = history[-2]['term_rate']
            curr = st['term_rate']
            arrow = 'UP' if curr > prev else ('DOWN' if curr < prev else 'FLAT')
            print(f"  Term: {prev:.0f}% -> {curr:.0f}% [{arrow}]")
            print(f"  Loss: {history[-2]['avg_loss']:.4f} -> {st['avg_loss']:.4f}")

        # Phase transition
        if it == warmup_iterations - 1 and warmup_iterations > 0:
            print(f"\n  >>> Switching from HEURISTIC to NEURAL evaluator <<<")

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'='*60}")
    print(f"PIPELINE DONE: {len(history)} iterations, {total_elapsed/3600:.1f}h")
    if history:
        print(f"  Final term rate: {history[-1]['term_rate']:.0f}%")
        print(f"  Final loss:      {history[-1]['avg_loss']:.4f}")
    print(f"  Checkpoint:      {out}/latest.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AlphaZero training pipeline")
    p.add_argument("--checkpoint", type=str, default=None, help="Cold-start checkpoint")
    p.add_argument("--resume", type=str, default=None, help="Resume checkpoint")
    p.add_argument("--output", type=str, default="AlphaZero/checkpoints")
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10, help="Heuristic warmup iterations")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--sims", type=int, default=50)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--filters", type=int, default=96)
    p.add_argument("--buffer", type=int, default=100000)
    p.add_argument("--max-moves", type=int, default=80)
    args = p.parse_args()

    config = AlphaZeroConfig(
        num_blocks=args.blocks, num_filters=args.filters,
        num_simulations=args.sims, games_per_iteration=args.games,
        epochs_per_iteration=args.epochs, batch_size=args.batch,
        replay_buffer_size=args.buffer, max_game_length=args.max_moves,
    )

    pipeline(config=config, max_iterations=args.iterations,
             warmup_iterations=args.warmup, checkpoint=args.checkpoint,
             resume=args.resume, output_dir=args.output)
