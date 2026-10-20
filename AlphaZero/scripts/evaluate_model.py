"""Measure actual games for a trained checkpoint; never changes learner weights."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch
from AlphaZero.model.player import ModelPlayer
from AlphaZero.train.benchmark import evaluate_agents, random_agent, heuristic_agent
from AlphaZero.scripts.paths import DEFAULT_CONFIG, run_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir")
    parser.add_argument("--checkpoint", help="default: configured run's latest.pt")
    parser.add_argument("--output", help="default: evaluation-<baseline>.json beside the selected model")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--baseline", choices=("random", "heuristic"), default="random")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cuda-memory-fraction", type=float, default=.05,
                        help="Bound evaluator allocations while training occupies the GPU")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--wait-for-trained", action="store_true")
    args = parser.parse_args()
    _, directory = run_paths(args.config, args.run_dir)
    checkpoint = Path(args.checkpoint).expanduser() if args.checkpoint else directory / "latest.pt"
    torch.set_num_threads(1)
    if args.device == "cuda":
        if not 0 < args.cuda_memory_fraction <= 1:
            raise ValueError("invalid CUDA memory fraction")
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
    while True:
        try:
            player = ModelPlayer(checkpoint, args.device, args.simulations)
            break
        except (FileNotFoundError, ValueError) as exc:
            if not args.wait_for_trained or (isinstance(exc, ValueError) and "随机初始化" not in str(exc)):
                raise
            print("等待首个已训练模型发布…", flush=True)
            time.sleep(30)
    baseline = random_agent if args.baseline == "random" else heuristic_agent
    started = time.monotonic()
    game_log = []
    outcomes = []
    output = Path(args.output).expanduser() if args.output else checkpoint.parent / f"evaluation-{args.baseline}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    def on_game(outcome):
        outcomes.append(outcome)
        print(json.dumps(outcome, ensure_ascii=False), flush=True)
        progress = output.with_suffix(".progress.json")
        temporary = progress.with_suffix(progress.suffix + ".tmp")
        temporary.write_text(json.dumps(dict(model_step=player.step, baseline=args.baseline,
            target_games=args.games, outcomes=outcomes), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(progress)
    def select(state, rng):
        move, diagnostics = player.choose(state)
        game_log.append(dict(ply=state.move_count, move=repr(move), simulations=diagnostics.get("simulations")))
        return move
    report = evaluate_agents(select, baseline, args.games, max_ply=player.config.max_game_ply,
                             seed=args.seed, on_game=on_game)
    payload = dict(model_step=player.step, checkpoint=str(player.path), baseline=args.baseline,
                   games=args.games, simulations=args.simulations, seed=args.seed,
                   device=args.device, precision=player.evaluator.precision,
                   seconds=time.monotonic()-started, rule=player.config.spec["rule"],
                   report=asdict(report), outcomes=outcomes, candidate_moves=len(game_log),
                   qualification="small development-rule benchmark; no claim of strong playing strength")
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
