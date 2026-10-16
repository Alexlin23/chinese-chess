"""Paired-color match runner for independent strength checks."""
from dataclasses import dataclass
import argparse
from math import sqrt, log
from typing import Callable
import numpy as np

from ..engine import GameState, RED, BLACK
from ..engine.heuristic import heuristic_move

Agent = Callable[[GameState, np.random.Generator], object]


def random_agent(state: GameState, rng: np.random.Generator):
    moves = state.legal_moves()
    return moves[int(rng.integers(len(moves)))] if moves else None


def heuristic_agent(state: GameState, rng: np.random.Generator):
    return heuristic_move(state, rng)


@dataclass
class MatchReport:
    wins: int
    losses: int
    draws: int
    score_rate: float
    score_ci95: tuple[float, float]
    by_color: dict


def evaluate_agents(candidate: Agent, baseline: Agent, num_games: int,
                    max_ply: int = 512, seed: int = 0, on_game=None) -> MatchReport:
    """Run paired colors with equal game budgets and report uncertainty."""
    if num_games < 2 or num_games % 2:
        raise ValueError("num_games must be a positive even number")
    rng = np.random.default_rng(seed)
    scores = []
    by_color = {'red': {'wins': 0, 'losses': 0, 'draws': 0},
                'black': {'wins': 0, 'losses': 0, 'draws': 0}}
    for game in range(num_games):
        candidate_is_red = game % 2 == 0
        state = GameState.new_game(max_ply=max_ply)
        while not state.is_terminal():
            agent = candidate if state.turn == candidate_is_red else baseline
            move = agent(state, rng)
            if move is None:
                raise ValueError("agent returned no move in a nonterminal position")
            if not state.is_legal(move):
                raise ValueError(f"agent returned illegal move: {move}")
            state = state.apply(move)
        result = state.game_result()
        color = 'red' if candidate_is_red else 'black'
        if result is None or result.is_draw:
            outcome = 'draws'
            score = 0.5
        elif result.winner == (RED if candidate_is_red else BLACK):
            outcome = 'wins'
            score = 1.0
        else:
            outcome = 'losses'
            score = 0.0
        by_color[color][outcome] += 1
        scores.append(score)
        if on_game is not None:
            on_game(dict(game=game+1, color=color, outcome=outcome, plies=state.move_count,
                         reason=result.reason, score=score))
    scores = np.asarray(scores, dtype=np.float64)
    mean = float(scores.mean())
    # Bounded-score Hoeffding interval, assuming independent color pairs.
    # A fixed deterministic starting position supplies no population sample:
    # retain [0,1] instead of manufacturing confidence from repeated games.
    margin = 1.0
    if getattr(candidate, "stochastic", False) or getattr(baseline, "stochastic", False):
        margin = sqrt(log(40.0) / (2 * (num_games // 2)))
    wins = sum(group['wins'] for group in by_color.values())
    losses = sum(group['losses'] for group in by_color.values())
    draws = sum(group['draws'] for group in by_color.values())
    return MatchReport(wins, losses, draws, mean,
                       (max(0.0, mean - margin), min(1.0, mean + margin)),
                       by_color)


def checkpoint_agent(path: str, simulations: int = 64,
                     device: str = 'cpu') -> Agent:
    """Load a model only when a trained checkpoint is available."""
    import torch
    from ..model import PolicyWDLEncoder, NeuralEvaluator
    from ..search import MCTS

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    model = PolicyWDLEncoder(
        num_blocks=config.get('num_blocks', 4),
        num_filters=config.get('num_filters', 64)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    search = MCTS(NeuralEvaluator(model, device),
                  num_simulations=simulations,
                  c_puct=config.get('c_puct', 1.25),
                  pb_c_base=config.get('pb_c_base', 19652.0),
                  batch_size=config.get('mcts_batch_size', 8),
                  virtual_loss=config.get('virtual_loss', 3.0))

    def select(state, rng):
        move, _ = search.select_move(
            state, temperature=0.0, add_exploration_noise=False)
        return move

    return select


def main():
    parser = argparse.ArgumentParser(description="Paired-color strength benchmark")
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--baseline', choices=('random', 'heuristic'),
                        required=True)
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--simulations', type=int, default=64)
    parser.add_argument('--max-ply', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    candidate = checkpoint_agent(args.checkpoint, args.simulations, args.device)
    baseline = random_agent if args.baseline == 'random' else heuristic_agent
    report = evaluate_agents(candidate, baseline, args.games,
                             args.max_ply, args.seed)
    print(report)


if __name__ == '__main__':
    main()
