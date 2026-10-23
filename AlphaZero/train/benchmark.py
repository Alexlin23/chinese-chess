"""Paired-color match runner for independent strength checks."""
from dataclasses import dataclass
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
    from ..model.player import ModelPlayer
    player = ModelPlayer(path, device=device, simulations=simulations)

    def select(state, rng):
        move, _ = player.choose(state)
        return move

    return select
