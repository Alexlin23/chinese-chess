"""Arena — 新旧模型对战评估"""
import time
from typing import Optional

import numpy as np
import torch

from .config import AlphaZeroConfig
from ..model import PolicyWDLEncoder, NeuralEvaluator
from ..search import MCTS
from ..engine import GameState, RED, BLACK


class Arena:
    """新旧模型对战评估

    用法：
        arena = Arena(config)
        result = arena.evaluate(new_model_path, best_model_path)
        if result['should_promote']:
            # 晋升新模型
    """

    def __init__(self, config: AlphaZeroConfig, device: str = 'cpu'):
        self.config = config
        self.device = device

    def evaluate(self, new_model_path: str,
                 best_model_path: str,
                 num_games: Optional[int] = None,
                 verbose: bool = True) -> dict:
        """新模型 vs 旧模型对战评估

        Args:
            new_model_path: 新模型权重路径
            best_model_path: 最佳模型权重路径
            num_games: 对战局数（覆盖配置）
            verbose: 是否打印进度

        Returns:
            dict: {
                'wins': int,
                'losses': int,
                'draws': int,
                'score_rate': float,  # (wins + 0.5 * draws) / total
                'should_promote': bool,
            }
        """
        num_games = num_games or self.config.arena_games

        # 加载模型
        new_model = self._load_model(new_model_path)
        best_model = self._load_model(best_model_path)

        new_eval = NeuralEvaluator(new_model, self.device)
        best_eval = NeuralEvaluator(best_model, self.device)

        wins = losses = draws = 0
        t_start = time.perf_counter()

        for i in range(num_games):
            # 交替执红：偶数局新模型执红，奇数局旧模型执红
            if i % 2 == 0:
                red_eval, black_eval = new_eval, best_eval
            else:
                red_eval, black_eval = best_eval, new_eval

            result = self._play_game(red_eval, black_eval)

            # 统计（从新模型视角）
            if i % 2 == 0:  # 新模型执红
                if result.winner == RED:
                    wins += 1
                elif result.winner == BLACK:
                    losses += 1
                else:
                    draws += 1
            else:  # 新模型执黑
                if result.winner == BLACK:
                    wins += 1
                elif result.winner == RED:
                    losses += 1
                else:
                    draws += 1

            if verbose:
                elapsed = time.perf_counter() - t_start
                print(f"\r  Arena {i+1}/{num_games}: "
                      f"W={wins} L={losses} D={draws} "
                      f"[{elapsed:.0f}s]", end="")

        if verbose:
            print()

        total = wins + losses + draws
        score_rate = (wins + 0.5 * draws) / total if total > 0 else 0.0
        should_promote = score_rate >= self.config.promotion_score_rate

        return {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'score_rate': score_rate,
            'should_promote': should_promote,
        }

    def _load_model(self, path: str) -> PolicyWDLEncoder:
        """加载模型"""
        model = PolicyWDLEncoder(
            num_blocks=self.config.num_blocks,
            num_filters=self.config.num_filters,
        ).to(self.device)

        state = torch.load(path, map_location=self.device, weights_only=False)
        model.load_state_dict(state['model_state_dict'])
        model.eval()
        return model

    def _play_game(self, red_eval: NeuralEvaluator,
                   black_eval: NeuralEvaluator) -> 'GameResult':
        """执行一局对战"""
        from ..engine.state import GameResult as GR
 
        state = GameState.new_game()

        red_mcts = MCTS(
            evaluator=red_eval,
            num_simulations=self.config.num_simulations,
            c_puct=self.config.c_puct,
        )
        black_mcts = MCTS(
            evaluator=black_eval,
            num_simulations=self.config.num_simulations,
            c_puct=self.config.c_puct,
        )

        while not state.is_terminal():
            if state.turn:  # 红方
                move, _ = red_mcts.select_move(state, temperature=0.0)
            else:  # 黑方
                move, _ = black_mcts.select_move(state, temperature=0.0)

            if move is None:
                break
            state = state.apply(move)

        result = state.game_result()
        if result is None:
            return GR(None, "max_ply")
        return result
