"""自我对弈 — 使用 MCTS + 当前模型生成训练数据。

支持两种模式:
  1. 批量生成（训练用）：run_self_play() 生成 N 局，保存到 ReplayBuffer 文件
  2. 单局观战（Demo 用）：SelfPlayGame 类支持逐步执行，每步可回调通知前端
"""
import sys
import os
import time
import json
from pathlib import Path
from typing import Optional, Callable

import numpy as np

# 确保可导入 backend 模块
_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root / "backend"))
sys.path.insert(0, str(_project_root))

from .config import AlphaZeroConfig
from .replay import ReplayBuffer

from AlphaZero.engine import GameState, KING, RED, BLACK
from AlphaZero.engine.fast_chess import create_initial_board
from AlphaZero.engine.move import ActionEncoder
from AlphaZero.search import MCTS
from AlphaZero.model import AlphaZeroNet, NeuralEvaluator


class SelfPlayGame:
    """单局自我对弈，支持逐步执行（用于实时观战）。

    用法：
        game = SelfPlayGame(evaluator, config)
        while not game.is_terminal():
            game.step()           # 走一步
            board_state = game.current_board_dict()  # 获取当前棋盘（dict 格式）
    """

    def __init__(self, mcts: MCTS, config: AlphaZeroConfig,
                 game_id: Optional[int] = None,
                 on_move: Optional[Callable] = None):
        """
        Args:
            mcts:     MCTS 搜索实例
            config:   训练配置
            game_id:  对局 ID（数据库记录用）
            on_move:  每步回调 on_move(state: GameState, move: Move, policy: np.ndarray)
        """
        self.mcts = mcts
        self.config = config
        self.game_id = game_id
        self.on_move = on_move

        self.state = GameState.new_game()
        self.positions: list[tuple[np.ndarray, np.ndarray, bool]] = []
        # (state_enc, mcts_policy, is_red_turn)

        self._move_history: list[dict] = []  # 供 DB 写入
        self._done = False

    def step(self) -> Optional[dict]:
        """执行一步走棋。返回走法信息，若终局则返回 None。"""
        if self.is_terminal():
            self._done = True
            return None

        # 温度控制
        if self.state.move_count < self.config.temperature_threshold:
            temperature = 1.0
        else:
            temperature = 0.0

        # MCTS 搜索
        move, policy = self.mcts.select_move(self.state, temperature=temperature)
        if move is None:
            self._done = True
            return None

        # 记录训练数据
        is_red = self.state.turn  # bool: True=红
        self.positions.append((self.state.encode(), policy, is_red))

        # 执行走棋
        self.state = self.state.apply(move)

        # 构建走法记录（供 DB 存储）
        move_record = {
            "from_row": move.from_row, "from_col": move.from_col,
            "to_row": move.to_row, "to_col": move.to_col,
            "turn": "r" if is_red else "b",
            "step": self.state.move_count,
        }
        self._move_history.append(move_record)

        # 回调通知
        if self.on_move:
            self.on_move(self.state, move, policy)

        return move_record

    def is_terminal(self) -> bool:
        if self._done or self.state.is_terminal():
            return True
        # 步数超限强制和棋
        if self.state.move_count >= self.config.max_game_length:
            return True
        return False

    def result(self) -> float:
        """终局结果：+1=红胜, -1=黑胜, 0=和棋。步数超限返回和棋。"""
        if self.state.move_count >= self.config.max_game_length:
            return 0.0
        return self.state.result() or 0.0

    def get_training_data(self) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """获取本局的训练数据。

        Returns:
            [(state_enc, mcts_policy, value), ...]
            其中 value 从该位置的行棋方视角标注。
        """
        final_result = self.result()
        data = []
        for state_enc, policy, is_red in self.positions:
            # 从行棋方视角标注结果
            # is_red: 行棋方是红方 → value = final_result (红胜=+1 对红好)
            # not is_red: 行棋方是黑方 → value = -final_result
            value = final_result if is_red else -final_result
            data.append((state_enc.copy(), policy.copy(), value))
        return data

    def current_board_dict(self) -> list[list[Optional[dict]]]:
        """将当前 numpy 棋盘转为 dict 格式（前端兼容）。

        Returns:
            10×9 的嵌套列表，每格 None 或 {"type": str, "color": str}
        """
        _num_to_type = {
            1: "帥", 2: "仕", 3: "相", 4: "馬", 5: "車", 6: "炮", 7: "兵",
        }
        _num_to_type_black = {
            1: "將", 2: "士", 3: "象", 4: "馬", 5: "車", 6: "砲", 7: "卒",
        }
        board = self.state.board
        result = []
        for r in range(10):
            row = []
            for c in range(9):
                val = board[r, c]
                if val == 0:
                    row.append(None)
                elif val > 0:
                    row.append({"type": _num_to_type[abs(val)], "color": "r"})
                else:
                    row.append({"type": _num_to_type_black[abs(val)], "color": "b"})
            result.append(row)
        return result

    def move_history(self) -> list[dict]:
        return self._move_history


# ── 批量自我对弈 ──

def run_self_play(config: AlphaZeroConfig,
                  model_path: Optional[str] = None,
                  data_dir: Optional[str] = None,
                  iteration: int = 0,
                  verbose: bool = True) -> ReplayBuffer:
    """执行一轮自我对弈，生成训练数据。

    Args:
        config:    训练配置
        model_path: 模型权重路径（None = 随机初始化）
        data_dir:   数据保存目录
        iteration:  当前迭代编号
        verbose:    是否打印进度

    Returns:
        ReplayBuffer 包含所有新生成的训练数据
    """
    # 创建模型和评估器
    model = AlphaZeroNet(
        num_blocks=config.num_blocks,
        num_filters=config.num_filters,
    )
    if model_path and os.path.exists(model_path):
        state = __import__('torch').load(model_path, map_location='cpu',
                                         weights_only=True)
        model.load_state_dict(state['model_state_dict'])
        if verbose:
            print(f"加载模型: {model_path}")

    evaluator = NeuralEvaluator(model)
    mcts = MCTS(
        evaluator=evaluator,
        num_simulations=config.num_simulations,
        c_puct=config.c_puct,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_epsilon=config.dirichlet_epsilon,
    )

    # 经验回放缓冲区
    replay = ReplayBuffer(max_size=config.replay_buffer_size)

    # 对弈统计
    total_moves = 0
    red_wins = 0
    black_wins = 0
    draws = 0
    t_start = time.perf_counter()

    for game_idx in range(config.games_per_iteration):
        game = SelfPlayGame(mcts, config)
        while not game.is_terminal():
            game.step()

        # 收集训练数据
        data = game.get_training_data()
        for state_enc, policy, value in data:
            replay.add(state_enc, policy, value)

        # 统计
        r = game.result()
        if r > 0.5:
            red_wins += 1
        elif r < -0.5:
            black_wins += 1
        else:
            draws += 1

        total_moves += game.state.move_count

        if verbose:
            elapsed = time.perf_counter() - t_start
            avg_time = elapsed / (game_idx + 1)
            print(f"\r对弈 {game_idx+1}/{config.games_per_iteration}  "
                  f"| 红胜:{red_wins} 黑胜:{black_wins} 和:{draws}  "
                  f"| 均步:{total_moves/(game_idx+1):.0f}  "
                  f"| 均时:{avg_time:.1f}s/局  "
                  f"| 已用:{elapsed:.0f}s", end="")

    if verbose:
        print()  # 换行
        total_time = time.perf_counter() - t_start
        print(f"自我对弈完成: {config.games_per_iteration} 局, "
              f"{total_moves} 步, {total_time:.0f}s "
              f"({total_time/config.games_per_iteration:.1f}s/局)")

    # 保存到磁盘
    if data_dir:
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        save_path = data_path / f"games_iter{iteration:03d}.npz"
        replay.save(str(save_path))

    return replay


# ── CLI 入口 ──

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AlphaZero 自我对弈")
    parser.add_argument("--games", type=int, default=50, help="对弈局数")
    parser.add_argument("--sims", type=int, default=200, help="MCTS 模拟次数")
    parser.add_argument("--blocks", type=int, default=20, help="残差块数")
    parser.add_argument("--filters", type=int, default=128, help="卷积通道数")
    parser.add_argument("--model", type=str, default=None, help="预训练权重路径")
    parser.add_argument("--output", type=str, default="AlphaZero/data",
                        help="输出目录")
    parser.add_argument("--iter", type=int, default=0, help="迭代编号")
    args = parser.parse_args()

    config = AlphaZeroConfig(
        num_blocks=args.blocks,
        num_filters=args.filters,
        num_simulations=args.sims,
        games_per_iteration=args.games,
    )

    print(f"配置: {args.blocks}块/{args.filters}通道, "
          f"{args.sims}模拟, {args.games}局")
    replay = run_self_play(config, model_path=args.model,
                           data_dir=args.output, iteration=args.iter)
    print(f"\n缓冲区: {len(replay)} 条训练样本")
