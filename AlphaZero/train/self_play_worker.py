"""多进程 Self-Play Worker — 批量推理优化版

每个 worker:
  1. 创建自己的 MCTS 实例
  2. 使用 BatchRemoteEvaluator 批量请求推理
  3. 完成一局后把训练数据放入 result_queue
"""
import sys
import time
import numpy as np
from multiprocessing import Process, Queue, Event
from pathlib import Path
from typing import Optional

_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))

from AlphaZero.engine import GameState, RED, BLACK
from AlphaZero.engine.move import ActionEncoder
from AlphaZero.engine.constants import POLICY_SIZE
from AlphaZero.train.config import AlphaZeroConfig
from AlphaZero.train.inference_server import BatchRequest, BatchResponse


class BatchRemoteEvaluator:
    """批量远程评估器 — 一次发送整个 batch，一次接收整个 batch

    大幅减少队列操作次数：从 2*N 次降到 2 次。
    """

    def __init__(self, worker_id: int,
                 request_queue: Queue,
                 response_queue: Queue):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._request_counter = 0

    def evaluate(self, state: GameState) -> tuple[np.ndarray, float]:
        """评估单个局面（兼容接口）"""
        policies, values = self.evaluate_batch([state])
        return policies[0], values[0]

    def evaluate_batch(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        """批量评估：一次发送，一次接收"""
        n = len(states)
        self._request_counter += 1

        # 编码所有状态
        state_list = []
        mask_list = []
        for s in states:
            state_list.append(s.encode())
            mask_list.append(s.legal_mask())

        # 打包成一个请求
        req = BatchRequest(
            request_id=self._request_counter,
            states=np.stack(state_list),
            masks=np.stack(mask_list),
            worker_id=self.worker_id,
        )

        # 一次发送
        self.request_queue.put(req)

        # 一次接收
        resp: BatchResponse = self.response_queue.get(timeout=120)

        return resp.policies, resp.values


def self_play_worker_fn(
    worker_id: int,
    config: AlphaZeroConfig,
    num_games: int,
    request_queue: Queue,
    response_queue: Queue,
    result_queue: Queue,
    stop_event: Event,
):
    """Self-Play Worker 主函数"""
    from AlphaZero.search import MCTS
    from AlphaZero.train.replay import ReplayBuffer

    evaluator = BatchRemoteEvaluator(worker_id, request_queue, response_queue)
    mcts = MCTS(
        evaluator=evaluator,
        num_simulations=config.num_simulations,
        c_puct=config.c_puct,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_epsilon=config.dirichlet_epsilon,
        batch_size=config.mcts_batch_size,
    )

    replay = ReplayBuffer(max_size=num_games * config.max_game_ply)
    stats = {'red': 0, 'black': 0, 'draw': 0, 'moves': 0, 'games': 0}
    t_start = time.perf_counter()

    for game_idx in range(num_games):
        if stop_event.is_set():
            break

        state = GameState.new_game()
        positions = []

        while not state.is_terminal() and state.move_count < config.max_game_ply:
            if stop_event.is_set():
                break

            temperature = 1.0 if state.move_count < config.temperature_ply else 0.1

            move, policy = mcts.select_move(state, temperature=temperature)
            if move is None:
                break

            # 检查是否吃子
            captured = state.board[move.to_row][move.to_col] is not None

            positions.append((state.encode(), policy.copy(), state.turn, captured))
            state = state.apply(move)

        # 终局结果
        result = state.game_result()
        if result is None:
            from AlphaZero.engine.state import GameResult
            result = GameResult(None, "max_ply")

        for state_enc, policy, is_red, captured in positions:
            current_player = RED if is_red else BLACK
            wdl = result.to_wdl(current_player)

            # 计算权重
            if result.is_draw or result.winner is None:
                # 和棋或非终局：基础权重
                weight = config.non_terminal_base_weight
            else:
                # 终局：根据是否吃子给不同权重
                weight = config.terminal_weight if captured else 1.0
                # 吃子额外奖励
                if captured:
                    weight += config.capture_reward

            replay.add(state_enc, policy, wdl, weight)

        if result.is_draw:
            stats['draw'] += 1
        elif result.winner == RED:
            stats['red'] += 1
        else:
            stats['black'] += 1
        stats['moves'] += state.move_count
        stats['games'] += 1

    elapsed = time.perf_counter() - t_start
    stats['elapsed'] = elapsed
    stats['worker_id'] = worker_id

    if len(replay) > 0:
        for i in range(len(replay)):
            result_queue.put((
                replay.states[i],
                replay.policies[i],
                replay.wdls[i],
                replay.weights[i],
            ))

    result_queue.put(('DONE', stats, None))
