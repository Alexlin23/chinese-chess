"""多进程 Self-Play Worker — 在独立进程中运行自对弈

每个 worker:
  1. 创建自己的 MCTS 实例
  2. 使用 RemoteEvaluator 通过 Queue 向 InferenceServer 请求推理
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
from AlphaZero.train.inference_server import InferenceRequest, InferenceResponse


class RemoteEvaluator:
    """远程评估器 — 通过 Queue 向 InferenceServer 请求推理

    实现 Evaluator 协议，但 evaluate() 是阻塞的（等待 InferenceServer 返回）。
    """

    def __init__(self, worker_id: int,
                 request_queue: Queue,
                 response_queue: Queue):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._request_counter = 0

    def evaluate(self, state: GameState) -> tuple[np.ndarray, float]:
        """评估单个局面，阻塞等待结果"""
        state_enc = state.encode()
        legal_mask = state.legal_mask()

        self._request_counter += 1
        req = InferenceRequest(
            request_id=self._request_counter,
            state_encoding=state_enc,
            legal_mask=legal_mask,
            worker_id=self.worker_id,
        )

        self.request_queue.put(req)
        resp: InferenceResponse = self.response_queue.get(timeout=60)

        return resp.policy_probs, resp.value

    def evaluate_batch(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        """批量评估：先全部提交，再全部收集（让 InferenceServer 真正 batch）"""
        n = len(states)
        # 1) 先全部提交
        request_ids = []
        for s in states:
            state_enc = s.encode()
            legal_mask = s.legal_mask()
            self._request_counter += 1
            req = InferenceRequest(
                request_id=self._request_counter,
                state_encoding=state_enc,
                legal_mask=legal_mask,
                worker_id=self.worker_id,
            )
            self.request_queue.put(req)
            request_ids.append(self._request_counter)

        # 2) 再全部收集
        policies = []
        values = []
        for _ in range(n):
            resp: InferenceResponse = self.response_queue.get(timeout=60)
            policies.append(resp.policy_probs)
            values.append(resp.value)

        return np.stack(policies), np.array(values, dtype=np.float32)


def self_play_worker_fn(
    worker_id: int,
    config: AlphaZeroConfig,
    num_games: int,
    request_queue: Queue,
    response_queue: Queue,
    result_queue: Queue,
    stop_event: Event,
):
    """Self-Play Worker 主函数

    Args:
        worker_id: 工作进程 ID
        config: 训练配置
        num_games: 本 worker 要完成的对局数
        request_queue: 发送推理请求的队列
        response_queue: 接收推理结果的队列
        result_queue: 输出训练数据的队列
        stop_event: 停止信号
    """
    from AlphaZero.search import MCTS
    from AlphaZero.train.replay import ReplayBuffer

    evaluator = RemoteEvaluator(worker_id, request_queue, response_queue)
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

            # 温度退火
            temperature = 1.0 if state.move_count < config.temperature_ply else 0.1

            # MCTS 搜索
            move, policy = mcts.select_move(state, temperature=temperature)
            if move is None:
                break

            # 记录
            positions.append((state.encode(), policy.copy(), state.turn))

            # 执行走法
            state = state.apply(move)

        # 终局结果
        result = state.game_result()
        if result is None:
            from AlphaZero.engine.state import GameResult
            result = GameResult(None, "max_ply")

        # 生成训练数据
        for state_enc, policy, is_red in positions:
            current_player = RED if is_red else BLACK
            wdl = result.to_wdl(current_player)
            replay.add(state_enc, policy, wdl)

        # 统计
        if result.is_draw:
            stats['draw'] += 1
        elif result.winner == RED:
            stats['red'] += 1
        else:
            stats['black'] += 1
        stats['moves'] += state.move_count
        stats['games'] += 1

    # 把训练数据放入 result_queue
    elapsed = time.perf_counter() - t_start
    stats['elapsed'] = elapsed
    stats['worker_id'] = worker_id

    # 把 replay buffer 中的数据逐个放入 result_queue
    if len(replay) > 0:
        for i in range(len(replay)):
            result_queue.put((
                replay.states[i],
                replay.policies[i],
                replay.wdls[i],
            ))

    result_queue.put(('DONE', stats, None))
