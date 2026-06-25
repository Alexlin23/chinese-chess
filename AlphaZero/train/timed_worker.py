"""带计时的 Self-Play Worker — 定位瓶颈"""
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


class TimedRemoteEvaluator:
    """带计时的远程评估器"""

    def __init__(self, worker_id: int,
                 request_queue: Queue,
                 response_queue: Queue):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._request_counter = 0

        # 计时统计
        self.t_encode = 0.0
        self.t_queue_put = 0.0
        self.t_queue_get = 0.0
        self.n_calls = 0

    def evaluate(self, state: GameState) -> tuple[np.ndarray, float]:
        """评估单个局面"""
        # 编码
        t0 = time.perf_counter()
        state_enc = state.encode()
        legal_mask = state.legal_mask()
        self.t_encode += time.perf_counter() - t0

        self._request_counter += 1
        req = InferenceRequest(
            request_id=self._request_counter,
            state_encoding=state_enc,
            legal_mask=legal_mask,
            worker_id=self.worker_id,
        )

        # 发送请求
        t0 = time.perf_counter()
        self.request_queue.put(req)
        self.t_queue_put += time.perf_counter() - t0

        # 等待响应
        t0 = time.perf_counter()
        resp: InferenceResponse = self.response_queue.get(timeout=120)
        self.t_queue_get += time.perf_counter() - t0

        self.n_calls += 1
        return resp.policy_probs, resp.value

    def evaluate_batch(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        """批量评估：先全部提交，再全部收集"""
        n = len(states)

        # 编码 + 提交
        t0 = time.perf_counter()
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
        self.t_encode += time.perf_counter() - t0

        # 收集
        t0 = time.perf_counter()
        policies = []
        values = []
        for _ in range(n):
            resp: InferenceResponse = self.response_queue.get(timeout=120)
            policies.append(resp.policy_probs)
            values.append(resp.value)
        self.t_queue_get += time.perf_counter() - t0

        self.n_calls += n
        return np.stack(policies), np.array(values, dtype=np.float32)

    def report(self, total_time: float):
        """打印计时报告"""
        print(f"  [Worker {self.worker_id}] 计时报告:")
        print(f"    总推理次数: {self.n_calls}")
        print(f"    编码耗时: {self.t_encode:.2f}s ({self.t_encode/total_time*100:.1f}%)")
        print(f"    发送请求: {self.t_queue_put:.2f}s ({self.t_queue_put/total_time*100:.1f}%)")
        print(f"    等待响应: {self.t_queue_get:.2f}s ({self.t_queue_get/total_time*100:.1f}%)")
        t_other = total_time - self.t_encode - self.t_queue_put - self.t_queue_get
        print(f"    其他(CPU): {t_other:.2f}s ({t_other/total_time*100:.1f}%)")


def timed_self_play_worker_fn(
    worker_id: int,
    config: AlphaZeroConfig,
    num_games: int,
    request_queue: Queue,
    response_queue: Queue,
    result_queue: Queue,
    stop_event: Event,
):
    """带计时的 Self-Play Worker"""
    from AlphaZero.search import MCTS
    from AlphaZero.train.replay import ReplayBuffer

    evaluator = TimedRemoteEvaluator(worker_id, request_queue, response_queue)
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

        t_game_start = time.perf_counter()
        state = GameState.new_game()
        positions = []
        move_count = 0

        while not state.is_terminal() and state.move_count < config.max_game_ply:
            if stop_event.is_set():
                break

            temperature = 1.0 if state.move_count < config.temperature_ply else 0.1

            t_move = time.perf_counter()
            move, policy = mcts.select_move(state, temperature=temperature)
            t_move_elapsed = time.perf_counter() - t_move

            if move is None:
                break

            positions.append((state.encode(), policy.copy(), state.turn))
            state = state.apply(move)
            move_count += 1

            # 每10步打印一次
            if move_count % 10 == 0:
                print(f"  [Worker {worker_id}] 局{game_idx+1} 步{move_count} "
                      f"({t_move_elapsed:.2f}s/步)")

        # 终局结果
        result = state.game_result()
        if result is None:
            from AlphaZero.engine.state import GameResult
            result = GameResult(None, "max_ply")

        for state_enc, policy, is_red in positions:
            current_player = RED if is_red else BLACK
            wdl = result.to_wdl(current_player)
            replay.add(state_enc, policy, wdl)

        if result.is_draw:
            stats['draw'] += 1
        elif result.winner == RED:
            stats['red'] += 1
        else:
            stats['black'] += 1
        stats['moves'] += state.move_count
        stats['games'] += 1

        t_game = time.perf_counter() - t_game_start
        print(f"  [Worker {worker_id}] 局{game_idx+1}完成: "
              f"{move_count}步 {t_game:.1f}s")

    # 报告
    elapsed = time.perf_counter() - t_start
    evaluator.report(elapsed)

    stats['elapsed'] = elapsed
    stats['worker_id'] = worker_id

    if len(replay) > 0:
        for i in range(len(replay)):
            result_queue.put((
                replay.states[i],
                replay.policies[i],
                replay.wdls[i],
            ))

    result_queue.put(('DONE', stats, None))
