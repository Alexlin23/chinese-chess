"""Continuous independent episodes, uniform root-visit targets and lossy UI events."""
from queue import Empty, Full
import json
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from .inference_server import BatchRequest
from .replay import pack_sample
from ..engine import GameState
from ..search import MCTS


def self_play_group_fn(worker_ids, config, request_queue, response_queues,
                       result_queue, event_queue, stop_event, start_sequences):
    """Independent games share a CPU process while each waits for its own inference.

    Each game still has a private RNG, search tree, response queue and pinned
    snapshot. Threads overlap independent games, never dependent tree leaves.
    """
    with ThreadPoolExecutor(max_workers=len(worker_ids)) as pool:
        futures = [pool.submit(self_play_worker_fn, worker, config, request_queue,
            response_queues[worker], result_queue, event_queue, stop_event,
            start_sequences.get(worker, 0)) for worker in worker_ids]
        done, _ = wait(futures, return_when=FIRST_COMPLETED)
        failures = [future.exception() for future in done if future.exception() is not None]
        if failures:
            stop_event.set()
            raise failures[0]
        if not stop_event.is_set():
            stop_event.set()
            raise RuntimeError("self-play slot exited unexpectedly")

class BatchRemoteEvaluator:
    def __init__(self, worker_id, request_queue, response_queue, stop_event=None):
        self.worker_id, self.request_queue, self.response_queue = worker_id, request_queue, response_queue
        self.stop_event = stop_event
        self._request_counter, self.model_step = 0, 0
        self.pinned_step = None

    def evaluate(self, state):
        policies, values = self.evaluate_batch([state])
        return policies[0], float(values[0])

    def evaluate_batch(self, states):
        self._request_counter += 1
        request = BatchRequest(self._request_counter, np.stack([s.encode() for s in states]),
                               np.stack([s.canonical_mask() for s in states]), self.worker_id,
                               self.pinned_step)
        self.request_queue.put(request, timeout=10)
        deadline = time.monotonic() + 120
        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                raise InterruptedError("self-play cancelled")
            try:
                response = self.response_queue.get(timeout=1)
                break
            except Empty:
                if time.monotonic() >= deadline:
                    raise TimeoutError("neural inference timed out")
        if response.worker_id != self.worker_id or response.request_id != request.request_id:
            raise RuntimeError("mismatched inference response")
        if response.error:
            raise RuntimeError(response.error)
        self.model_step = response.model_step
        if self.pinned_step is not None and self.model_step != self.pinned_step:
            raise RuntimeError("actor version changed inside an episode")
        self.pinned_step = self.model_step
        for i, state in enumerate(states):
            if not state.turn:
                response.policies[i] = response.policies[i, ::-1].copy()
        return response.policies, response.values

def selfplay_temperature(ply, num_sampling_moves=30):
    """Data S1 select_action: first 30 plies sample; then highest visit count."""
    return 1.0 if ply < num_sampling_moves else 0.0

def self_play_worker_fn(worker_id, config, request_queue, response_queue,
                        result_queue, event_queue, stop_event, start_sequence=0):
    evaluator = BatchRemoteEvaluator(worker_id, request_queue, response_queue, stop_event)
    sequence = start_sequence
    def emit(data):
        try:
            event_queue.put_nowait(data)
        except Full:
            pass  # Display cannot stall simulation or the reliable training queue.
    try:
        while not stop_event.is_set():
            evaluator.pinned_step = None
            episode = f"w{worker_id}-g{sequence}"
            rng = np.random.default_rng(np.random.SeedSequence([config.seed, worker_id, sequence]))
            state = GameState.new_game(config.max_game_ply)
            search = MCTS(evaluator, config.num_simulations, config.c_puct,
                          config.dirichlet_alpha, config.dirichlet_epsilon,
                          config.mcts_batch_size, config.virtual_loss, config.pb_c_base, rng,
                          reuse_tree=config.reuse_tree)
            positions, moves, versions = [], [], []
            started = time.time()
            actor_start = evaluator.model_step
            while not state.is_terminal() and not stop_event.is_set():
                emit(dict(kind="board", worker=worker_id, episode=episode, sequence=sequence, ply=state.move_count,
                          board=state.board.tolist(), turn=state.current_player, started=started,
                          actor_step=evaluator.model_step, diagnostics=search.diagnostics,
                          natural_limit=state.natural_limit_status(),
                          last_move=moves[-1] if moves else None))
                move, pi = search.select_move(state, temperature=selfplay_temperature(state.move_count, config.num_sampling_moves),
                                              add_exploration_noise=True)
                if move is None:
                    raise RuntimeError("no action in nonterminal episode")
                if not positions:
                    actor_start = evaluator.model_step
                positions.append((pack_sample(state, pi, 0), state.current_player))
                moves.append([move.from_row, move.from_col, move.to_row, move.to_col])
                versions.append(evaluator.model_step)
                state = state.apply(move)
            if stop_event.is_set():
                break
            result = state.game_result()
            if result is None:
                raise RuntimeError("unfinished episode cannot enter replay")
            samples = []
            for sample, player in positions:
                sample.value = result.to_value(player)
                samples.append(sample)
            message = dict(kind="game", episode=episode, worker=worker_id, sequence=sequence,
                           samples=samples, reason=result.reason, winner=result.winner,
                           actor_start=actor_start, actor_end=evaluator.model_step,
                           trace=json.dumps(dict(moves=moves, actor_steps=versions,
                                                 num_simulations=config.num_simulations)), ply=state.move_count,
                           board=state.board.tolist(), turn=state.current_player)
            while not stop_event.is_set():
                try:
                    result_queue.put(message, timeout=1)
                    break
                except Full:
                    continue
            sequence += 1
    except InterruptedError:
        if not stop_event.is_set():
            raise
    except BaseException as exc:
        emit(dict(kind="worker_error", worker=worker_id, error=repr(exc)))
        raise
