"""GPU Batch 推理服务 — 独立进程，批量接收请求，batch 推理后返回

架构:
  N 个 self-play worker → Queue → InferenceServer (GPU) → Queue → workers

优化:
  - 批量请求/响应：一次发送整个 MCTS batch，减少队列操作
  - 共享内存：避免 pickle 序列化大 numpy 数组
"""
import os
import sys
import time
from queue import Empty
import torch
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Queue, Event, shared_memory
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from AlphaZero.model import PolicyWDLEncoder
from AlphaZero.engine.constants import POLICY_SIZE, INPUT_CHANNELS


@dataclass
class BatchRequest:
    """批量推理请求 — 一次发送多个状态"""
    request_id: int
    states: np.ndarray      # (N, 18, 10, 9) float32
    masks: np.ndarray       # (N, 8100) bool
    worker_id: int


@dataclass
class BatchResponse:
    """批量推理响应 — 一次返回多个结果"""
    request_id: int
    policies: np.ndarray    # (N, 8100) float32
    wdls: np.ndarray        # (N, 3) float32
    values: np.ndarray      # (N,) float32
    worker_id: int
    error: Optional[str] = None


class InferenceServer:
    """GPU Batch 推理服务

    独立进程运行，从 request_queue 读取批量请求，
    batch 推理后把结果放入 response_queues[worker_id]。
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = 'cuda',
                 batch_size: int = 256,
                 max_wait_ms: float = 2.0,
                 num_workers: int = 4):
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.num_workers = num_workers

        # 进程间通信
        self.request_queue = Queue(maxsize=max(2, num_workers * 2))
        self.response_queues = {
            i: Queue(maxsize=2) for i in range(num_workers)
        }
        self.stop_event = Event()

        self._process = None

    def start(self):
        """启动推理服务进程"""
        self._process = Process(
            target=self._run,
            args=(self.model_path, self.device, self.batch_size,
                  self.max_wait_ms, self.request_queue,
                  self.response_queues, self.stop_event, self.num_workers),
            daemon=True,
        )
        self._process.start()
        return self._process.pid

    def stop(self):
        """停止推理服务"""
        self.stop_event.set()
        if self._process:
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()

    @staticmethod
    def _run(model_path, device, batch_size, max_wait_ms,
             request_queue, response_queues, stop_event, num_workers):
        """推理服务主循环"""
        # 加载模型
        if model_path and os.path.exists(model_path):
            state = torch.load(model_path, map_location=device, weights_only=False)
            cfg = state.get('config', {})
            model = PolicyWDLEncoder(
                num_blocks=cfg.get('num_blocks', 8),
                num_filters=cfg.get('num_filters', 128),
            ).to(device)
            model.load_state_dict(state['model_state_dict'])
        else:
            model = PolicyWDLEncoder(num_blocks=8, num_filters=128).to(device)

        model.eval()
        print(f"[InferenceServer] 模型已加载到 {device}, "
              f"参数量: {model.count_parameters():,}")

        # 预分配 GPU tensor（避免重复分配）
        max_batch = batch_size * 2  # 留余量
        prealloc_states = torch.zeros((max_batch, INPUT_CHANNELS, 10, 9),
                                      dtype=torch.float32, device=device)
        prealloc_masks = torch.zeros((max_batch, POLICY_SIZE),
                                     dtype=torch.bool, device=device)

        total_requests = 0
        total_batches = 0
        t_start = time.perf_counter()
        t_log = time.perf_counter()
        pending = None

        while not stop_event.is_set():
            # 收集批量请求
            try:
                req = pending if pending is not None else request_queue.get(timeout=0.1)
                pending = None
            except Empty:
                continue

            if req is None:
                continue

            requests = [req]
            requested = len(req.states)
            deadline = time.perf_counter() + max_wait_ms / 1000
            while requested < batch_size and not stop_event.is_set():
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    next_req = request_queue.get(timeout=remaining)
                except Empty:
                    break
                if next_req is None:
                    continue
                if requested + len(next_req.states) > batch_size:
                    pending = next_req
                    break
                requests.append(next_req)
                requested += len(next_req.states)

            try:
                t0 = time.perf_counter()
                for item in requests:
                    count = len(item.states)
                    if (count < 1 or count > batch_size or
                            item.states.shape != (count, INPUT_CHANNELS, 10, 9) or
                            item.masks.shape != (count, POLICY_SIZE) or
                            item.worker_id not in response_queues):
                        raise ValueError("invalid inference request")
                states = np.concatenate([item.states for item in requests])
                masks_arr = np.concatenate([item.masks for item in requests])
                n = len(states)

                # 使用预分配 tensor（避免重复分配内存）
                prealloc_states[:n].copy_(torch.from_numpy(states))
                prealloc_masks[:n].copy_(torch.from_numpy(masks_arr))
                state_tensor = prealloc_states[:n]
                mask_tensor = prealloc_masks[:n]
                t_to_gpu = time.perf_counter() - t0

                t0 = time.perf_counter()
                with torch.no_grad():
                    p_logits, w_logits = model(state_tensor)
                t_forward = time.perf_counter() - t0

                t0 = time.perf_counter()
                p_logits[~mask_tensor] = float('-inf')
                policy_probs = torch.softmax(p_logits, dim=1).cpu().numpy()
                wdl_probs = torch.softmax(w_logits, dim=1).cpu().numpy()
                values = wdl_probs[:, 0] - wdl_probs[:, 2]
                t_post = time.perf_counter() - t0

                t0 = time.perf_counter()
                offset = 0
                for item in requests:
                    count = len(item.states)
                    resp = BatchResponse(
                        request_id=item.request_id,
                        policies=policy_probs[offset:offset + count],
                        wdls=wdl_probs[offset:offset + count],
                        values=values[offset:offset + count],
                        worker_id=item.worker_id,
                    )
                    response_queues[item.worker_id].put(resp, timeout=30)
                    offset += count
                t_dist = time.perf_counter() - t0

                total_requests += n
                total_batches += 1

                # 每20个batch打印一次详细计时 + 吞吐量
                if total_batches % 20 == 0:
                    elapsed = time.perf_counter() - t_start
                    throughput = total_requests / max(elapsed, 1)
                    print(f"[InferenceServer] n={n} "
                          f"to_gpu={t_to_gpu*1000:.1f}ms "
                          f"forward={t_forward*1000:.1f}ms "
                          f"post={t_post*1000:.1f}ms "
                          f"dist={t_dist*1000:.1f}ms "
                          f"total={total_requests} req "
                          f"throughput={throughput:.0f} req/s")

            except Exception as e:
                print(f"[InferenceServer] 推理错误: {e}")
                for item in requests:
                    if item.worker_id in response_queues:
                        resp = BatchResponse(
                            request_id=item.request_id, policies=None,
                            wdls=None, values=None, worker_id=item.worker_id,
                            error=repr(e),
                        )
                        try:
                            response_queues[item.worker_id].put(resp, timeout=1)
                        except Exception:
                            pass
                stop_event.set()

        elapsed = time.perf_counter() - t_start
        print(f"[InferenceServer] 停止: {total_requests} 请求, "
              f"{total_batches} batch, {elapsed:.0f}s, "
              f"{total_requests/max(elapsed,1):.0f} req/s")

