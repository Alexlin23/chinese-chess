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
        self.request_queue = Queue(maxsize=num_workers * 100)
        self.response_queues = {
            i: Queue(maxsize=100) for i in range(num_workers)
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

        total_requests = 0
        total_batches = 0
        t_start = time.perf_counter()
        t_log = time.perf_counter()

        while not stop_event.is_set():
            # 收集批量请求
            try:
                req: BatchRequest = request_queue.get(timeout=max_wait_ms / 1000)
            except:
                continue

            if req is None:
                continue

            try:
                t0 = time.perf_counter()
                states = req.states
                masks_arr = req.masks
                n = len(states)

                state_tensor = torch.from_numpy(states).to(device, non_blocking=True)
                mask_tensor = torch.from_numpy(masks_arr).to(device, non_blocking=True)
                t_to_gpu = time.perf_counter() - t0

                t0 = time.perf_counter()
                with torch.no_grad():
                    p_logits, w_logits = model(state_tensor)
                t_forward = time.perf_counter() - t0

                t0 = time.perf_counter()
                # 避免 clone: 用 masked_fill 替代
                p_logits = p_logits.masked_fill(~mask_tensor, float('-inf'))
                policy_probs = torch.softmax(p_logits, dim=1)
                wdl_probs = torch.softmax(w_logits, dim=1)
                values = wdl_probs[:, 0] - wdl_probs[:, 2]
                # 一次性转 numpy（减少 GPU→CPU 次数）
                policy_probs = policy_probs.cpu().numpy()
                wdl_probs = wdl_probs.cpu().numpy()
                values = values.cpu().numpy()
                t_post = time.perf_counter() - t0

                t0 = time.perf_counter()
                resp = BatchResponse(
                    request_id=req.request_id,
                    policies=policy_probs,
                    wdls=wdl_probs,
                    values=values,
                    worker_id=req.worker_id,
                )
                response_queues[req.worker_id].put(resp)
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
                # 返回均匀分布作为 fallback
                n = len(req.states)
                resp = BatchResponse(
                    request_id=req.request_id,
                    policies=np.ones((n, POLICY_SIZE), dtype=np.float32) / POLICY_SIZE,
                    wdls=np.tile(np.array([0.33, 0.34, 0.33], dtype=np.float32), (n, 1)),
                    values=np.zeros(n, dtype=np.float32),
                    worker_id=req.worker_id,
                )
                response_queues[req.worker_id].put(resp)

        elapsed = time.perf_counter() - t_start
        print(f"[InferenceServer] 停止: {total_requests} 请求, "
              f"{total_batches} batch, {elapsed:.0f}s, "
              f"{total_requests/max(elapsed,1):.0f} req/s")
