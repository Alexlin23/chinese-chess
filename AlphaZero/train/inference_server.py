"""GPU Batch 推理服务 — 独立进程，接收请求，batch 推理后返回

架构:
  N 个 self-play worker → Queue → InferenceServer (GPU) → Queue → workers

优势:
  - 多个 self-play 进程共享一个 GPU 模型
  - batch 推理提高 GPU 利用率
  - self-play 在 CPU 上并行，不阻塞 GPU
"""
import os
import sys
import time
import torch
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Queue, Event
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from AlphaZero.model import PolicyWDLEncoder
from AlphaZero.engine.constants import POLICY_SIZE, INPUT_CHANNELS


@dataclass
class InferenceRequest:
    """推理请求"""
    request_id: int
    state_encoding: np.ndarray   # (18, 10, 9) float32
    legal_mask: np.ndarray       # (8100,) bool
    worker_id: int


@dataclass
class InferenceResponse:
    """推理响应"""
    request_id: int
    policy_probs: np.ndarray     # (8100,) float32
    wdl_probs: np.ndarray        # (3,) float32
    value: float
    worker_id: int


class InferenceServer:
    """GPU Batch 推理服务

    独立进程运行，从 request_queue 读取请求，
    batch 推理后把结果放入 response_queues[worker_id]。
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = 'cuda',
                 batch_size: int = 64,
                 max_wait_ms: float = 5.0,
                 num_workers: int = 4):
        """
        Args:
            model_path: 模型权重路径
            device: 'cuda' 或 'cpu'
            batch_size: 最大 batch 大小
            max_wait_ms: 最大等待时间（ms），凑不满 batch 也推理
            num_workers: worker 数量
        """
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

    def submit(self, request: InferenceRequest):
        """提交推理请求"""
        self.request_queue.put(request)

    def get_result(self, worker_id: int, timeout: float = 10.0) -> InferenceResponse:
        """获取推理结果"""
        return self.response_queues[worker_id].get(timeout=timeout)

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
            # 收集 batch
            batch = []
            t_wait = time.perf_counter()

            while len(batch) < batch_size:
                # 检查是否超时
                elapsed_ms = (time.perf_counter() - t_wait) * 1000
                if len(batch) > 0 and elapsed_ms >= max_wait_ms:
                    break

                try:
                    req = request_queue.get(timeout=max_wait_ms / 1000)
                    batch.append(req)
                except:
                    if len(batch) > 0:
                        break
                    continue

            if not batch:
                continue

            # 每5秒打印一次状态
            if time.perf_counter() - t_log > 5:
                print(f"[InferenceServer] 已处理 {total_requests} 请求, "
                      f"{total_batches} batch, "
                      f"队列大小: ~{request_queue.qsize()}")
                t_log = time.perf_counter()

            # 批量推理
            try:
                t0 = time.perf_counter()
                states = np.stack([r.state_encoding for r in batch])
                masks = np.stack([r.legal_mask for r in batch])
                t_stack = time.perf_counter() - t0

                t0 = time.perf_counter()
                state_tensor = torch.from_numpy(states).to(device, non_blocking=True)
                mask_tensor = torch.from_numpy(masks).to(device, non_blocking=True)
                t_to_gpu = time.perf_counter() - t0

                t0 = time.perf_counter()
                with torch.no_grad():
                    p_logits, w_logits = model(state_tensor)
                t_forward = time.perf_counter() - t0

                t0 = time.perf_counter()
                p_logits = p_logits.clone()
                p_logits[~mask_tensor] = float('-inf')
                policy_probs = torch.softmax(p_logits, dim=1).cpu().numpy()
                wdl_probs = torch.softmax(w_logits, dim=1).cpu().numpy()
                values = wdl_probs[:, 0] - wdl_probs[:, 2]
                t_post = time.perf_counter() - t0

                # 分发结果
                t0 = time.perf_counter()
                for i, req in enumerate(batch):
                    resp = InferenceResponse(
                        request_id=req.request_id,
                        policy_probs=policy_probs[i],
                        wdl_probs=wdl_probs[i],
                        value=float(values[i]),
                        worker_id=req.worker_id,
                    )
                    response_queues[req.worker_id].put(resp)
                t_dist = time.perf_counter() - t0

                total_requests += len(batch)
                total_batches += 1

                # 每100个batch打印一次详细计时
                if total_batches % 100 == 0:
                    print(f"[InferenceServer] batch={len(batch)} "
                          f"stack={t_stack*1000:.1f}ms "
                          f"to_gpu={t_to_gpu*1000:.1f}ms "
                          f"forward={t_forward*1000:.1f}ms "
                          f"post={t_post*1000:.1f}ms "
                          f"dist={t_dist*1000:.1f}ms")

            except Exception as e:
                print(f"[InferenceServer] 推理错误: {e}")
                # 返回均匀分布作为 fallback
                for req in batch:
                    resp = InferenceResponse(
                        request_id=req.request_id,
                        policy_probs=np.ones(POLICY_SIZE, dtype=np.float32) / POLICY_SIZE,
                        wdl_probs=np.array([0.33, 0.34, 0.33], dtype=np.float32),
                        value=0.0,
                        worker_id=req.worker_id,
                    )
                    response_queues[req.worker_id].put(resp)

        elapsed = time.perf_counter() - t_start
        print(f"[InferenceServer] 停止: {total_requests} 请求, "
              f"{total_batches} batch, {elapsed:.0f}s, "
              f"{total_requests/max(elapsed,1):.0f} req/s")
