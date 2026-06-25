"""训练监控 — 实时显示 CPU/GPU 使用率和训练进度"""
import os
import time
import threading
import psutil
import torch


class TrainingMonitor:
    """训练监控器，每 N 秒刷新一次状态"""

    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self.running = False
        self._thread = None
        self.stats = {
            'iteration': 0,
            'max_iterations': 0,
            'phase': 'idle',
            'workers': 0,
            'games_done': 0,
            'games_total': 0,
            'samples': 0,
            'loss': None,
            'arena_w': 0,
            'arena_l': 0,
            'arena_d': 0,
            'promoted': 0,
        }

    def start(self):
        """启动监控线程"""
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """停止监控"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=self.interval + 1)

    def update(self, **kwargs):
        """更新统计信息"""
        self.stats.update(kwargs)

    def _run(self):
        """监控主循环"""
        while self.running:
            self._print_status()
            time.sleep(self.interval)

    def _print_status(self):
        """打印状态"""
        # CPU 信息
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        cpu_avg = sum(cpu_percent) / len(cpu_percent)
        cpu_count = len(cpu_percent)

        # 内存信息
        mem = psutil.virtual_memory()

        # GPU 信息
        gpu_info = self._get_gpu_info()

        # 进程信息
        proc = psutil.Process(os.getpid())
        proc_mem = proc.memory_info().rss / 1024**2

        # 清屏并打印
        print("\033[2J\033[H", end="")  # 清屏
        print("=" * 60)
        print("  AlphaZero 训练监控")
        print("=" * 60)
        print()

        # 训练进度
        s = self.stats
        print(f"  迭代: {s['iteration']}/{s['max_iterations']}")
        print(f"  阶段: {s['phase']}")
        print(f"  Workers: {s['workers']}")
        print()

        # 自对弈进度
        if s['phase'] == 'self_play':
            print(f"  自对弈: {s['games_done']}/{s['games_total']} 局")
            print(f"  样本数: {s['samples']}")
        elif s['phase'] == 'arena':
            print(f"  Arena: W={s['arena_w']} L={s['arena_l']} D={s['arena_d']}")
        elif s['phase'] == 'training':
            print(f"  训练 loss: {s['loss']:.4f}" if s['loss'] else "  训练中...")

        print(f"  累计晋升: {s['promoted']} 次")
        print()

        # CPU 信息
        print("  ── CPU ──")
        print(f"  核心数: {cpu_count}")
        print(f"  使用率: {cpu_avg:.1f}%")

        # 显示每个核心的使用率（每行8个）
        for i in range(0, cpu_count, 8):
            cores = cpu_percent[i:i+8]
            bar = " ".join([f"{c:5.1f}%" for c in cores])
            print(f"    {bar}")
        print()

        # 内存信息
        print("  ── 内存 ──")
        print(f"  总量: {mem.total / 1024**3:.1f} GB")
        print(f"  已用: {mem.used / 1024**3:.1f} GB ({mem.percent:.1f}%)")
        print(f"  进程: {proc_mem:.0f} MB")
        print()

        # GPU 信息
        if gpu_info:
            print("  ── GPU ──")
            print(f"  设备: {gpu_info['name']}")
            print(f"  显存: {gpu_info['mem_used']:.1f} / {gpu_info['mem_total']:.1f} GB ({gpu_info['mem_percent']:.1f}%)")
            print(f"  利用率: {gpu_info['utilization']:.1f}%")
            print(f"  温度: {gpu_info['temperature']}°C")
            print(f"  功率: {gpu_info['power']:.0f}W")
        print()

        # 进程数
        print("  ── 进程 ──")
        python_procs = [p for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info'])
                        if 'python' in p.info['name'].lower()]
        print(f"  Python 进程: {len(python_procs)}")
        for p in python_procs[:5]:  # 只显示前5个
            try:
                mem_mb = p.info['memory_info'].rss / 1024**2
                print(f"    PID {p.info['pid']}: CPU {p.info['cpu_percent']:.1f}%, MEM {mem_mb:.0f}MB")
            except:
                pass
        print()
        print("=" * 60)

    def _get_gpu_info(self) -> dict:
        """获取 GPU 信息（使用 nvidia-smi 命令）"""
        if not torch.cuda.is_available():
            return None

        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                
                # 处理 [N/A] 的情况
                def parse_float(v, default=0):
                    try:
                        return float(v.replace('[N/A]', str(default)))
                    except:
                        return default
                
                mem_used = parse_float(values[0]) / 1024  # MB -> GB
                mem_total = parse_float(values[1]) / 1024
                utilization = parse_float(values[2])
                temperature = int(parse_float(values[3]))
                power = parse_float(values[4])
                
                # 如果内存是0，使用torch获取
                if mem_total == 0:
                    mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    mem_used = torch.cuda.memory_allocated(0) / 1024**3
                
                return {
                    'name': torch.cuda.get_device_name(0),
                    'mem_total': mem_total,
                    'mem_used': mem_used,
                    'mem_percent': mem_used / mem_total * 100 if mem_total > 0 else 0,
                    'utilization': utilization,
                    'temperature': temperature,
                    'power': power,
                }
        except Exception:
            pass

        # 回退到 torch
        return {
            'name': torch.cuda.get_device_name(0),
            'mem_total': torch.cuda.get_device_properties(0).total_memory / 1024**3,
            'mem_used': torch.cuda.memory_allocated(0) / 1024**3,
            'mem_percent': torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory * 100,
            'utilization': 0,
            'temperature': 0,
            'power': 0,
        }


# 全局监控实例
monitor = TrainingMonitor(interval=5.0)
