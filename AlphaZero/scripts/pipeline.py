"""Continuous AlphaZero learner and independent self-play actors with a live browser."""
import argparse
from contextlib import contextmanager
import json
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import random
import os
import uuid
import threading
import time
import numpy as np
import torch
from AlphaZero.train.config import AlphaZeroConfig
from AlphaZero.train.trainer import Trainer
from AlphaZero.train.replay import ReplayBuffer
from AlphaZero.train.inference_server import InferenceServer
from AlphaZero.train.self_play_worker import self_play_group_fn
from AlphaZero.train.live import LiveTraining
from AlphaZero.train.performance import Recorder, Hub

@contextmanager
def run_lock(directory):
    stream = (directory / ".run.lock").open("a+")
    try:
        if __import__("os").name == "posix":
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            import msvcrt
            stream.write("0")
            stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        stream.close()
        raise RuntimeError("another learner owns this run directory") from exc
    try:
        (directory / "process.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
        yield
    finally:
        stream.close()

def pipeline(config, max_iterations=None, checkpoint=None, resume=None,
             output_dir=None, enable_monitor=True, performance=False,
             performance_interval=5.0, performance_gpu_every=64):
    if performance_interval <= 0 or performance_gpu_every < 1:
        raise ValueError("performance interval and GPU sampling period must be positive")
    config.__post_init__()
    if checkpoint is not None:
        raise ValueError("legacy WDL/18-channel weights require explicit migration; fresh v3 run recommended")
    directory = Path(output_dir or config.checkpoint_dir)
    config.checkpoint_dir = str(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with run_lock(directory):
        options = dict(interval=performance_interval, gpu_every=performance_gpu_every) if performance else None
        return _pipeline(config, directory, max_iterations, resume, enable_monitor, options)

def _pipeline(config, directory, max_iterations, resume, enable_monitor, performance_options=None):
    performance = Recorder("learner", **performance_options) if performance_options is not None else None
    performance_hub = Hub(performance) if performance is not None else None
    def storage_call(name, function, *args):
        if performance is None:
            return function(*args)
        started = time.perf_counter()
        result = function(*args)
        performance.observe(name, time.perf_counter() - started)
        return result
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; install a CUDA wheel or explicitly select CPU")
    torch.set_num_threads(config.torch_threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = True
    trainer = Trainer(config, config.device)
    trainer.performance = performance
    learner = directory / "learner.pt"
    recovery = None
    if resume:
        recovery = trainer.load_checkpoint(resume)
    elif learner.exists():
        raise ValueError("learner.pt exists: use --resume explicitly or choose a new run directory")
    replay = ReplayBuffer(directory / "replay.sqlite", config.replay_window_games, config.replay_cache_mb)
    try:
        replay.bind_spec(config.fingerprint)
        if resume and (recovery.get("replay_run_id") != replay.run_id
                       or recovery.get("replay_generation", -1) > replay.generation
                       or replay.max_actor_step > trainer.global_optimizer_step):
            raise ValueError("checkpoint/replay run or committed generation mismatch")
    except BaseException:
        replay.close()
        raise
    if replay.game_count and not resume:
        replay.close()
        raise ValueError("existing replay requires explicit resume; use a new directory for fresh training")
    config_file = directory / "config.json.tmp"
    config_file.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    config_file.replace(directory / "config.json")
    actor = directory / "latest.pt"
    trainer.run_id = replay.run_id
    storage_call("data.publish_wall", trainer.export_actor, actor)
    # Initial recovery point exists even if interrupted before the first update.
    if not resume:
        storage_call("data.checkpoint_wall", trainer.save_checkpoint, learner, replay.recovery_marker())
    context = mp.get_context("spawn")
    stop = context.Event()
    collector_stop = threading.Event()
    shutting_down = threading.Event()
    results = context.Queue(maxsize=config.num_workers * 2)
    events = context.Queue(maxsize=max(32, config.num_workers * 4))
    performance_queue = context.Queue(maxsize=max(64, config.num_workers * 2)) if performance else None
    server = InferenceServer(actor, config, context, replay.run_id, performance_queue, performance_options)
    processes = []
    live = None
    collector = None
    errors = []
    metrics = (directory / "metrics.jsonl").open("a", encoding="utf-8")
    session_id = uuid.uuid4().hex
    metrics.write(json.dumps(dict(event="session_start", session_id=session_id,
        step=trainer.global_optimizer_step, run_id=replay.run_id, resumed=bool(resume),
        config=config.to_dict(), config_changes=trainer.resume_config_changes)) + "\n")
    metrics.flush()
    performance_log = (directory / "performance.jsonl").open("a", encoding="utf-8") if performance else None
    last_performance = time.perf_counter() if performance else None
    def drain_performance():
        if performance_queue is not None:
            for _ in range(200):
                try:
                    performance_hub.receive(performance_queue.get_nowait())
                except Empty:
                    break
    def publish_performance(force=False):
        nonlocal last_performance
        if performance is None:
            return
        now = time.perf_counter()
        if not force and now - last_performance < performance_options["interval"]:
            return
        last_performance = now
        report = performance_hub.snapshot()
        report.update(session_id=session_id, step=trainer.global_optimizer_step, run_id=replay.run_id,
                      interval_seconds=performance_options["interval"],
                      gpu_sample_every=performance_options["gpu_every"])
        if live:
            with live.lock:
                report["hardware"] = dict(live.state["hardware"])
            live.update(performance=report)
        try:
            payload = json.dumps(report, allow_nan=False)
            performance_log.write(payload + "\n")
            performance_log.flush()
            temporary = directory / "performance.json.tmp"
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(directory / "performance.json")
        except OSError:
            performance.increment("monitor.output_errors")
    try:
        if enable_monitor:
            live = LiveTraining(config.telemetry_host, config.telemetry_port, config.telemetry_interval)
            host, port = live.address
            print(f"Training dashboard: http://{host}:{port}", flush=True)
            live.update(workers=config.num_workers, device=config.device, precision=config.precision,
                        actor_processes=min(config.actor_processes, config.num_workers),
                        actor_step=trainer.global_optimizer_step, step=trainer.global_optimizer_step,
                        positions=len(replay), completed_games=replay.generation, config=config.to_dict())
            live.update(performance=dict(enabled=performance is not None))
        ready = server.start()
        sequences = replay.next_sequences()
        process_count = min(config.actor_processes, config.num_workers)
        for group in range(process_count):
            worker_ids = list(range(group, config.num_workers, process_count))
            worker_args = (
                worker_ids, config, server.request_queue,
                {worker:server.response_queues[worker] for worker in worker_ids},
                results, events, stop, sequences)
            if performance is not None:
                worker_args += (performance_queue, performance_options)
            process = context.Process(target=self_play_group_fn, args=worker_args, daemon=True)
            process.start()
            processes.append(process)
        def collect():
            try:
                while True:
                    try:
                        message = results.get(timeout=0.05)
                        if message.get("kind") != "game":
                            raise RuntimeError("invalid reliable worker message")
                        if performance is not None:
                            write_started = time.perf_counter()
                        added = replay.add_game(message["episode"], message["samples"],
                            worker=message["worker"], sequence=message["sequence"],
                            reason=message["reason"], actor_start=message["actor_start"],
                            actor_end=message["actor_end"], trace=message["trace"])
                        if performance is not None:
                            performance.observe("data.replay_write_wall", time.perf_counter() - write_started)
                            if added:
                                performance.increment("data.completed_games")
                                performance.increment("data.new_positions", len(message["samples"]))
                        if live:
                            live.complete(message, len(replay), replay.generation)
                    except Empty:
                        if collector_stop.is_set():
                            break
                    for _ in range(100):
                        try:
                            event = events.get_nowait()
                        except Empty:
                            break
                        if live:
                            live.event(event)
                    drain_performance()
                    publish_performance()
                    failed = [(i,p.exitcode) for i,p in enumerate(processes)
                              if p.exitcode is not None]
                    if not shutting_down.is_set() and (failed or not server._process.is_alive() or stop.is_set()):
                        raise RuntimeError(f"actor/inference process failed: {failed}")
            except BaseException as exc:
                errors.append(exc)
                stop.set()
                if live:
                    live.update(phase="failed", errors=[dict(error=repr(exc))])
        collector = threading.Thread(target=collect, daemon=True)
        collector.start()
        if live:
            live.update(phase="collecting")
        while replay.game_count < config.min_replay_games:
            if errors:
                raise errors[0]
            time.sleep(0.1)
        limit = min(config.max_optimizer_steps,
                    (max_iterations or config.max_iterations) * config.checkpoint_interval)
        if live:
            live.update(phase="training")
        while trainer.global_optimizer_step < limit:
            if errors:
                raise errors[0]
            started = time.perf_counter()
            stats = trainer.step(replay)
            if errors:
                raise errors[0]
            stats["seconds"] = time.perf_counter() - started
            stats["replay_generation"] = replay.generation
            stats["run_id"] = replay.run_id
            stats["session_id"] = session_id
            stats["num_simulations"] = config.num_simulations
            stats["generated_positions"] = replay.total_positions
            stats["training_to_data_ratio"] = (trainer.global_optimizer_step * config.batch_size
                                                / max(replay.total_positions, 1))
            metrics.write(json.dumps(stats, allow_nan=False) + "\n")
            metrics.flush()
            if live:
                live.update(**stats, positions=len(replay), completed_games=replay.generation)
            step = trainer.global_optimizer_step
            if step % config.recovery_interval == 0 and step % config.checkpoint_interval != 0:
                storage_call("data.checkpoint_wall", trainer.save_checkpoint, learner, replay.recovery_marker())
            if step % config.checkpoint_interval == 0 or step == limit:
                trainer.current_iteration = step // config.checkpoint_interval
                storage_call("data.checkpoint_wall", trainer.save_checkpoint, learner, replay.recovery_marker())
                storage_call("data.publish_wall", trainer.export_actor, actor)
                server.refresh()
                if live:
                    live.update(actor_step=step)
                print(f"step={step} loss={stats['loss']:.5f} lr={stats['lr']:.5g} "
                      f"games={replay.generation} positions={len(replay)} "
                      f"update={stats['seconds']:.3f}s", flush=True)
    except BaseException as exc:
        if live and not isinstance(exc, KeyboardInterrupt):
            live.update(phase="failed", errors=[dict(error=repr(exc))])
        raise
    finally:
        shutting_down.set()
        stop.set()
        for process in processes:
            process.join(4)
            if process.is_alive():
                process.terminate()
                process.join(4)
        collector_stop.set()
        if collector:
            collector.join(10)
        server.stop()
        drain_performance()
        publish_performance(force=True)
        if performance_log:
            performance_log.close()
        if performance_queue is not None:
            performance_queue.cancel_join_thread()
            performance_queue.close()
        for queue in (results, events):
            queue.cancel_join_thread()
            queue.close()
        if live:
            live.close()
        metrics.close()
        replay.flush()
        replay.close()
    return dict(step=trainer.global_optimizer_step, games=replay.generation, positions=len(replay))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--workers", type=int, help="number of simultaneous independent games")
    parser.add_argument("--actor-processes", type=int, help="CPU processes sharing the independent games")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--no-monitor", action="store_true")
    parser.add_argument("--performance", action=argparse.BooleanOptionalAction, default=False,
                        help="optional five-module performance accounting; checkpoint format unchanged")
    parser.add_argument("--performance-interval", type=float, default=5.0)
    parser.add_argument("--performance-gpu-every", type=int, default=64)
    args = parser.parse_args()
    config = AlphaZeroConfig.from_yaml(args.config)
    if args.workers is not None:
        config.num_workers = args.workers
    if args.actor_processes is not None:
        config.actor_processes = args.actor_processes
    if args.device:
        config.device = args.device
    config.__post_init__()
    result = pipeline(config, args.max_iterations, resume=args.resume,
                      output_dir=args.output_dir, enable_monitor=not args.no_monitor,
                      performance=args.performance, performance_interval=args.performance_interval,
                      performance_gpu_every=args.performance_gpu_every)
    print(json.dumps(result), flush=True)

if __name__ == "__main__":
    main()
