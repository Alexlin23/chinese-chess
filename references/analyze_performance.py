"""Summarize only measured windows; do not add overlapping module times."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup-seconds", type=float, default=60)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (Path(args.run_dir) / "performance.jsonl").read_text().splitlines()]
    final = rows[-1]
    rows = [row for row in rows if row["session_id"] == final["session_id"]]
    baseline = next((row for row in rows if row["elapsed_seconds"] >= args.warmup_seconds), rows[0])
    seconds = final["elapsed_seconds"] - baseline["elapsed_seconds"]
    if seconds <= 0:
        parser.error("at least two reports with a positive observation interval are required")
    metrics = {}
    for name, state in final["metrics"].items():
        old = baseline["metrics"].get(name, {})
        count, total = state["count"] - old.get("count", 0), state["total"] - old.get("total", 0)
        metrics[name] = dict(count=count, total=total, mean=total / count if count else None,
                             whole_session_p95_upper_approx=state["p95_upper_approx"])
    counters = {name: value - baseline["counters"].get(name, 0) for name, value in final["counters"].items()}
    hardware = {}
    for key in ("gpu_percent", "vram_gb", "cpu_percent", "ram_gb"):
        samples = [row.get("hardware", {}) for row in rows
                   if row["elapsed_seconds"] >= baseline["elapsed_seconds"]]
        if key in ("gpu_percent", "vram_gb"):
            samples = [sample for sample in samples
                       if sample.get("gpu_status") not in ("stale", "unavailable")
                       and sample.get("gpu_sample_age_seconds", 0) <= 15
                       and sample.get("gpu") != "unavailable"]
        values = [sample.get(key) for sample in samples]
        values = [value for value in values if value is not None]
        if values:
            hardware[key] = dict(mean=statistics.mean(values), min=min(values), max=max(values), samples=len(values))
    result = dict(session_id=final["session_id"], run_id=final["run_id"],
                  interval_seconds=final.get("interval_seconds"), gpu_sample_every=final.get("gpu_sample_every"),
                  full_session=final, steady_window=dict(start_seconds=baseline["elapsed_seconds"],
                  end_seconds=final["elapsed_seconds"], seconds=seconds, start_step=baseline["step"], end_step=final["step"],
                  metrics=metrics, counters=counters, hardware=hardware,
                  rates={key: counters.get(counter, 0) * scale / seconds for key, counter, scale in [
                    ("mcts_simulations_per_second", "mcts.simulations", 1),
                    ("inference_positions_per_second", "inference.positions", 1),
                    ("new_positions_per_hour", "data.new_positions", 3600),
                    ("completed_games_per_hour", "data.completed_games", 3600)]}),
                  limitations=["First 60 seconds excluded using cumulative-count differences",
                               "Source reports may lag by one reporting interval",
                               "P95 is whole-session only; it is not a steady-window quantile",
                               "Hourly rates from a short window do not establish long-run throughput",
                               "GPU/CPU hardware snapshots are sampled, not kernel traces"])
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["steady_window"], indent=2))


if __name__ == "__main__":
    main()
