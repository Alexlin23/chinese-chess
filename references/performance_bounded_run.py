"""Resume the unchanged training CLI, request SIGINT after a short observation.

External watchdog only; no new stop/checkpoint policy is added to the learner.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="config/local-train.yaml")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 600:
        parser.error("use a short observation of 1..600 seconds")
    directory = Path(args.run_dir)
    if not (directory / "learner.pt").is_file():
        parser.error("a resumable learner is required")
    command = [sys.executable, "-m", "AlphaZero.scripts.pipeline", "--config", args.config,
               "--output-dir", str(directory), "--resume", str(directory / "learner.pt"), "--performance"]
    record = dict(command=command, requested_seconds=args.seconds,
                  started_utc=datetime.now(timezone.utc).isoformat(), stop_requested=False)
    started = time.perf_counter()
    with (directory / "performance-run.log").open("w") as output, (directory / "performance-run.stderr.log").open("w") as errors:
        process = subprocess.Popen(command, stdout=output, stderr=errors, start_new_session=True)
        record["pipeline_pid"] = process.pid
        (directory / "performance-run-control.json").write_text(json.dumps(record, indent=2))
        try:
            process.wait(timeout=args.seconds)
        except subprocess.TimeoutExpired:
            record["stop_requested"] = True
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                record["forced_group_shutdown"] = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        record.update(exit_code=process.returncode, elapsed_seconds=time.perf_counter() - started,
                      stopped_utc=datetime.now(timezone.utc).isoformat())
        (directory / "performance-run-control.json").write_text(json.dumps(record, indent=2))
        print(json.dumps(record, indent=2), flush=True)
    if not record["stop_requested"] and process.returncode:
        raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
