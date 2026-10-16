"""Short full-search/full-batch check, not a full learning experiment."""
import argparse
import signal
import tempfile
from AlphaZero.train.config import AlphaZeroConfig
from AlphaZero.scripts.pipeline import pipeline
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument("--timeout", type=int, default=300,
                        help="WSL wall-clock bound; an unfinished check is reported as such")
    args=parser.parse_args()
    config=AlphaZeroConfig.from_yaml("config/default.yaml")
    config.max_optimizer_steps=2
    config.telemetry_port=8765
    directory=args.output_dir or tempfile.mkdtemp(prefix="az-full-budget-")
    def timed_out(signum, frame):
        raise TimeoutError("bounded acceptance expired before the two-update closure")
    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(args.timeout)
    try:
        print(pipeline(config,output_dir=directory),flush=True)
    finally:
        signal.alarm(0)
