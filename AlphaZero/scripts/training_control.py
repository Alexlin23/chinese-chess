"""Read or stop exactly one local WSL training run."""
import argparse
import json
import os
from pathlib import Path
import signal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "stop"))
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    directory = Path(args.run_dir).resolve()
    matches = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().decode().strip("\0").split("\0")
            if "AlphaZero.scripts.pipeline" not in argv or "--output-dir" not in argv:
                continue
            run_arg = argv[argv.index("--output-dir") + 1]
            cwd = (proc / "cwd").resolve()
            if (cwd / run_arg).resolve() == directory:
                matches.append(int(proc.name))
        except (OSError, ValueError, UnicodeDecodeError):
            continue
    if len(matches) > 1:
        raise RuntimeError("More than one process matches this exact run; refusing to stop")
    if args.action == "stop" and matches:
        os.kill(matches[0], signal.SIGINT)
    print(json.dumps(dict(run_dir=str(directory), pids=matches,
                         stop_requested=args.action == "stop" and bool(matches))))


if __name__ == "__main__":
    main()
