#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
AZ_ENV_DIR="${AZ_ENV_DIR:-$HOME/.venvs/xiangqi-alphazero}"
if [[ ! -x "$AZ_ENV_DIR/bin/python" ]]; then
  echo "Missing environment: run bash scripts/setup_wsl.sh first." >&2
  exit 1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "$AZ_ENV_DIR/bin/python" -m AlphaZero.scripts.pipeline "$@"
