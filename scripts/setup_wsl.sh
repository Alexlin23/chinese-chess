#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
AZ_ENV_DIR="${AZ_ENV_DIR:-$HOME/.venvs/xiangqi-alphazero}"
python3 -m venv "$AZ_ENV_DIR"
"$AZ_ENV_DIR/bin/python" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
"$AZ_ENV_DIR/bin/python" -m pip install -r requirements.txt
"$AZ_ENV_DIR/bin/python" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())'
