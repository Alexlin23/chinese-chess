"""Resolve all local entry points against one configured run directory."""
from pathlib import Path
from AlphaZero.train.config import AlphaZeroConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "config/local-train.yaml"


def run_paths(config_path=DEFAULT_CONFIG, run_dir=None):
    path = Path(config_path).expanduser()
    config = AlphaZeroConfig.from_yaml(path if path.is_absolute() else PROJECT_ROOT / path)
    directory = Path(run_dir or config.checkpoint_dir).expanduser()
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    return config, directory.resolve()
