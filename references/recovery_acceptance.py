"""Fresh CPU run followed by two additional updates from the same committed run."""
import tempfile
from pathlib import Path
from AlphaZero.train.config import AlphaZeroConfig
from AlphaZero.scripts.pipeline import pipeline
if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="az-recovery-") as directory:
        config=AlphaZeroConfig.from_yaml("config/test.yaml")
        first=pipeline(config,output_dir=directory,enable_monitor=False)
        assert first["step"]==4
        config.max_optimizer_steps=6
        config.max_iterations=3
        second=pipeline(config,output_dir=directory,
                        resume=Path(directory)/"learner.pt",enable_monitor=False)
        assert second["step"]==6
        print(dict(fresh=first,resumed=second),flush=True)
