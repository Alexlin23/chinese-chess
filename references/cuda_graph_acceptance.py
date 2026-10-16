import tempfile
from AlphaZero.train.config import AlphaZeroConfig
from AlphaZero.scripts.pipeline import pipeline
if __name__=='__main__':
 with tempfile.TemporaryDirectory(prefix='az-graph-integration-') as directory:
  config=AlphaZeroConfig.from_yaml('config/cuda-smoke.yaml')
  print(pipeline(config,output_dir=directory,enable_monitor=False),flush=True)
