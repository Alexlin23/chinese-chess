"""Bounded hardware check; never writes to a real training run."""
import argparse
import json
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from AlphaZero.engine import GameState
from AlphaZero.train.config import AlphaZeroConfig
from AlphaZero.train.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--loss-reduction", choices=("mean", "sum"))
    args = parser.parse_args()
    config = AlphaZeroConfig.from_yaml(args.config)
    if args.loss_reduction:
        config.loss_reduction = args.loss_reduction
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(config.torch_threads)
    state = GameState.new_game()
    encoded = state.encode()
    pi = state.legal_mask().astype(np.float32)
    pi /= pi.sum()
    pi = state.canonical_policy(pi)
    class Batch:
        generation = 0
        def __len__(self): return config.batch_size
        def sample(self,n):
            return (np.repeat(encoded[None],n,axis=0),
                    np.repeat(pi[None],n,axis=0),
                    np.zeros(n,dtype=np.float32), np.ones(n,dtype=np.float32))
    trainer = Trainer(config, config.device)
    if config.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    stats = {}
    for _ in range(args.updates):
        stats = trainer.step(Batch())
    if config.device == "cuda":
        torch.cuda.synchronize()
    result = dict(device=config.device, torch=torch.__version__, batch=config.batch_size,
                  loss_reduction=config.loss_reduction, seed=config.seed,
                  parameters=trainer.model.count_parameters(), updates=args.updates,
                  seconds=time.perf_counter()-started, stats=stats)
    if config.device == "cuda":
        from AlphaZero.model import NeuralEvaluator
        trainer.model.eval()
        fp_policy, fp_value = NeuralEvaluator(trainer.model, "cuda", "fp32", True).evaluate(state)
        bf_policy, bf_value = NeuralEvaluator(trainer.model, "cuda", "bf16", True).evaluate(state)
        result["bf16_vs_fp32"] = dict(policy_max_abs=float(np.max(np.abs(fp_policy-bf_policy))),
                                      value_abs=abs(fp_value-bf_value))
    if config.device == "cuda":
        result.update(gpu=torch.cuda.get_device_name(),
                      peak_allocated_gb=torch.cuda.max_memory_allocated()/1024**3,
                      peak_reserved_gb=torch.cuda.max_memory_reserved()/1024**3)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory)/"learner.pt"
        trainer.save_checkpoint(checkpoint)
        expected = {k:v.cpu().clone() for k,v in trainer.model.state_dict().items()}
        trainer.load_checkpoint(checkpoint)
        result["restore_equal"] = all(torch.equal(expected[k],v.cpu()) for k,v in trainer.model.state_dict().items())
        result["restored_step"] = trainer.global_optimizer_step
    print(json.dumps(result, indent=2), flush=True)

if __name__ == "__main__":
    main()
