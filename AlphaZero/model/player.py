"""Load a versioned, trained checkpoint for actual play and evaluation."""
from pathlib import Path
import torch
from ..engine import GameState
from ..search import MCTS
from ..train.config import AlphaZeroConfig
from .network import PolicyValueNetwork
from .neural_eval import NeuralEvaluator, CudaGraphEvaluator


def _publication_count(payload, path, model_step):
    count = payload.get("publication_count")
    if type(count) is int and count >= 0:
        return count
    # SGD updates equal publications only for the one-update publication schedule.
    if payload["config"].get("checkpoint_interval") != 1:
        return None
    if payload["kind"] == "learner" or "optimizer_step_offset" in payload:
        return int(payload["step"])
    # Older actor exports store only a version including the pretraining offset.
    # Read metadata from their own run; never replace the loaded game's weights.
    learner = path.with_name("learner.pt")
    if path.name != "latest.pt" or not learner.is_file() or not payload.get("run_id"):
        return None
    saved = torch.load(learner, map_location="meta", weights_only=False, mmap=True)
    if (saved.get("kind") != "learner" or saved.get("run_id") != payload["run_id"]
            or saved.get("fingerprint") != payload["fingerprint"]):
        return None
    offset = saved.get("optimizer_step_offset", 0)
    count = model_step - offset
    return count if 0 <= count <= saved["step"] else None


class ModelPlayer:
    def __init__(self, checkpoint, device="cpu", simulations=800, *,
                 exploration_noise=False, search_noise_depth=None):
        if type(exploration_noise) is not bool:
            raise ValueError("exploration_noise must be boolean")
        if search_noise_depth is not None and (type(search_noise_depth) is not int or search_noise_depth < 0):
            raise ValueError("search_noise_depth must be a nonnegative integer")
        self.path = Path(checkpoint).resolve()
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        if payload.get("kind") not in ("actor", "learner"):
            raise ValueError("需要当前版本的训练模型")
        self.config = AlphaZeroConfig(**payload["config"])
        if payload.get("fingerprint") != self.config.fingerprint:
            raise ValueError("模型版本或规则规格不匹配")
        self.step = int(payload["step"]) + payload.get("optimizer_step_offset", 0)
        if self.step < 1:
            raise ValueError("第 0 步是随机初始化，尚无训练成果可用")
        self.publication_count = _publication_count(payload, self.path, self.step)
        if simulations < 1:
            raise ValueError("搜索次数必须大于零")
        self.simulations = simulations
        self.exploration_noise = exploration_noise
        self.search_noise_depth = (self.config.search_noise_depth if search_noise_depth is None
                                   else search_noise_depth)
        self.device = device
        self.model = PolicyValueNetwork(self.config.num_blocks, self.config.num_filters,
                                        self.config.input_channels).to(device)
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        if device.startswith("cuda") and self.config.channels_last:
            self.model.to(memory_format=torch.channels_last)
        self.model.eval()
        precision = self.config.precision if device.startswith("cuda") else "fp32"
        if device.startswith("cuda") and self.config.inference_cuda_graph:
            self.evaluator = CudaGraphEvaluator(self.model, device, precision, self.config.channels_last)
        else:
            self.evaluator = NeuralEvaluator(self.model, device, precision,
                                            self.config.channels_last and device.startswith("cuda"))
        self.evaluator.evaluate(GameState.new_game())

    def choose(self, state):
        search = MCTS(self.evaluator, num_simulations=self.simulations,
                      c_puct=self.config.c_puct, pb_c_base=self.config.pb_c_base,
                      dirichlet_alpha=self.config.dirichlet_alpha,
                      dirichlet_epsilon=self.config.dirichlet_epsilon,
                      search_noise_depth=self.search_noise_depth)
        move, _ = search.select_move(state, temperature=0, add_exploration_noise=self.exploration_noise)
        return move, search.diagnostics
