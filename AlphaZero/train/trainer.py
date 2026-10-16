"""Continuous SGD learner: Scalar MSE, policy CE and half-L2, with explicit batch reduction contract."""
from pathlib import Path
import json
import os
import random
import numpy as np
import torch
from torch.nn import functional as F
from .config import AlphaZeroConfig
from ..model.network import PolicyValueNetwork
from ..model.neural_eval import autocast_context

def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            directory = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)

def create_model(config, device):
    model = PolicyValueNetwork(config.num_blocks, config.num_filters, config.input_channels).to(device)
    if str(device).startswith("cuda") and config.channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model

class Trainer:
    def __init__(self, config: AlphaZeroConfig, device="cpu"):
        self.config, self.device = config, device
        self.model = create_model(config, device)
        self.forward_model = torch.compile(self.model) if config.compile_model else self.model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.learning_rate,
                                         momentum=config.lr_momentum, weight_decay=config.weight_decay)
        self.global_optimizer_step = 0
        self.current_iteration = 0
        self.resume_config_changes = {}

    def step(self, replay):
        self.model.train()
        decay = sum(self.global_optimizer_step >= milestone for milestone in self.config.lr_milestones)
        lr = self.config.learning_rate * self.config.lr_decay_rate ** decay
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        states, policies, values, _ = replay.sample(self.config.batch_size)
        def transfer(array):
            tensor = torch.from_numpy(array)
            if str(self.device).startswith("cuda"):
                tensor = tensor.pin_memory()
            return tensor.to(self.device, non_blocking=True)
        x, pi, z = transfer(states), transfer(policies), transfer(values)
        if str(self.device).startswith("cuda") and self.config.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        self.optimizer.zero_grad(set_to_none=True)
        with autocast_context(self.device, self.config.precision):
            logits, v = self.forward_model(x)
        # Full action CE is intentional; legal masking is conditional inference.
        policy_per_position = -(pi * F.log_softmax(logits.float(), dim=1)).sum(1)
        policy_loss = getattr(policy_per_position, self.config.loss_reduction)()
        value_loss = F.mse_loss(v.float(), z, reduction=self.config.loss_reduction)
        loss = policy_loss + value_loss
        regularization = (self.config.weight_decay / 2) * sum(
            p.detach().square().sum() for p in self.model.parameters())
        if not bool(torch.isfinite(loss + regularization)):
            raise FloatingPointError("non-finite learner loss")
        loss.backward()
        # Fail closed; clipping would introduce a separate algorithmic choice.
        finite = torch.stack([torch.isfinite(p.grad).all() for p in self.model.parameters()
                              if p.grad is not None]).all()
        if not bool(finite):
            raise FloatingPointError("non-finite learner gradients")
        self.optimizer.step()
        updated = list(self.model.parameters()) + list(self.model.buffers()) + [value for state in self.optimizer.state.values()
                    for value in state.values() if isinstance(value, torch.Tensor)]
        if not bool(torch.stack([torch.isfinite(value).all() for value in updated]).all()):
            raise FloatingPointError("non-finite parameters or optimizer state after SGD")
        self.global_optimizer_step += 1
        return dict(step=self.global_optimizer_step, loss=float(loss.detach() + regularization),
                    data_loss=float(loss.detach()), data_loss_per_position=float(loss.detach()) / (self.config.batch_size if self.config.loss_reduction == "sum" else 1),
                    regularization=float(regularization),
                    policy_loss=float(policy_loss.detach()), value_loss=float(value_loss.detach()),
                    lr=lr, replay_positions=len(replay))

    def train(self, replay, updates=None):
        stats = {}
        for _ in range(updates or self.config.checkpoint_interval):
            stats = self.step(replay)
        self.current_iteration += 1
        return stats

    def actor_payload(self):
        return dict(kind="actor", spec=self.config.spec, fingerprint=self.config.fingerprint,
                    run_id=getattr(self, "run_id", None),
                    config=self.config.to_dict(), step=self.global_optimizer_step,
                    model_state_dict={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()})

    def export_actor(self, path):
        # Finite training parameters do not guarantee finite evaluation (notably
        # with running BN statistics). Reject a broken snapshot before publishing.
        from ..engine import GameState
        from ..model.neural_eval import NeuralEvaluator
        was_training = self.model.training
        try:
            self.model.eval()
            NeuralEvaluator(self.model, self.device, self.config.precision,
                            self.config.channels_last).evaluate(GameState.new_game())
        finally:
            self.model.train(was_training)
        atomic_torch_save(self.actor_payload(), path)

    def save_checkpoint(self, path=None, extra=None):
        payload = self.actor_payload()
        payload.update(kind="learner", optimizer_type="SGD",
                       optimizer_state_dict=self.optimizer.state_dict(),
                       current_iteration=self.current_iteration,
                       python_rng_state=random.getstate(), numpy_rng_state=np.random.get_state(),
                       torch_rng_state=torch.get_rng_state(),
                       cuda_rng_states=[s.cpu() for s in torch.cuda.get_rng_state_all()]
                       if str(self.device).startswith("cuda") else None,
                       extra=extra or {})
        path = path or self.config.checkpoint_path / "learner.pt"
        atomic_torch_save(payload, path)
        return str(path)

    def load_checkpoint(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("kind") != "learner" or state.get("optimizer_type") != "SGD":
            raise ValueError("not a versioned SGD learner checkpoint; use explicit weight import/migration")
        if state.get("fingerprint") != self.config.fingerprint:
            raise ValueError("checkpoint model/rule/observation/target specification mismatch")
        for key in ("batch_size", "loss_reduction", "learning_rate", "lr_momentum", "weight_decay",
                    "lr_milestones", "lr_decay_rate", "c_puct", "pb_c_base",
                    "dirichlet_alpha", "dirichlet_epsilon", "mcts_batch_size", "virtual_loss",
                    "replay_window_games", "seed", "checkpoint_interval", "num_sampling_moves", "reuse_tree"):
            if state["config"].get(key) != self.config.to_dict().get(key):
                raise ValueError(f"resume changes training semantics: {key}")
        self.model.load_state_dict(state["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.global_optimizer_step = state["step"]
        self.current_iteration = state["current_iteration"]
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if str(self.device).startswith("cuda") and state.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng_states"]])
        # Search budget changes affect future self-play, not checkpoint tensor
        # compatibility. Keep the caller's config and record the transition.
        previous = json.loads(json.dumps(state["config"]))
        current = json.loads(json.dumps(self.config.to_dict()))
        self.resume_config_changes = {
            key: dict(old=previous.get(key), new=value)
            for key, value in current.items() if previous.get(key) != value
        }
        return state.get("extra", {})
