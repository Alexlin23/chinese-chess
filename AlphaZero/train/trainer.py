"""Continuous SGD learner: Scalar MSE, policy CE and half-L2, with explicit batch reduction contract."""
from pathlib import Path
import json
import os
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from .config import AlphaZeroConfig, RESUME_FIXED_FIELDS
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
        self.optimizer_step_offset = 0  # Preserve monotonic model versions if the SGD schedule is rebased.
        self.optimizer_step_changes = []
        self.current_iteration = 0
        self.resume_config_changes = {}
        self.performance = None  # Runtime observer; never serialized in checkpoints/config.

    def step(self, replay):
        performance = self.performance
        if performance is not None:
            step_started = time.perf_counter()
        self.model.train()
        decay = sum(self.global_optimizer_step >= milestone for milestone in self.config.lr_milestones)
        lr = self.config.learning_rate * self.config.lr_decay_rate ** decay
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        if performance is not None:
            sample_started = time.perf_counter()
        if self.config.mirror_augmentation:
            states, policies, values, _ = replay.sample(self.config.batch_size, mirror_augmentation=True)
        else:
            states, policies, values, _ = replay.sample(self.config.batch_size)
        if performance is not None:
            performance.observe("sgd.sample_wall", time.perf_counter() - sample_started)
            transfer_started = time.perf_counter()
            gpu_sample = performance.cuda_sample(self.device, ("sgd.transfer_cuda", "sgd.compute_cuda"))
        def transfer(array):
            tensor = torch.from_numpy(array)
            if str(self.device).startswith("cuda"):
                tensor = tensor.pin_memory()
            return tensor.to(self.device, non_blocking=True)
        x, pi, z = transfer(states), transfer(policies), transfer(values)
        if str(self.device).startswith("cuda") and self.config.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        if performance is not None:
            performance.observe("sgd.transfer_host_wall", time.perf_counter() - transfer_started)
            if gpu_sample is not None:
                gpu_sample.mark()
            compute_started = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        batch_size = len(states)
        microbatch = self.config.training_microbatch_size or batch_size
        policy_loss = torch.zeros((), device=self.device)
        value_loss = torch.zeros((), device=self.device)
        for start in range(0, batch_size, microbatch):
            end = min(start + microbatch, batch_size)
            with autocast_context(self.device, self.config.precision):
                logits, v = self.forward_model(x[start:end])
            # Full action CE is intentional; legal masking is conditional inference.
            policy_per_position = -(pi[start:end] * F.log_softmax(logits.float(), dim=1)).sum(1)
            chunk_policy = getattr(policy_per_position, self.config.loss_reduction)()
            chunk_value = F.mse_loss(v.float(), z[start:end], reduction=self.config.loss_reduction)
            # Weight short final chunks correctly; sum reduction already adds across samples.
            weight = (end - start) / batch_size if self.config.loss_reduction == "mean" else 1
            chunk_loss = (chunk_policy + chunk_value) * weight
            if not bool(torch.isfinite(chunk_loss)):
                raise FloatingPointError("non-finite learner loss")
            chunk_loss.backward()
            policy_loss += chunk_policy.detach() * weight
            value_loss += chunk_value.detach() * weight
            del logits, v, policy_per_position, chunk_policy, chunk_value, chunk_loss
        loss = policy_loss + value_loss
        regularization = (self.config.weight_decay / 2) * sum(
            p.detach().square().sum() for p in self.model.parameters())
        if not bool(torch.isfinite(loss + regularization)):
            raise FloatingPointError("non-finite learner loss")
        # Fail closed; clipping would introduce a separate algorithmic choice.
        finite = torch.stack([torch.isfinite(p.grad).all() for p in self.model.parameters()
                              if p.grad is not None]).all()
        if not bool(finite):
            raise FloatingPointError("non-finite learner gradients")
        self.optimizer.step()
        updated = list(self.model.parameters()) + list(self.model.buffers()) + [value for state in self.optimizer.state.values()
                    for value in state.values() if isinstance(value, torch.Tensor)]
        updated_finite = torch.stack([torch.isfinite(value).all() for value in updated]).all()
        if performance is not None and gpu_sample is not None:
            gpu_sample.mark()  # Before the existing bool/D2H synchronization.
        if not bool(updated_finite):
            raise FloatingPointError("non-finite parameters or optimizer state after SGD")
        self.global_optimizer_step += 1
        stats = dict(step=self.global_optimizer_step, loss=float(loss.detach() + regularization),
                    data_loss=float(loss.detach()), data_loss_per_position=float(loss.detach()) / (self.config.batch_size if self.config.loss_reduction == "sum" else 1),
                    regularization=float(regularization),
                    policy_loss=float(policy_loss.detach()), value_loss=float(value_loss.detach()),
                    lr=lr, replay_positions=len(replay), batch_size=batch_size,
                    training_microbatch_size=min(microbatch, batch_size),
                    accumulation_steps=(batch_size + microbatch - 1) // microbatch)
        if performance is not None:
            performance.observe("sgd.compute_wall", time.perf_counter() - compute_started)
            performance.observe("sgd.update_wall", time.perf_counter() - step_started)
            performance.increment("sgd.updates")
            if gpu_sample is not None:
                gpu_sample.finish()
        return stats

    def train(self, replay, updates=None):
        stats = {}
        for _ in range(updates or self.config.checkpoint_interval):
            stats = self.step(replay)
        self.current_iteration += 1
        return stats

    @property
    def model_step(self):
        return self.global_optimizer_step + self.optimizer_step_offset

    def actor_payload(self):
        return dict(kind="actor", spec=self.config.spec, fingerprint=self.config.fingerprint,
                    run_id=getattr(self, "run_id", None),
                    config=self.config.to_dict(), step=self.global_optimizer_step,
                    optimizer_step_offset=self.optimizer_step_offset,
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
        payload = self.actor_payload()
        payload["step"] = self.model_step
        payload.pop("optimizer_step_offset")
        # Omit learner/self-play settings so already-running human-play servers
        # can load newly published actor snapshots without new config fields.
        payload["config"].pop("replay_window_positions", None)
        payload["config"].pop("mirror_augmentation", None)
        payload["config"].pop("search_noise_depth", None)
        payload["config"].pop("new_positions_per_update", None)
        payload["config"].pop("training_microbatch_size", None)
        atomic_torch_save(payload, path)

    def save_checkpoint(self, path=None, extra=None):
        payload = self.actor_payload()
        extra = dict(extra or {})
        if self.optimizer_step_changes:
            extra["optimizer_step_changes"] = self.optimizer_step_changes
        payload.update(kind="learner", optimizer_type="SGD",
                       optimizer_state_dict=self.optimizer.state_dict(),
                       current_iteration=self.current_iteration,
                       python_rng_state=random.getstate(), numpy_rng_state=np.random.get_state(),
                       torch_rng_state=torch.get_rng_state(),
                       cuda_rng_states=[s.cpu() for s in torch.cuda.get_rng_state_all()]
                       if str(self.device).startswith("cuda") else None,
                       extra=extra)
        path = path or self.config.checkpoint_path / "learner.pt"
        atomic_torch_save(payload, path)
        return str(path)

    def load_checkpoint(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("kind") != "learner" or state.get("optimizer_type") != "SGD":
            raise ValueError("not a versioned SGD learner checkpoint; use explicit weight import/migration")
        if state.get("fingerprint") != self.config.fingerprint:
            raise ValueError("checkpoint model/rule/observation/target specification mismatch")
        for key in RESUME_FIXED_FIELDS:
            if state["config"].get(key) != self.config.to_dict().get(key):
                raise ValueError(f"resume changes training semantics: {key}")
        self.model.load_state_dict(state["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.global_optimizer_step = state["step"]
        self.optimizer_step_offset = state.get("optimizer_step_offset", 0)
        if type(self.optimizer_step_offset) is not int or self.optimizer_step_offset < 0:
            raise ValueError("invalid optimizer step offset")
        self.optimizer_step_changes = state.get("extra", {}).get("optimizer_step_changes", [])
        self.current_iteration = state["current_iteration"]
        if state["config"].get("checkpoint_interval") != self.config.checkpoint_interval:
            self.current_iteration = self.global_optimizer_step // self.config.checkpoint_interval
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if str(self.device).startswith("cuda") and state.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng_states"]])
        # Batch size, search effort/noise depth, publication cadence and replay retention may change on resume.
        # Keep the caller's config and log changes without resetting learner state.
        previous = json.loads(json.dumps(state["config"]))
        previous.setdefault("replay_window_positions", 0)
        previous.setdefault("mirror_augmentation", False)
        previous.setdefault("search_noise_depth", 1)
        previous.setdefault("new_positions_per_update", 0)
        previous.setdefault("training_microbatch_size", 0)
        current = json.loads(json.dumps(self.config.to_dict()))
        self.resume_config_changes = {
            key: dict(old=previous.get(key), new=value)
            for key, value in current.items() if previous.get(key) != value
        }
        return state.get("extra", {})
