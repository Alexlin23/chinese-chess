"""Executable contracts extracted from the supplied Data S1, without importing it."""
import ast
from pathlib import Path
import numpy as np
import pytest
import torch
from AlphaZero.engine import GameState
from AlphaZero.engine.constants import POLICY_SIZE
from AlphaZero.search import MCTS
from AlphaZero.search.node import MCTSNode
from AlphaZero.train.config import AlphaZeroConfig
from AlphaZero.train.self_play_worker import selfplay_temperature
from AlphaZero.train.trainer import Trainer
from AlphaZero.tests.test_mcts_correctness import ConstantEvaluator


def test_reference_constants_and_default_execution():
    reference = Path(__file__).resolve().parents[2] / "aar6404_datas1/pseudocode.py"
    tree = ast.parse(reference.read_text())
    constants = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                try:
                    constants[target.attr] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    config = AlphaZeroConfig()
    for actual, reference_key in [("num_sampling_moves", "num_sampling_moves"),
            ("num_simulations", "num_simulations"), ("batch_size", "batch_size"),
            ("weight_decay", "weight_decay"), ("lr_momentum", "momentum"),
            ("pb_c_base", "pb_c_base"), ("c_puct", "pb_c_init")]:
        assert getattr(config, actual) == constants[reference_key]
    assert config.checkpoint_interval == 1000 and config.max_optimizer_steps == 700000
    assert config.mcts_batch_size == 1 and config.virtual_loss == 0 and not config.reuse_tree
    assert "training_games" not in config.to_dict()
    assert config.spec["loss_reduction"] == "mean"
    assert selfplay_temperature(29) == 1 and selfplay_temperature(30) == 0
    assert selfplay_temperature(511) == 0


def test_fresh_root_and_exact_sequential_budget():
    state = GameState.new_game()
    search = MCTS(ConstantEvaluator(), num_simulations=32)
    move, pi = search.select_move(state, temperature=0, add_exploration_noise=False)
    assert np.count_nonzero(pi) > 1  # greedy action must not turn target into one-hot
    assert search._reuse_root is None
    assert search.last_root.visit_count == 32
    search.search(state.apply(move), add_exploration_noise=False)
    assert search.diagnostics["reused_visits"] == 0
    assert search.last_root.visit_count == 32


def test_zero_parent_visits_and_action_ties_follow_datas1():
    root = MCTSNode(1.0, children={2: MCTSNode(.99), 8: MCTSNode(.01)})
    assert root.select_child(1.25) == 8
    root.visit_count = 1
    assert root.select_child(1.25) == 2


def test_state_cache_preserves_search_and_avoids_reapplying_paths(monkeypatch):
    original = GameState.apply
    calls = []
    def counted(self, move):
        calls.append(move)
        return original(self, move)
    monkeypatch.setattr(GameState, "apply", counted)
    results = []
    for cached in (False, True):
        calls.clear()
        search = MCTS(ConstantEvaluator(), num_simulations=128,
                      rng=np.random.default_rng(19), cache_states=cached)
        pi = search.search(GameState.new_game())
        results.append((pi, len(calls), search.last_root.total_value))
    np.testing.assert_array_equal(results[0][0], results[1][0])
    assert results[0][2] == results[1][2]
    assert results[1][1] <= 128 < results[0][1]


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_data_gradient_and_tensorflow_half_l2(reduction):
    config = AlphaZeroConfig(num_blocks=1, num_filters=4, batch_size=3, device="cpu", precision="fp32", loss_reduction=reduction)
    trainer = Trainer(config)
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.linspace(-.1, .1, POLICY_SIZE))
            self.value = torch.nn.Parameter(torch.tensor(.2))
        def forward(self, x):
            return self.logits.expand(len(x), -1), self.value.expand(len(x))
    trainer.model = trainer.forward_model = Tiny()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=config.learning_rate,
                                         momentum=config.lr_momentum, weight_decay=config.weight_decay)
    pi = np.zeros((3, POLICY_SIZE), np.float32)
    pi[:, 4] = 1
    z = np.ones(3, np.float32)
    class Batch:
        def __len__(self): return 3
        def sample(self, n):
            return np.zeros((3, 131, 10, 9), np.float32), pi, z, np.ones(3)
    before = [p.detach().clone() for p in trainer.model.parameters()]
    scale = 3 if reduction == "sum" else 1
    data = -scale * torch.log_softmax(trainer.model.logits, 0)[4] + scale * (trainer.model.value - 1).square()
    regularizer = config.weight_decay / 2 * sum(p.square().sum() for p in trainer.model.parameters())
    grads = torch.autograd.grad(data + regularizer, tuple(trainer.model.parameters()))
    stats = trainer.step(Batch())
    assert stats["data_loss"] == pytest.approx(float(data.detach()), rel=1e-6)
    assert stats["regularization"] == pytest.approx(float(regularizer.detach()), rel=1e-6)
    for old, grad, actual in zip(before, grads, trainer.model.parameters()):
        assert torch.allclose(actual, old - config.learning_rate * grad, atol=1e-7)
