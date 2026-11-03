"""Local play graph preserves coordinates, validation, ownership and worker use."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
import torch

from AlphaZero.engine import GameState
from AlphaZero.engine.constants import POLICY_SIZE, RED
from AlphaZero.model.cuda_graph import CudaGraphForward
from AlphaZero.model.neural_eval import CudaGraphEvaluator, NeuralEvaluator
from AlphaZero.model.network import PolicyValueNetwork
from AlphaZero.model.player import ModelPlayer
from AlphaZero.train.config import AlphaZeroConfig
from AlphaZero.train.trainer import Trainer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_play_graph_matches_previous_path_and_preserves_old_responses(precision):
    torch.manual_seed(13)
    model = PolicyValueNetwork(1, 8).cuda().eval().to(memory_format=torch.channels_last)
    old = NeuralEvaluator(model, "cuda", precision, True,
                          CudaGraphForward(model, precision, True))
    new = CudaGraphEvaluator(model, "cuda", precision, True)
    red = GameState.new_game()
    black = red.apply(red.legal_moves()[0])
    previous = []
    for count in (3, 1, 5, 2):
        states = [red, black, red, black, red][:count]
        expected = old.evaluate_batch(states)
        actual = new.evaluate_batch(states)
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got, want, atol=2e-3 if precision == "bf16" else 2e-6,
                                       rtol=2e-3 if precision == "bf16" else 2e-6)
        for state, policy in zip(states, actual[0]):
            assert (policy[~state.legal_mask()] == 0).all()
            np.testing.assert_allclose(policy.sum(), 1, atol=1e-6)
        previous.append((actual, tuple(array.copy() for array in actual)))
    for response, frozen in previous:
        for got, want in zip(response, frozen):
            np.testing.assert_array_equal(got, want)
    with pytest.raises(ValueError, match="empty inference batch"):
        new.evaluate_batch([])
    with pytest.raises(ValueError, match="empty legal set"):
        new.evaluate(GameState(np.zeros((10, 9), np.int8), RED))


@pytest.mark.parametrize("invalid", ["illegal_logit", "value"])
def test_play_graph_rejects_invalid_outputs_even_outside_legal_mask(invalid):
    class InvalidNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1, device="cuda"))

        def forward(self, states):
            logits = torch.zeros(len(states), POLICY_SIZE, device="cuda") + self.weight
            values = torch.zeros(len(states), device="cuda")
            if invalid == "illegal_logit":
                logits[:, 1] = torch.nan
            else:
                values[:] = torch.nan
            return logits, values

    state = GameState.new_game()
    assert not state.canonical_mask()[1]
    evaluator = CudaGraphEvaluator(InvalidNetwork().eval())
    with pytest.raises(FloatingPointError, match="non-finite"):
        evaluator.evaluate(state)


def test_player_warms_graph_then_uses_it_in_background_thread_and_reloads(tmp_path):
    torch.set_num_threads(1)
    trainer = Trainer(AlphaZeroConfig(num_blocks=1, num_filters=4, device="cpu",
                                     precision="bf16", checkpoint_interval=1))
    trainer.global_optimizer_step = 1
    path = tmp_path / "actor.pt"
    trainer.export_actor(path)
    player = ModelPlayer(path, "cuda", simulations=8, exploration_noise=True, search_noise_depth=1)
    assert isinstance(player.evaluator, CudaGraphEvaluator)
    state = GameState.new_game()
    with ThreadPoolExecutor(max_workers=1) as pool:
        move, diagnostics = pool.submit(player.choose, state).result(timeout=20)
        assert state.is_legal(move) and diagnostics["simulations"] == 8
        trainer.global_optimizer_step = 2
        trainer.export_actor(path)
        updated = ModelPlayer(path, "cuda", simulations=8)
        assert updated.step == 2 and updated.evaluator.graph is not player.evaluator.graph
        next_state = state.apply(move)
        move, diagnostics = pool.submit(updated.choose, next_state).result(timeout=20)
        assert next_state.is_legal(move) and diagnostics["simulations"] == 8
