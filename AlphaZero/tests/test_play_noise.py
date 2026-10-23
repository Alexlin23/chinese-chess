"""Human-play exploration is explicit and survives session resets."""
import importlib
import pytest
import torch

from AlphaZero.engine import GameState
from AlphaZero.model.player import ModelPlayer
from AlphaZero.scripts.play_server import PlaySession
from AlphaZero.train.config import AlphaZeroConfig
from AlphaZero.train.trainer import Trainer


@pytest.fixture
def actor(tmp_path):
    torch.set_num_threads(1)
    trainer = Trainer(AlphaZeroConfig(num_blocks=1, num_filters=4, device="cpu", precision="fp32"))
    trainer.global_optimizer_step = 1
    path = tmp_path / "actor.pt"
    trainer.export_actor(path)
    return path


@pytest.mark.parametrize("enabled", [False, True])
def test_player_passes_explicit_noise_configuration_to_real_search(actor, monkeypatch, enabled):
    module = importlib.import_module("AlphaZero.model.player")
    real_mcts = module.MCTS
    seen = []
    class Search(real_mcts):
        def select_move(self, state, temperature=1, add_exploration_noise=True):
            seen.append((self.search_noise_depth, self.dirichlet_alpha, self.dirichlet_epsilon,
                         temperature, add_exploration_noise))
            return super().select_move(state, temperature, add_exploration_noise)
    monkeypatch.setattr(module, "MCTS", Search)
    player = ModelPlayer(actor, simulations=12, exploration_noise=enabled, search_noise_depth=3)
    state = GameState.new_game()
    move, diagnostics = player.choose(state)
    assert state.is_legal(move) and diagnostics["simulations"] == 12
    assert seen == [(3, .3, .25, 0, enabled)]
    session = PlaySession(player)
    try:
        assert session.snapshot()["exploration_noise"] is enabled
        session.reset(1)
        assert session.snapshot()["exploration_noise"] is enabled
        assert session.snapshot()["search_noise_depth"] == 3
    finally:
        session.close()


@pytest.mark.parametrize("bad", [-1, True, 1.5])
def test_invalid_noise_depth_fails_before_loading_model(bad):
    with pytest.raises(ValueError, match="search_noise_depth"):
        ModelPlayer("missing.pt", search_noise_depth=bad)
