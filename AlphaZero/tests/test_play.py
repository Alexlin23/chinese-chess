"""Real checkpoint load plus authoritative browser-play state checks."""
from types import SimpleNamespace
import threading
import time
import pytest
import torch
from AlphaZero.engine import GameState
from AlphaZero.train.config import AlphaZeroConfig
from AlphaZero.train.trainer import Trainer
from AlphaZero.train.replay import ReplayBuffer, pack_sample
from AlphaZero.model.player import ModelPlayer
from AlphaZero.scripts.play_server import PlaySession


def test_player_rejects_initialization_and_plays_from_sgd_checkpoint(tmp_path):
    torch.set_num_threads(1)
    trainer = Trainer(AlphaZeroConfig(num_blocks=1, num_filters=4, batch_size=2,
                                     device="cpu", precision="fp32"))
    path = tmp_path / "actor.pt"
    trainer.export_actor(path)
    with pytest.raises(ValueError, match="随机初始化"):
        ModelPlayer(path)
    state = GameState.new_game()
    pi = state.legal_mask() / state.legal_mask().sum()
    replay = ReplayBuffer()
    replay.add_game("unit", [pack_sample(state, pi, 0)])
    trainer.step(replay)
    trainer.export_actor(path)
    player = ModelPlayer(path, simulations=16)
    move, diagnostics = player.choose(state)
    assert player.step == 1 and state.is_legal(move)
    assert diagnostics["simulations"] == 16
    replay.close()


def test_abandoned_ai_reply_cannot_overwrite_new_game():
    entered, release = threading.Event(), threading.Event()
    class Player:
        config = SimpleNamespace(max_game_ply=512)
        step, simulations = 10, 16
        def choose(self, state):
            entered.set()
            assert release.wait(3)
            return state.legal_moves()[0], {}
    session = PlaySession(Player())
    try:
        first = session.reset(-1)
        assert entered.wait(3)
        second = session.reset(1)
        assert first["game_id"] != second["game_id"]
        release.set()
        session.pool.submit(lambda: None).result(timeout=3)
        assert session.snapshot()["ply"] == 0
        assert not session.snapshot()["thinking"]
        with pytest.raises(ValueError, match="棋局已更新"):
            session.move(first["game_id"], 0, [6, 0, 5, 0])
        current = session.snapshot()
        moved = session.move(current["game_id"], 0, [6, 0, 5, 0])
        assert moved["revision"] >= 1
        with pytest.raises(ValueError):
            session.move(current["game_id"], 0, [6, 2, 5, 2])
    finally:
        release.set()
        session.close()


def test_cuda_graph_padding_and_repeated_inputs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from AlphaZero.model.network import PolicyValueNetwork
    from AlphaZero.model.cuda_graph import CudaGraphForward
    model = PolicyValueNetwork(1, 4).cuda().eval()
    graph = CudaGraphForward(model, max_batch=8)
    for n in (3, 1, 5, 3):
        x = torch.randn(n, 131, 10, 9, device="cuda").contiguous(memory_format=torch.channels_last)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            expected = tuple(t.float().clone() for t in model(x))
        actual = graph(x)
        for got, want in zip(actual, expected):
            assert torch.allclose(got.float(), want, atol=.01, rtol=.02)
