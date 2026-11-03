"""Real spawned shared transport preserves batches, responses and buffer ownership."""
import numpy as np
import pytest
import torch
from AlphaZero.engine import GameState
from AlphaZero.train.config import AlphaZeroConfig
from AlphaZero.train.trainer import Trainer
from AlphaZero.train.inference_server import InferenceServer
from AlphaZero.train.self_play_worker import BatchRemoteEvaluator


def test_shared_response_survives_subsequent_requests(tmp_path):
    torch.set_num_threads(1)
    cfg = AlphaZeroConfig(num_blocks=1, num_filters=4, device="cpu", precision="fp32",
                          num_workers=1, inference_batch_size=8, inference_shared_memory=True)
    trainer = Trainer(cfg)
    trainer.export_actor(tmp_path / "actor.pt")
    server = InferenceServer(tmp_path / "actor.pt", cfg)
    try:
        server.start()
        evaluator = BatchRemoteEvaluator(0, server.request_queue, server.response_queues[0])
        state = GameState.new_game()
        states = [state, state.apply(state.legal_moves()[0]), state]
        policies, values = evaluator.evaluate_batch(states)
        frozen_p, frozen_v = policies.copy(), values.copy()
        for count in (1, 5, 2, 8):
            p, v = evaluator.evaluate_batch([state] * count)
            assert p.shape == (count, policies.shape[1]) and v.shape == (count,)
            np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
        np.testing.assert_array_equal(policies, frozen_p)
        np.testing.assert_array_equal(values, frozen_v)
    finally:
        server.stop()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_shared_cuda_pipeline_publishes_and_resumes_with_actor_groups(tmp_path):
    from AlphaZero.scripts.pipeline import pipeline
    cfg = AlphaZeroConfig(num_blocks=1, num_filters=8, device="cuda", precision="bf16",
        num_workers=4, actor_processes=2, inference_batch_size=8,
        inference_shared_memory=True, inference_buffer_reuse=True, inference_cuda_graph=True,
        num_simulations=2, max_game_ply=4, batch_size=4, training_microbatch_size=2,
        new_positions_per_update=8, checkpoint_interval=1, recovery_interval=1,
        max_optimizer_steps=2, max_iterations=2, telemetry_port=0)
    for expected in (1, 2):
        result = pipeline(cfg, output_dir=tmp_path, enable_monitor=False,
            resume=tmp_path / "learner.pt" if expected > 1 else None,
            stop_after_publications=1, performance=True, performance_interval=.05)
        assert result["step"] == expected
        learner = torch.load(tmp_path / "learner.pt", map_location="cpu", weights_only=False)
        actor = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=False)
        assert actor["step"] == learner["step"] == expected
        assert all(torch.equal(value, actor["model_state_dict"][key])
                   for key, value in learner["model_state_dict"].items())
