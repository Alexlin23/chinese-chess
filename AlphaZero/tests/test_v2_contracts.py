"""Regression checks for the rebuilt learning contracts."""
import json
import urllib.request
import numpy as np
import pytest
import torch
from AlphaZero.engine import GameState
from AlphaZero.engine.constants import KING, ROOK, POLICY_SIZE, BLACK
from AlphaZero.search import MCTS
from AlphaZero.train.config import AlphaZeroConfig
from AlphaZero.train.trainer import Trainer
from AlphaZero.train.replay import ReplayBuffer, pack_sample
from AlphaZero.train.live import LiveTraining
from AlphaZero.tests.test_mcts_correctness import ConstantEvaluator

def test_visit_conservation_large_batch_and_exception():
    search = MCTS(ConstantEvaluator(), num_simulations=64, batch_size=64)
    pi = search.search(GameState.new_game(), add_exploration_noise=False)
    assert search.last_root.visit_count == sum(c.visit_count for c in search.last_root.children.values()) == 64
    assert np.isclose(pi.sum(), 1)
    class Failure(ConstantEvaluator):
        def evaluate_batch(self, states):
            raise RuntimeError("injected")
    search = MCTS(Failure(), num_simulations=64, batch_size=64)
    with pytest.raises(RuntimeError):
        search.search(GameState.new_game())
    assert search.last_root.visit_count == 0
    assert all(c.visit_count == 0 and c.total_value == 0 for c in search.last_root.children.values())

def test_terminal_cannot_return_action_and_mate_precedes_cap():
    board=np.zeros((10,9),dtype=np.int8)
    board[0,4]=-KING;board[9,3]=KING
    board[1,0]=ROOK;board[2,3]=ROOK;board[2,6]=4
    state=GameState(board,True,max_ply=1)
    from AlphaZero.engine import Move
    finished=state.apply(Move(1,0,1,4))
    assert finished.game_result().winner == 1
    assert finished.game_result().reason == "no_legal_move"
    move,pi=MCTS(ConstantEvaluator()).select_move(finished,temperature=0)
    assert move is None and not pi.any()

def test_color_history_and_owned_snapshot():
    board=GameState.new_game().board.copy()
    state=GameState(board,BLACK)
    board[:]=0
    assert not state.turn and state.board.any()
    assert state.encode().shape == (131,10,9)
    assert state.encode()[128].sum()==0
    following=state.apply(state.legal_moves()[0])
    assert len(following.history)==2
    assert following.encode()[16:30].sum()>0
    with pytest.raises(ValueError):
        state.board[:]=0

def test_checkpoint_roundtrip_optimizer_clock_and_reject_legacy(tmp_path):
    config=AlphaZeroConfig(num_blocks=1,num_filters=4,batch_size=2,
        lr_milestones=(1,),device="cpu",precision="fp32")
    replay=ReplayBuffer()
    state=GameState.new_game()
    p=state.legal_mask()/state.legal_mask().sum()
    replay.add_game("one",[pack_sample(state,p,1)])
    trainer=Trainer(config)
    trainer.step(replay)
    path=tmp_path/"learner.pt"
    trainer.save_checkpoint(path)
    restored=Trainer(config)
    restored.load_checkpoint(path)
    assert restored.global_optimizer_step==1
    expected=trainer.step(replay)
    actual=restored.step(replay)
    assert actual["lr"]==pytest.approx(config.learning_rate*.1)
    assert expected["loss"]==pytest.approx(actual["loss"],abs=1e-6)
    assert all(torch.equal(a,b) for a,b in zip(trainer.model.parameters(),restored.model.parameters()))
    torch.save(dict(kind="learner",optimizer_type="Adam"),tmp_path/"old.pt")
    with pytest.raises(ValueError,match="SGD"):
        restored.load_checkpoint(tmp_path/"old.pt")
    replay.close()

def test_unknown_config_rejected(tmp_path):
    path=tmp_path/"wrong.yaml"
    path.write_text("training:\n  lr: 0.2\n")
    with pytest.raises(ValueError,match="obsolete"):
        AlphaZeroConfig.from_yaml(path)
    with pytest.raises(ValueError):
        AlphaZeroConfig(mcts_batch_size=64,inference_batch_size=8)

def test_live_snapshot_any_number_boards_and_stale_events():
    live=LiveTraining(port=0,interval=.01)
    try:
        for worker in range(25):
            live.event(dict(kind="board",worker=worker,episode=f"w{worker}",sequence=0,
                            ply=0,board=GameState.new_game().board.tolist(),turn=1))
        host,port=live.address
        with urllib.request.urlopen(f"http://{host}:{port}/api/snapshot") as response:
            state=json.load(response)
        assert len(state["boards"])==25
        live.complete(dict(worker=0,sequence=0,episode="w0",reason="max_ply",winner=None,ply=4),10,1)
        live.event(dict(kind="board",worker=0,episode="w0",sequence=0,ply=3))
        assert "0" not in live.snapshot()["boards"]
        with urllib.request.urlopen(f"http://{host}:{port}/events") as response:
            assert response.readline().startswith(b"data: ")
    finally:
        live.close()


def test_inference_snapshot_pinning_and_nonfinite_failure(tmp_path):
    import time
    from AlphaZero.train.inference_server import InferenceServer
    from AlphaZero.train.self_play_worker import BatchRemoteEvaluator
    config=AlphaZeroConfig(num_blocks=1,num_filters=4,num_workers=2,
        batch_size=2,device="cpu",precision="fp32")
    trainer=Trainer(config)
    path=tmp_path/"actor.pt"
    trainer.export_actor(path)
    server=InferenceServer(path,config)
    server.start()
    first=BatchRemoteEvaluator(0,server.request_queue,server.response_queues[0])
    second=BatchRemoteEvaluator(1,server.request_queue,server.response_queues[1])
    state=GameState.new_game()
    try:
        p0,v0=first.evaluate(state)
        with torch.no_grad():
            trainer.model.value_fc2.bias.add_(0.7)
        trainer.global_optimizer_step=1
        trainer.export_actor(path)
        server.refresh()
        for _ in range(20):
            second.pinned_step=None
            second.evaluate(state)
            if second.model_step==1:
                break
            time.sleep(.02)
        assert second.model_step==1
        p1,v1=first.evaluate(state)
        assert first.model_step==0 and v1==pytest.approx(v0)
        np.testing.assert_array_equal(p0,p1)
        trainer.global_optimizer_step=2
        with torch.no_grad():
            next(trainer.model.parameters()).view(-1)[0]=float("nan")
        with pytest.raises(FloatingPointError, match="non-finite"):
            trainer.export_actor(path)
        assert torch.load(path, weights_only=False)["step"] == 1
        # Deliberately bypass publication validation to test the server's own boundary.
        torch.save(trainer.actor_payload(), path)
        server.refresh()
        failed=False
        for _ in range(20):
            second.pinned_step=None
            try:
                second.evaluate(state)
            except RuntimeError as exc:
                assert "non-finite" in str(exc)
                failed=True
                break
            time.sleep(.02)
        assert failed
    finally:
        server.stop()


def test_corrupt_replay_rejected_at_sampling_boundary():
    state=GameState.new_game()
    replay=ReplayBuffer()
    p=state.legal_mask()/state.legal_mask().sum()
    packed=pack_sample(state,p,0)
    replay.add_game("g",[packed])
    replay.db.execute("UPDATE positions SET probabilities=?", (bytes(len(packed.probabilities)),))
    replay.db.commit()
    with pytest.raises(ValueError,match="policy"):
        replay.sample(1)
    replay.close()

def test_pipeline_rejects_cross_run_resume_before_starting_actors(tmp_path):
    from AlphaZero.scripts.pipeline import pipeline
    config=AlphaZeroConfig(num_blocks=1,num_filters=4,batch_size=2,
                            device="cpu",precision="fp32")
    first=ReplayBuffer(tmp_path/"a"/"replay.sqlite")
    first.bind_spec(config.fingerprint)
    trainer=Trainer(config)
    checkpoint=tmp_path/"a"/"learner.pt"
    trainer.save_checkpoint(checkpoint,first.recovery_marker())
    first.close()
    with pytest.raises(ValueError,match="run or committed"):
        pipeline(config,resume=checkpoint,output_dir=tmp_path/"b",enable_monitor=False)


def test_forced_reply_releases_all_reservations_and_tree_reuse():
    board=np.zeros((10,9),dtype=np.int8)
    for r,c,p in [(0,4,-KING),(9,3,KING),(1,0,ROOK),(2,3,ROOK)]:
        board[r,c]=p
    state=GameState(board,False)
    assert len(state.legal_moves())==1
    search=MCTS(ConstantEvaluator(),num_simulations=64,batch_size=16,reuse_tree=True)
    move,pi=search.select_move(state,add_exploration_noise=False)
    assert search.last_root.visit_count==sum(c.visit_count for c in search.last_root.children.values())==64
    retained=sum(c.visit_count for c in search._reuse_root.children.values())
    search.search(state.apply(move),add_exploration_noise=False)
    assert search.diagnostics["reused_visits"]==retained
    assert search.diagnostics["simulations"]==64
    assert search.last_root.visit_count==sum(c.visit_count for c in search.last_root.children.values())

def test_perpetual_checker_loses_even_when_current_player_wins():
    from AlphaZero.engine import Move
    board=np.zeros((10,9),dtype=np.int8)
    board[0,4]=-KING;board[9,3]=KING;board[1,4]=ROOK
    state=GameState(board,BLACK)
    cycle=[Move(0,4,0,5),Move(1,4,1,5),Move(0,5,0,4),Move(1,5,1,4)]
    for move in cycle*2:
        state=state.apply(move)
    result=state.game_result()
    assert result.reason=="perpetual_check" and result.winner==BLACK
    assert result.to_value(False)==1
