"""Check the offline teacher against the real engine, independent of GPU/model quality."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from AlphaZero.engine import ActionEncoder
from AlphaZero.tactics.data import CATEGORIES, DTYPE, decode, records, verify,CombinedRecords
from AlphaZero.tactics.evaluate import evaluate, evaluation_ids
from AlphaZero.train.replay import _MIRROR_ACTION_IDS


@pytest.fixture(scope='module')
def generated(tmp_path_factory):
    if shutil.which('g++') is None:
        pytest.skip('independent C++ generator needs g++')
    root=tmp_path_factory.mktemp('tactics')
    source=Path(__file__).resolve().parents[1]/'tactics/generator.cpp'
    subprocess.run(['g++','-O2','-std=c++17',str(source),'-o',str(root/'generator')],check=True)
    subprocess.run([str(root/'generator'),str(root/'rows.bin'),'600','179','0'],check=True,capture_output=True)
    return records(root/'rows.bin')


def test_all_classes_proof_legality_and_runtime_inputs(generated):
    assert DTYPE.itemsize==1784
    assert set(generated['category'])==set(range(6))
    for cat in range(6):
        ids=np.flatnonzero(generated['category']==cat)[:6]
        h=generated['history_length'][generated['category']==cat]
        assert np.bincount(h,minlength=9).max()/len(h)<.5
        for i in ids:
            row=generated[i];state=verify(row)
            x,pi,z,pw,vw=decode(generated[i:i+1])
            np.testing.assert_array_equal(x[0],state.encode())
            assert np.isclose(pi.sum(),1)
            assert (vw[0]>0)==(cat<3)
            assert (pw[0]==0)==(cat==2)
            if cat>=3:assert vw[0]==0  # Unconditional tastes are never declared wins.
    assert {int(x) for x in generated['turn']}=={-1,1}
    assert len(set(generated['key'].tolist()))==len(generated)
    for sign in (-1,1):
        h=generated['history_length'][generated['value']==sign]
        assert set(h)==set(range(1,9))
        assert .25<float(np.mean(h%2==0))<.75


def test_mirror_changes_targets_and_board_together(generated):
    class AlwaysMirror:
        def random(self,n):return np.zeros(n)
    batch=generated[:12]
    a,pi,*_=decode(batch)
    b,other,*_=decode(batch,mirror=True,rng=AlwaysMirror())
    np.testing.assert_array_equal(b,a[:,:,:,::-1])
    np.testing.assert_array_equal(other[:,_MIRROR_ACTION_IDS],pi)


def test_combined_records_keep_index_order_and_fields(generated):
    combined=CombinedRecords([generated[:21],generated[50:80]])
    assert len(combined)==51
    expected=np.concatenate([generated[:21],generated[50:80]])
    np.testing.assert_array_equal(combined[[50,0,21,20]],expected[[50,0,21,20]])
    np.testing.assert_array_equal(combined['category'],expected['category'])
    np.testing.assert_array_equal(combined[21],expected[21])


def test_runtime_audit_rejects_corrupt_legal_list(generated):
    row=generated[0].copy();row['legal_count']=0
    with pytest.raises(ValueError,match='legal masks disagree'):verify(row)


def test_audit_rejects_negative_probabilities_and_fake_draw_labels(generated):
    row=generated[np.flatnonzero(generated['count']>=2)[0]].copy()
    row['probabilities'][:int(row['count'])]=0
    row['probabilities'][:2]=[-.5,1.5]
    with pytest.raises(ValueError,match='policy distribution'):verify(row)
    row=generated[np.flatnonzero(generated['category']==4)[0]].copy();row['value_weight']=1
    with pytest.raises(ValueError,match='false value label'):verify(row)


def test_evaluation_legal_mask_and_known_value_masks(generated):
    class Teacher(torch.nn.Module):
        def __init__(self):super().__init__();self.offset=0
        def forward(self,x):
            batch=selected[self.offset:self.offset+len(x)];self.offset+=len(x)
            _,pi,z,_,_=decode(batch)
            return torch.tensor(pi),torch.tensor(z)
    ids=evaluation_ids(generated,8);selected=generated[ids]
    result=evaluate(Teacher(),generated,ids,device='cpu',batch_size=7)
    assert result['selection_score']==1
    assert all(result[name]['policy_top1']==1 for name in CATEGORIES)
    assert result['value_win']['n']==16 and result['value_loss']['n']==8
    assert result['value_win']['mse']==result['value_loss']['mse']==0


def test_actual_match_terminal_result_and_legal_opponent(generated):
    from AlphaZero.scripts.play_tactics import play,opponent
    from AlphaZero.tactics.data import state_from_record,mate1_moves
    class MatingPlayer:
        step=1;simulations=1
        def choose(self,state):return mate1_moves(state)[0],dict(simulations=1)
    row=generated[np.flatnonzero(generated['category']==0)[0]]
    state=state_from_record(row)
    report=play(MatingPlayer(),state,state.current_player,np.random.default_rng(4),'mate')
    assert report['outcome']=='win' and report['plies']==1
    assert json.loads(json.dumps(report))['initial_history']
    reply=opponent(state,np.random.default_rng(4))
    assert state.apply(reply).game_result().winner==state.current_player
    from AlphaZero.scripts.analyze_tactics_games import analyze
    report['initial_record']=int(np.flatnonzero(generated['category']==0)[0])
    audit=analyze(dict(games=[report]),generated)
    assert audit['games'][0]['first_target_hit']
    assert audit['games'][0]['immediate_win_not_taken']==[]


def test_fresh_cpu_pretraining_exports_playable_actor_and_fresh_sgd(generated,tmp_path):
    from AlphaZero.train.config import AlphaZeroConfig,ACTION_VERSION,RULE_VERSION
    from AlphaZero.train.trainer import Trainer
    from AlphaZero.model.player import ModelPlayer
    from AlphaZero.engine import GameState
    data=tmp_path/'data';data.mkdir()
    for split in ('train','validation','test','ood'):
        generated[:120].tofile(data/(split+'.bin'))
    (data/'manifest.json').write_text(json.dumps(dict(record_bytes=DTYPE.itemsize,teacher_version='history-balanced-v2',
        action_version=ACTION_VERSION,rule_version=RULE_VERSION,seed=179)))
    config=AlphaZeroConfig(num_blocks=1,num_filters=8,max_game_ply=180,batch_size=32,
        checkpoint_dir=str(tmp_path/'run'),inference_cuda_graph=False,precision='fp32')
    import yaml
    cfg=tmp_path/'config.yaml';cfg.write_text(yaml.safe_dump(config.to_dict()))
    command=[sys.executable,'-m','AlphaZero.scripts.pretrain_tactics','--data-root',str(data),
        '--dataset','',
        '--run-dir',config.checkpoint_dir,'--config',str(cfg),'--epochs','1','--warmup-epochs','0',
        '--batch-size','32','--device','cpu','--eval-per-category','4','--no-monitor']
    subprocess.run(command,check=True,capture_output=True)
    run=Path(config.checkpoint_dir)
    report=json.loads((run/'pretraining-results.json').read_text())
    assert report['self_play_started'] is False and report['completed_updates']==4
    player=ModelPlayer(run/'latest.pt',simulations=2)
    move,_=player.choose(GameState.new_game(max_ply=180))
    assert GameState.new_game().is_legal(move)
    learner=Trainer(config);learner.load_checkpoint(run/'learner.pt')
    assert learner.global_optimizer_step==0 and learner.model_step==4
    assert learner.optimizer.state=={}  # Adam moments are never mistaken for SGD momentum.
    resumed=subprocess.run(command+['--resume'],check=True,capture_output=True)
    assert resumed.returncode==0
    assert json.loads((run/'pretraining-results.json').read_text())['baseline']==report['baseline']
    # An explicit new curriculum retains actor weights/version, and never resets to random.
    staged=command+['--initial-checkpoint',str(run/'latest.pt'),'--phase','bridge','--max-train-rows','64']
    subprocess.run(staged,check=True,capture_output=True)
    continuation=json.loads((run/'pretraining-results.json').read_text())
    assert continuation['completed_updates']==6 and continuation['stage_updates']==2
    assert (run/'pretraining-phase-history.jsonl').exists()
    subprocess.run(command+['--restart-random','--phase','corrected'],check=True,capture_output=True)
    restarted=json.loads((run/'pretraining-results.json').read_text())
    assert restarted['completed_updates']==4 and restarted['stage_updates']==4
    assert json.loads((run/'pretraining-control.json').read_text())['status']=='pretraining_finished'


@pytest.mark.skipif(sys.platform!='linux',reason='WSL process controller uses /proc')
def test_controller_identifies_pretraining_by_actual_open_run_lock(tmp_path):
    code="import pathlib,signal,sys; f=open(pathlib.Path(sys.argv[1])/'.run.lock','a+'); print('ready',flush=True); signal.pause()"
    child=subprocess.Popen([sys.executable,'-c',code,str(tmp_path),'AlphaZero.scripts.pretrain_tactics'],
                           stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        assert child.stdout.readline().strip()=='ready'
        base=[sys.executable,'-m','AlphaZero.scripts.training_control']
        status=subprocess.run(base+['status','--run-dir',str(tmp_path)],capture_output=True,text=True,check=True)
        assert json.loads(status.stdout)['pids']==[child.pid]
        other=tmp_path/'other';other.mkdir()
        status=subprocess.run(base+['status','--run-dir',str(other)],capture_output=True,text=True,check=True)
        assert json.loads(status.stdout)['pids']==[]
        subprocess.run(base+['stop','--run-dir',str(tmp_path)],capture_output=True,check=True)
        assert child.wait(timeout=5)!=0
    finally:
        if child.poll() is None:child.terminate();child.wait(timeout=5)
