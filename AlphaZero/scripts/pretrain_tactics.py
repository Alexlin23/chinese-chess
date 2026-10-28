"""Random-start supervised tactics curriculum; stop after pretraining, never launch self-play."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from AlphaZero.train.config import AlphaZeroConfig
from AlphaZero.train.trainer import create_model,atomic_torch_save,Trainer
from AlphaZero.train.replay import ReplayBuffer
from AlphaZero.train.live import LiveTraining
from AlphaZero.scripts.pipeline import run_lock
from AlphaZero.tactics.data import records,decode,DTYPE,CombinedRecords
from AlphaZero.tactics.evaluate import evaluate,evaluation_ids


def export_actor(model,config,run_id,step,path):
    values=config.to_dict()
    for key in ('mirror_augmentation','replay_window_positions','new_positions_per_update','search_noise_depth','training_microbatch_size'):
        values.pop(key,None)
    payload=dict(kind='actor',spec=config.spec,fingerprint=config.fingerprint,config=values,
                 run_id=run_id,step=step,model_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                 source='verified-tactics-supervised-pretraining')
    atomic_torch_save(payload,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',default='/mnt/d/project/Alphazero-data/policy-tactics/data')
    p.add_argument('--dataset',default='curriculum',help='verified dataset prefix')
    p.add_argument('--run-dir',default='/mnt/d/project/Alphazero-data/policy-tactics/run')
    p.add_argument('--config',default='config/local-train.yaml')
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--learning-rate',type=float,default=3e-4)
    p.add_argument('--warmup-epochs',type=int,default=2)
    p.add_argument('--seed',type=int,default=20261023)
    p.add_argument('--device',default='cuda',choices=('cpu','cuda'))
    p.add_argument('--port',type=int,default=8768)
    p.add_argument('--eval-per-category',type=int,default=512)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--no-monitor',action='store_true')
    p.add_argument('--additional-data',help='verified dataset prefix, e.g. dense, in the same data root')
    p.add_argument('--initial-checkpoint',help='explicit completed-stage actor to start a new AdamW curriculum stage')
    p.add_argument('--max-train-rows',type=int,default=0,help='per-epoch cap, balanced across data sources; 0 uses all')
    p.add_argument('--phase',default='initial')
    p.add_argument('--restart-random',action='store_true',help='explicitly replace experimental weights with a new random curriculum')
    args=p.parse_args()
    if (args.epochs<1 or args.batch_size<1 or args.eval_per_category<1
            or not 0<=args.warmup_epochs<=args.epochs):
        raise ValueError('positive epoch/batch/evaluation counts and valid warmup required')
    data=Path(args.data_root);directory=Path(args.run_dir);directory.mkdir(parents=True,exist_ok=True)
    prefix=args.dataset+'-' if args.dataset else ''
    manifest=json.loads((data/(prefix+'manifest.json')).read_text())
    if manifest.get('teacher_version')!='history-balanced-v2':raise ValueError('obsolete teacher history sampling; regenerate data')
    if manifest['record_bytes']!=DTYPE.itemsize:raise ValueError('dataset record version mismatch')
    train=records(data/(prefix+'train.bin'));valid=records(data/(prefix+'validation.bin'))
    if args.additional_data:
        extra=json.loads((data/(args.additional_data+'-manifest.json')).read_text())
        for field in ('record_bytes','action_version','rule_version','teacher_version'):
            if extra[field]!=manifest[field]:raise ValueError('additional dataset specification mismatch')
        manifest=dict(manifest,additional=extra)
        train=CombinedRecords([train,records(data/(args.additional_data+'-train.bin'))])
        valid=CombinedRecords([valid,records(data/(args.additional_data+'-validation.bin'))])
    if args.max_train_rows and args.max_train_rows<max(args.batch_size,2):
        raise ValueError('per-epoch cap must hold at least one batch and one sample per source')
    config=AlphaZeroConfig.from_yaml(args.config);config.checkpoint_dir=str(directory)
    if manifest['action_version']!=config.spec['action'] or manifest['rule_version']!=config.spec['rule']:
        raise ValueError('dataset/runtime specification mismatch')
    if config.max_game_ply!=manifest.get('max_game_ply',180):
        raise ValueError('tactical input normalization requires the dataset game-ply limit')
    torch.set_num_threads(1);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if args.device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA required for this run')
        torch.cuda.manual_seed_all(args.seed);torch.backends.cudnn.benchmark=True
    with run_lock(directory):
        control=dict(pid=os.getpid(),command=sys.argv,cwd=os.getcwd(),phase=args.phase,dataset=args.dataset,
                     started_unix=time.time(),expected_epochs=args.epochs,status='running',self_play_started=False)
        if Path('/proc/self/stat').exists():
            stat=Path('/proc/self/stat').read_text()
            control.update(start_ticks=int(stat[stat.rfind(')')+2:].split()[19]),
                           boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        control_path=directory/'pretraining-control.json';control_path.write_text(json.dumps(control,indent=2))
        try:
            run(args,config,manifest,train,valid,directory)
            result=json.loads((directory/'pretraining-results.json').read_text())
            control.update(status='pretraining_finished',run_id=result['run_id'],
                           completed_updates=result['completed_updates'],best_update=result['best_update'])
        except KeyboardInterrupt:
            control['status']='interrupted';raise
        except Exception as exc:
            control.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
        finally:
            control['ended_unix']=time.time();control_path.write_text(json.dumps(control,indent=2))


def run(args,config,manifest,train,valid,directory):
    device=args.device;model=create_model(config,device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4)
    replay=ReplayBuffer(directory/'replay.sqlite',config.replay_window_games,window_positions=config.replay_window_positions)
    replay.bind_spec(config.fingerprint);run_id=replay.run_id
    if replay.game_count!=0:raise ValueError('offline cold-start requires an empty self-play replay')
    replay.close()
    step=0;first_epoch=0;first_batch=0;best=-1.;best_epoch=0;history=[];initial_baseline=None
    checkpoint=directory/'pretraining.pt'
    dataset_id=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    training_identity=dict(seed=args.seed,batch_size=args.batch_size,epochs=args.epochs,
                           learning_rate=args.learning_rate,warmup_epochs=args.warmup_epochs,
                           phase=args.phase,dataset=args.dataset,additional_data=args.additional_data,max_train_rows=args.max_train_rows)
    if sum((args.resume,bool(args.initial_checkpoint),args.restart_random))>1:raise ValueError('choose one resume/restart mode')
    if checkpoint.exists() and not args.resume and not args.initial_checkpoint and not args.restart_random:
        raise ValueError('pretraining checkpoint exists; explicitly --resume')
    if args.resume:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['fingerprint']!=config.fingerprint or saved['run_id']!=run_id:raise ValueError('resume identity mismatch')
        if saved.get('dataset_id')!=dataset_id:raise ValueError('resume dataset changed')
        if saved.get('training_identity')!=training_identity:raise ValueError('resume training settings changed')
        model.load_state_dict(saved['model_state_dict']);optimizer.load_state_dict(saved['optimizer_state_dict'])
        step=saved['step'];first_epoch=saved['epoch'];first_batch=saved['next_batch'];best=saved['best_score'];best_epoch=saved['best_epoch']
        history=saved['history'];torch.set_rng_state(saved['torch_rng_state'])
        initial_baseline=saved.get('baseline')
        if initial_baseline is None:
            # Older interrupted runs recorded this before their first update.
            for line in (directory/'pretraining-metrics.jsonl').read_text().splitlines():
                event=json.loads(line)
                if event.get('event')=='baseline':initial_baseline=event['metrics'];break
    elif args.initial_checkpoint:
        previous=json.loads((directory/'pretraining-results.json').read_text())
        if previous['status']!='pretraining_finished':raise ValueError('previous stage must have completed')
        initial=torch.load(args.initial_checkpoint,map_location='cpu',weights_only=False)
        if initial.get('kind')!='actor' or initial.get('fingerprint')!=config.fingerprint or initial.get('run_id')!=run_id:
            raise ValueError('curriculum initial actor identity mismatch')
        model.load_state_dict(initial['model_state_dict']);step=int(initial['step'])
        with (directory/'pretraining-phase-history.jsonl').open('a') as out:out.write(json.dumps(previous)+'\n')
    else:
        if args.restart_random and checkpoint.exists():
            previous=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if previous.get('run_id')!=run_id:raise ValueError('random restart run identity mismatch')
            entry=dict(event='explicit_random_restart',previous_step=previous['step'],previous_identity=previous.get('training_identity'))
            with (directory/'pretraining-phase-history.jsonl').open('a') as out:out.write(json.dumps(entry)+'\n')
        atomic_torch_save(dict(model_state_dict=model.state_dict(),seed=args.seed,step=0),directory/'random-initialization.pt')
    start_step=saved.get('stage_initial_step',step) if args.resume else step
    if isinstance(valid,CombinedRecords):
        ids=np.concatenate([evaluation_ids(s,args.eval_per_category)+offset
                            for s,offset in zip(valid.sources,valid.offsets[:-1])])
    else:ids=evaluation_ids(valid,args.eval_per_category)
    started=time.time();live=None
    if not args.no_monitor:
        live=LiveTraining(port=args.port,interval=1)
        live.update(phase='training',workers=0,actor_processes=0,device=device,precision='bf16',
                    positions=len(train),completed_games=0,step=step,actor_step=step,config=config.to_dict())
    (directory/'pretraining-config.json').write_text(json.dumps(dict(vars(args),run_id=run_id,dataset=manifest),indent=2))
    (directory/'pretraining-results.json').write_text(json.dumps(dict(status='running',run_id=run_id,phase=args.phase,self_play_started=False)))
    log=(directory/'pretraining-metrics.jsonl').open('a',buffering=1)
    current_evaluation=evaluate(model,valid,ids,device,args.batch_size)
    baseline=initial_baseline if initial_baseline is not None else current_evaluation
    evaluation_event=dict(event='resume_evaluation' if args.resume else 'baseline',step=step,metrics=current_evaluation,curriculum=args.phase)
    log.write(json.dumps(evaluation_event)+'\n');print(json.dumps(evaluation_event),flush=True)
    if args.initial_checkpoint and not args.resume:
        best=baseline['selection_score'];export_actor(model,config,run_id,step,directory/'best.pt')
    def save(epoch,next_batch):
        atomic_torch_save(dict(kind='tactics_pretraining',optimizer_type='AdamW',
            fingerprint=config.fingerprint,config=config.to_dict(),run_id=run_id,dataset_seed=manifest['seed'],
            dataset_id=dataset_id,training_identity=training_identity,
            step=step,epoch=epoch,next_batch=next_batch,best_score=best,best_epoch=best_epoch,history=history,
            baseline=baseline,stage_initial_step=start_step,
            model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
            torch_rng_state=torch.get_rng_state()),checkpoint)
    def prepare(batch_ids,epoch,index):
        rows=train[batch_ids]
        values=decode(rows,mirror=True,rng=np.random.default_rng(args.seed+epoch*2000003+index))
        return values,rows['category'].copy()
    if not args.resume:save(0,0)
    try:
        for epoch in range(first_epoch,args.epochs):
            epoch_start=time.time()
            rng=np.random.default_rng(args.seed+epoch)
            if args.max_train_rows and isinstance(train,CombinedRecords):
                each=args.max_train_rows//len(train.sources)
                order=np.concatenate([rng.choice(len(s),min(len(s),each),replace=False)+offset
                                      for s,offset in zip(train.sources,train.offsets[:-1])])
                if epoch<args.warmup_epochs:order=order[train['category'][order]!=1]
            else:
                order=np.flatnonzero(train['category']!=1) if epoch<args.warmup_epochs else np.arange(len(train))
                if args.max_train_rows and len(order)>args.max_train_rows:order=rng.choice(order,args.max_train_rows,replace=False)
            rng.shuffle(order)
            batches=math.ceil(len(order)/args.batch_size)
            offset=first_batch if epoch==first_epoch else 0
            losses=[];model.train()
            with ThreadPoolExecutor(max_workers=2) as pool:
                pending=[];next_index=offset
                def submit():
                    nonlocal next_index
                    if next_index<batches:
                        i=next_index;next_index+=1
                        pending.append((i,pool.submit(prepare,order[i*args.batch_size:(i+1)*args.batch_size],epoch,i)))
                submit();submit()
                while pending:
                    index,future=pending.pop(0);values,cats=future.result();submit()
                    x,pi,z,pw,vw=values
                    # Positive mate roots outnumber forced-loss roots four to one.
                    vw[z>0]*=(.5 if epoch<args.warmup_epochs else .25)
                    # Phase out unconditional stylistic preferences as tactical solutions improve.
                    fade=max(.2,1-epoch/max(args.epochs-1,1));pw[cats>=4]*=fade
                    progress=(epoch+(index+1)/max(batches,1))/args.epochs
                    lr=args.learning_rate*(.1+.9*.5*(1+math.cos(math.pi*progress)))
                    for group in optimizer.param_groups:group['lr']=lr
                    tensors=[torch.from_numpy(a).pin_memory().to(device,non_blocking=True) if device=='cuda'
                             else torch.from_numpy(a) for a in (x,pi,z,pw,vw)]
                    xt,pit,zt,pwt,vwt=tensors;xt=xt.contiguous(memory_format=torch.channels_last)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device=='cuda'):
                        logits,v=model(xt)
                    policy=(-(pit*F.log_softmax(logits.float(),dim=1)).sum(1)*pwt).sum()/pwt.sum().clamp_min(1)
                    value=((v.float()-zt).square()*vwt).sum()/vwt.sum().clamp_min(1)
                    loss=policy+value
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite pretraining loss')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step();step+=1;losses.append(float(loss.detach()))
                    if step%25==0:
                        stats=dict(event='update',curriculum=args.phase,epoch=epoch+1,batch=index+1,batches=batches,step=step,
                                   loss=losses[-1],policy_loss=float(policy.detach()),value_loss=float(value.detach()),
                                   lr=lr,elapsed_seconds=time.time()-started,gradient_norm=float(norm))
                        log.write(json.dumps(stats)+'\n')
                        (directory/'pretraining-progress.json').write_text(json.dumps(stats))
                        if live:live.update(**{k:v for k,v in stats.items() if k not in ('event',)},actor_step=step)
                    if step%256==0:save(epoch,index+1)
            metrics=evaluate(model,valid,ids,device,args.batch_size)
            result=dict(event='epoch',curriculum=args.phase,epoch=epoch+1,step=step,train_loss=float(np.mean(losses)) if losses else None,
                        train_loss_batches=len(losses),epoch_batches=batches,resumed_at_batch=offset,
                        seconds=time.time()-epoch_start,metrics=metrics)
            if device=='cuda':
                result['peak_allocated_gib']=torch.cuda.max_memory_allocated()/1024**3
            history.append(result);log.write(json.dumps(result)+'\n');print(json.dumps(result),flush=True)
            if metrics['selection_score']>best:
                best=metrics['selection_score'];best_epoch=epoch+1
                export_actor(model,config,run_id,step,directory/'best.pt')
            save(epoch+1,0)
        selected=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(selected['model_state_dict'])
        final={}
        for split in ('validation','test','ood'):
            path=Path(args.data_root)/((args.dataset+'-' if args.dataset else '')+split+'.bin')
            if path.exists():
                rows=records(path);sample=evaluation_ids(rows,4096,seed=1731)
                final[split]=evaluate(model,rows,sample,device,args.batch_size)
            if args.additional_data:
                path=Path(args.data_root)/(args.additional_data+'-'+split+'.bin')
                if path.exists():
                    rows=records(path);sample=evaluation_ids(rows,4096,seed=1731)
                    final[args.additional_data+'_'+split]=evaluate(model,rows,sample,device,args.batch_size)
        export_actor(model,config,run_id,int(selected['step']),directory/'latest.pt')
        # An explicit fresh SGD learner makes the trained weights ready for a later authorized self-play phase.
        learner=Trainer(config,'cpu');learner.model.load_state_dict(selected['model_state_dict'])
        learner.run_id=run_id;learner.optimizer_step_offset=int(selected['step'])
        replay=ReplayBuffer(directory/'replay.sqlite',config.replay_window_games,window_positions=config.replay_window_positions)
        extra=dict(replay.recovery_marker(),pretraining=dict(source='verified-tactics',updates=int(selected['step']),best_epoch=best_epoch,
                                                           phase=args.phase,dataset_id=dataset_id))
        learner.save_checkpoint(directory/'learner.pt',extra);replay.close()
        result=dict(status='pretraining_finished',run_id=run_id,best_epoch=best_epoch,best_update=int(selected['step']),
                    completed_updates=step,epochs=args.epochs,elapsed_seconds=time.time()-started,baseline=baseline,
                    history=history,evaluation=final,self_play_started=False,parameters=model.count_parameters(),
                    dataset_id=dataset_id,phase=args.phase,stage_updates=step-start_step)
        (directory/'pretraining-results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    finally:
        log.close()
        if live:live.close()


if __name__=='__main__':main()
