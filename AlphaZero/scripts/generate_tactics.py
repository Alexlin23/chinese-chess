"""Compile the independent offline prover, generate shards, deduplicate and split."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from AlphaZero.tactics.data import DTYPE,CATEGORIES,records,verify,decode
from AlphaZero.train.config import ACTION_VERSION,RULE_VERSION


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='/mnt/d/project/Alphazero-data/policy-tactics')
    p.add_argument('--count',type=int,default=12000)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=20261023)
    p.add_argument('--audit-per-category',type=int,default=8)
    p.add_argument('--ood-count',type=int,default=0)
    p.add_argument('--name',default='',help='optional dataset prefix in the same data directory')
    p.add_argument('--dense',action='store_true',help='add legal material and re-prove tactical labels')
    p.add_argument('--mixed-density',action='store_true',help='equal sparse/dense quota within every target class')
    args=p.parse_args()
    if args.count<6 or args.workers<1 or args.count//args.workers<6 or args.audit_per_category<1 or args.ood_count<0:
        raise ValueError('positive worker/audit counts and at least six records per worker required')
    root=Path(args.root);data=root/'data';data.mkdir(parents=True,exist_ok=True)
    if args.name and (not args.name.replace('-','').isalnum()):raise ValueError('simple dataset name required')
    prefix=args.name+'-' if args.name else ''
    if args.dense and args.mixed_density:raise ValueError('choose dense OR mixed-density')
    density=2 if args.mixed_density else int(args.dense)
    # Refuse an occupied output before starting any generator, including OOD.
    if any((data/(prefix+s+'.bin')).exists() for s in ('train','validation','test','ood')) or (data/(prefix+'manifest.json')).exists():
        raise FileExistsError('dataset output already exists; choose an empty generation root')
    source=Path(__file__).resolve().parents[1]/'tactics/generator.cpp'
    binary=root/'bin/generator';binary.parent.mkdir(exist_ok=True)
    subprocess.run(['g++','-O3','-std=c++17',str(source),'-o',str(binary)],check=True)
    started=time.time();jobs=[]
    for i in range(args.workers):
        shard=data/f'{prefix}shard-{args.seed}-{i}.bin'
        if shard.exists():raise FileExistsError(shard)
        n=args.count//args.workers+(i<args.count%args.workers)
        log=(shard.with_suffix('.log')).open('w')
        jobs.append((shard,subprocess.Popen([str(binary),str(shard),str(n),str(args.seed+i*9973),'0',str(density)],stdout=log,stderr=subprocess.STDOUT),log))
    if args.ood_count:
        shard=data/f'{prefix}shard-{args.seed}-ood.bin'
        if shard.exists():raise FileExistsError(shard)
        log=shard.with_suffix('.log').open('w')
        jobs.append((shard,subprocess.Popen([str(binary),str(shard),str(args.ood_count),str(args.seed+900001),'1',str(density)],stdout=log,stderr=subprocess.STDOUT),log))
    try:
        while any(proc.poll() is None for _,proc,_ in jobs):
            state=[]
            for shard,proc,_ in jobs:
                if proc.poll() not in (None,0):raise RuntimeError(f'generator failed: {shard}')
                f=Path(str(shard)+'.progress.json')
                try:state.append(json.loads(f.read_text()))
                except (FileNotFoundError,json.JSONDecodeError):pass
            (data/(prefix+'generation-progress.json')).write_text(json.dumps(dict(seconds=time.time()-started,workers=state)))
            time.sleep(2)
    finally:
        for _,proc,log in jobs:
            if proc.poll() is None:proc.terminate()
            proc.wait();log.close()
    for shard,proc,log in jobs:
        log.close()
        if proc.returncode:raise RuntimeError(f'generator failed: {shard} ({proc.returncode})')
    paths={s:data/f'{prefix}{s}.bin' for s in ('train','validation','test','ood') if s!='ood' or args.ood_count}
    if any(f.exists() for f in paths.values()):raise FileExistsError('dataset splits already exist')
    outputs={s:f.open('wb') for s,f in paths.items()};seen=set();counts={s:[0]*6 for s in paths}
    previous=set()
    for existing in data.glob('*manifest.json'):
        # Keep every already-generated held-out set isolated from later additions.
        earlier_prefix=existing.name[:-len('manifest.json')]
        for s in ('train','validation','test','ood'):
            old=data/(earlier_prefix+s+'.bin')
            if old.exists():previous.update(records(old)['key'].tolist())
    try:
        for shard,_,_ in jobs:
            for row in records(shard):
                key=int(row['key'])
                if key in seen or key in previous:continue
                seen.add(key);split=('ood' if int(row['family'])>=10 else
                                    'train' if key%100<80 else 'validation' if key%100<90 else 'test')
                outputs[split].write(row.tobytes());counts[split][int(row['category'])]+=1
    finally:
        for f in outputs.values():f.close()
    audited=[]
    history_quality={}
    rng=np.random.default_rng(args.seed)
    for split,path in paths.items():
        rows=records(path)
        table=np.array([np.bincount(rows['history_length'][rows['category']==cat],minlength=9)[1:9]
                        for cat in range(6)])
        labels=np.stack([table[0]+table[1],table[2]]).astype(float)
        expected=labels.sum(1)[:,None]*labels.sum(0)[None,:]/max(labels.sum(),1)
        cramer_v=float(np.sqrt((((labels-expected)**2/np.maximum(expected,1)).sum())/max(labels.sum(),1)))
        if labels.sum()>1000 and cramer_v>.1:raise ValueError('history length leaks value label')
        expected_class=table.sum(1)[:,None]*table.sum(0)[None,:]/max(table.sum(),1)
        class_v=float(np.sqrt((((table-expected_class)**2/np.maximum(expected_class,1)).sum())/max(table.sum(),1)/5))
        if table.sum()>1000 and class_v>.1:raise ValueError('history length leaks policy category')
        history_quality[split]=dict(by_category=table.tolist(),value_history_cramer_v=cramer_v,class_history_cramer_v=class_v)
        for cat in range(6):
            ids=np.flatnonzero(rows['category']==cat)
            sample=rng.choice(ids,min(len(ids),args.audit_per_category),replace=False)
            for i in sample:
                state=verify(rows[int(i)])
                actual=decode(rows[int(i):int(i)+1])[0][0]
                if not np.array_equal(actual,state.encode()):raise ValueError('runtime encoding differs')
            audited.append(dict(split=split,category=CATEGORIES[cat],checked=len(sample)))
    result=dict(version=3,teacher_version='history-balanced-v2',record_bytes=DTYPE.itemsize,action_version=ACTION_VERSION,rule_version=RULE_VERSION,
                max_game_ply=180,
                dense=args.dense,mixed_density=args.mixed_density,name=args.name,
                history_quality=history_quality,
                seed=args.seed,raw_count=args.count+args.ood_count,unique_count=len(seen),counts=counts,audit=audited,
                elapsed_seconds=time.time()-started,split='canonical board hash; mirrors and colour rotations share a split')
    (data/(prefix+'manifest.json')).write_text(json.dumps(result,indent=2))
    # The deduplicated splits are authoritative; raw shards are temporary.
    for shard,_,_ in jobs:shard.unlink()
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
