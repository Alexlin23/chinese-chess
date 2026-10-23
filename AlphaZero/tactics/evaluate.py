"""Held-out policy accuracy and balanced value checks, with runtime legal masking."""
import numpy as np
import torch
from AlphaZero.tactics.data import CATEGORIES,decode


def evaluation_ids(rows,per_category=512,seed=711):
    rng=np.random.default_rng(seed);ids=[]
    for cat in range(6):
        available=np.flatnonzero(rows['category']==cat)
        ids.extend(rng.choice(available,min(len(available),per_category),replace=False).tolist())
    return np.array(ids,dtype=np.int64)


@torch.inference_mode()
def evaluate(model,rows,ids,device='cuda',batch_size=256):
    model.eval();hits=np.zeros(6,np.int64);counts=np.zeros(6,np.int64)
    raw_hits=np.zeros(6,np.int64);value_errors={1:[], -1:[]};value_sign={1:[], -1:[]}
    for start in range(0,len(ids),batch_size):
        batch=rows[ids[start:start+batch_size]]
        x,pi,z,_,vw=decode(batch)
        tensor=torch.from_numpy(x).to(device).contiguous(memory_format=torch.channels_last)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=str(device).startswith('cuda')):
            logits,values=model(tensor)
        logits=logits.float().cpu().numpy();values=values.float().cpu().numpy()
        if not np.isfinite(logits).all() or not np.isfinite(values).all():raise FloatingPointError('nonfinite evaluation')
        for i,row in enumerate(batch):
            cat=int(row['category']);counts[cat]+=1
            targets=set(row['actions'][:int(row['count'])].tolist())
            raw_hits[cat]+=int(logits[i].argmax() in targets)
            legal=row['legal_actions'][:int(row['legal_count'])].astype(int)
            selected=int(legal[np.argmax(logits[i,legal])]);hits[cat]+=int(selected in targets)
            if vw[i]:
                key=int(z[i]);value_errors[key].append(float((values[i]-z[i])**2))
                value_sign[key].append(float(values[i]*z[i]>0))
    result={name:dict(n=int(counts[i]),policy_top1=float(hits[i]/max(counts[i],1)),
                     unmasked_top1=float(raw_hits[i]/max(counts[i],1))) for i,name in enumerate(CATEGORIES)}
    for key in (1,-1):
        result['value_win' if key==1 else 'value_loss']=dict(n=len(value_errors[key]),
            mse=float(np.mean(value_errors[key])) if value_errors[key] else None,
            sign_accuracy=float(np.mean(value_sign[key])) if value_sign[key] else None)
    result['selection_score']=(.4*result['mate1']['policy_top1']+.3*result['mate3']['policy_top1']+
        .2*result['defence']['policy_top1']+.05*(result['value_win']['sign_accuracy'] or 0)+
        .05*(result['value_loss']['sign_accuracy'] or 0))
    return result


def main():
    import argparse
    import json
    from pathlib import Path
    from AlphaZero.train.config import AlphaZeroConfig
    from AlphaZero.train.trainer import create_model
    from AlphaZero.tactics.data import records
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',default='/mnt/d/project/Alphazero-data/policy-tactics/run')
    p.add_argument('--data-root',default='/mnt/d/project/Alphazero-data/policy-tactics/data')
    p.add_argument('--dataset',default='curriculum')
    p.add_argument('--checkpoint',default='latest.pt')
    p.add_argument('--random-initialization',action='store_true')
    p.add_argument('--per-category',type=int,default=4096)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=p.parse_args();root=Path(args.run_dir);torch.set_num_threads(1)
    path=root/('random-initialization.pt' if args.random_initialization else args.checkpoint)
    payload=torch.load(path,map_location='cpu',weights_only=False)
    specification=(torch.load(root/'pretraining.pt',map_location='cpu',weights_only=False)
                   if args.random_initialization else payload)
    config=AlphaZeroConfig(**specification['config'])
    if not args.random_initialization and payload.get('fingerprint')!=config.fingerprint:
        raise ValueError('evaluation specification mismatch')
    model=create_model(config,args.device);model.load_state_dict(payload['model_state_dict'])
    result={}
    for split in ('validation','test','ood'):
        rows=records(Path(args.data_root)/((args.dataset+'-' if args.dataset else '')+split+'.bin'))
        result[split]=evaluate(model,rows,evaluation_ids(rows,args.per_category,seed=1731),args.device)
    report=dict(checkpoint=str(path),random_initialization=args.random_initialization,metrics=result)
    (root/('random-evaluation.json' if args.random_initialization else 'actor-evaluation.json')).write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
