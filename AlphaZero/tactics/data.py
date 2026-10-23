"""Compact disk records, vectorized 131-plane decoding, and independent Python audit."""
import numpy as np
from AlphaZero.engine import GameState, Move, ActionEncoder
from AlphaZero.engine.constants import POLICY_SIZE
from AlphaZero.train.replay import _MIRROR_ACTION_IDS

CATEGORIES = ('mate1', 'mate3', 'forced_loss', 'defence', 'capture', 'check')
DTYPE = np.dtype([
    ('history', 'i1', (8,10,9)), ('repetitions','u1',(8,)),
    ('turn','i1'), ('history_length','u1'), ('category','u1'), ('family','u1'),
    ('ply','<u2'), ('no_capture','<u2'), ('count','<u2'), ('legal_count','<u2'),
    ('actions','<u2',(128,)), ('legal_actions','<u2',(128,)), ('probabilities','<f4',(128,)),
    ('value','<f4'), ('policy_weight','<f4'), ('value_weight','<f4'), ('key','<u8')])
assert DTYPE.itemsize == 1784


def records(path):
    return np.memmap(path, dtype=DTYPE, mode='r')


class CombinedRecords:
    """Index several compact datasets without copying their board histories into RAM."""
    def __init__(self,sources):
        self.sources=sources
        self.offsets=np.cumsum([0]+[len(s) for s in sources])
    def __len__(self):return int(self.offsets[-1])
    def __getitem__(self,index):
        if isinstance(index,str):return np.concatenate([s[index] for s in self.sources])
        if isinstance(index,slice):index=np.arange(len(self))[index]
        scalar=np.isscalar(index);ids=np.atleast_1d(index).astype(np.int64)
        if np.any(ids<0) or np.any(ids>=len(self)):raise IndexError('tactics record index')
        out=np.empty(len(ids),dtype=DTYPE)
        for s,begin,end in zip(self.sources,self.offsets[:-1],self.offsets[1:]):
            mask=(ids>=begin)&(ids<end)
            out[mask]=s[ids[mask]-begin]
        return out[0] if scalar else out


def decode(batch, mirror=False, rng=None):
    n=len(batch)
    board=batch['history'].copy()
    black=batch['turn']<0
    board[black]=-board[black, :, ::-1, ::-1]
    reflected=(rng.random(n)<.5) if mirror else np.zeros(n,dtype=bool)
    board[reflected]=board[reflected,:,:,::-1]
    x=np.zeros((n,8,16,10,9),dtype=np.float32)
    for p in range(1,8):
        x[:,:,p-1]=board==p
        x[:,:,p+6]=board==-p
    x[:,:,14]=(batch['repetitions']>=2)[:,:,None,None]
    x[:,:,15]=(batch['repetitions']>=3)[:,:,None,None]
    output=np.empty((n,131,10,9),dtype=np.float32)
    output[:,:128]=x.reshape(n,128,10,9)
    output[:,128]=(batch['turn']>0)[:,None,None]
    output[:,129]=(batch['ply']/180)[:,None,None]
    output[:,130]=(batch['no_capture']/120)[:,None,None]
    policy=np.zeros((n,POLICY_SIZE),np.float32)
    for i,row in enumerate(batch):
        size=int(row['count']);a=row['actions'][:size].astype(int)
        if reflected[i]:a=_MIRROR_ACTION_IDS[a]
        policy[i,a]=row['probabilities'][:size]
    return output,policy,batch['value'].copy(),batch['policy_weight'].copy(),batch['value_weight'].copy()


def state_from_record(row):
    h=int(row['history_length']);side=int(row['turn']);ply=int(row['ply'])
    state=GameState(row['history'][h-1],side*(-1 if (h-1)%2 else 1),move_count=ply-h+1,max_ply=180)
    for i in range(h-2,-1,-1):
        nxt=row['history'][i]
        changed=np.argwhere(state.board!=nxt)
        if len(changed)!=2:raise ValueError('history is not a single legal move')
        origins=[(int(r),int(c)) for r,c in changed if state.board[r,c]*state.current_player>0 and nxt[r,c]==0]
        if len(origins)!=1:raise ValueError('ambiguous history origin')
        fr,fc=origins[0];tr,tc=next((int(r),int(c)) for r,c in changed if (int(r),int(c))!=(fr,fc))
        state=state.apply(Move(fr,fc,tr,tc))
        if not np.array_equal(state.board,nxt):raise ValueError('history board mismatch')
    if state.current_player!=side or state.no_capture_count!=int(row['no_capture']):
        raise ValueError('history rule counters mismatch')
    return state


def mate1_moves(state):
    side=state.current_player;answer=[]
    for m in state.legal_moves():
        result=state.apply(m).game_result()
        if result and result.winner==side:answer.append(m)
    return answer


def verify(row, prove=True):
    """Independent from C++ move legality AND mate proof, using the actual runtime rules."""
    state=state_from_record(row)
    if state.is_terminal():raise ValueError('terminal training input')
    n=int(row['count']);prob=row['probabilities'][:n]
    if not 0<n<=128 or not np.isfinite(prob).all() or np.any(prob<0) or not np.isclose(prob.sum(),1,atol=1e-5):
        raise ValueError('invalid policy distribution')
    actions=row['actions'][:n].astype(int)
    if len(np.unique(actions))!=n or np.any(actions>=POLICY_SIZE):raise ValueError('invalid actions')
    if not state.canonical_mask()[actions].all():raise ValueError('illegal target action')
    if not 0<int(row['legal_count'])<=128:raise ValueError('independent legal masks disagree: invalid count')
    legal=np.sort(row['legal_actions'][:int(row['legal_count'])])
    if not np.array_equal(legal,np.flatnonzero(state.canonical_mask())):raise ValueError('independent legal masks disagree')
    target=[ActionEncoder.decode(int(a) if state.turn else ActionEncoder.rotate_action(int(a))) for a in actions]
    kind=CATEGORIES[int(row['category'])]
    if kind=='mate1' and any(state.apply(m).game_result() is None or state.apply(m).game_result().winner!=state.current_player for m in target):
        raise ValueError('false mate1')
    if kind=='mate3' and prove:
        if mate1_moves(state):raise ValueError('mate3 is already mate1')
        for m in target:
            after=state.apply(m)
            if after.is_terminal() or not all(mate1_moves(after.apply(reply)) for reply in after.legal_moves()):
                raise ValueError('false forced mate3')
    if kind=='forced_loss' and prove:
        if not all(mate1_moves(state.apply(m)) for m in state.legal_moves()):raise ValueError('false forced loss')
    if kind=='defence' and prove:
        if any(mate1_moves(state.apply(m)) for m in target):raise ValueError('unsafe defensive label')
    if kind=='capture' and any(abs(state.board[m.to_row,m.to_col]) not in (4,5,6) for m in target):
        raise ValueError('false capture label')
    if kind=='check' and any(not state.apply(m).is_in_check() for m in target):raise ValueError('false checking label')
    expected=1 if kind in ('mate1','mate3') else -1 if kind=='forced_loss' else 0
    if (not np.isfinite([row['value'],row['value_weight'],row['policy_weight']]).all()
            or float(row['value'])!=expected or float(row['value_weight'])!=(1 if expected else 0)
            or float(row['policy_weight'])<0):raise ValueError('false value label or weight')
    return state
