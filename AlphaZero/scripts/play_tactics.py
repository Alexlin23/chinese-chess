"""Play bounded real games against an independent two-ply material/terminal opponent.

This is a cold-start smoke match, not an Elo estimate or normal self-play training.
All moves and starting boards are retained for inspection. No replay is written.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from AlphaZero.engine import GameState,ActionEncoder
from AlphaZero.engine.heuristic import material_score
from AlphaZero.model.player import ModelPlayer
from AlphaZero.tactics.data import records,state_from_record,CATEGORIES


def coordinates(move):
    return [move.from_row,move.from_col,move.to_row,move.to_col]


def static_score(state,side):
    result=state.game_result()
    if result is not None:return 100000*result.to_value(side)
    return material_score(state.board,side>0)+(.15 if state.is_in_check() and state.current_player!=side else 0)


def opponent(state,rng):
    """Exact one-reply minimax, choosing immediate mate before material."""
    side=state.current_player;best=-float('inf');choices=[]
    for move in state.legal_moves():
        child=state.apply(move)
        if child.is_terminal():score=static_score(child,side)
        else:
            score=float('inf')
            for reply in child.legal_moves():
                score=min(score,static_score(child.apply(reply),side))
                if score<best:break
        if score>best+1e-6:best=score;choices=[move]
        elif abs(score-best)<1e-6:choices.append(move)
    return choices[int(rng.integers(len(choices)))]


def play(player,state,candidate_side,rng,name,initial_record=None):
    started=time.time();initial=state;trace=[]
    while not state.is_terminal():
        candidate=state.current_player==candidate_side
        if candidate:
            move,diagnostics=player.choose(state)
        else:move=opponent(state,rng);diagnostics=None
        if move is None or not state.is_legal(move):raise ValueError('illegal match move')
        entry=dict(ply=state.move_count+1,side=state.current_player,agent='model' if candidate else 'independent-opponent',
                   move=coordinates(move),captured=int(state.board[move.to_row,move.to_col]))
        state=state.apply(move);entry['check']=state.is_in_check()
        if diagnostics:entry['search']=diagnostics
        trace.append(entry)
    result=state.game_result()
    return dict(name=name,candidate_side=candidate_side,initial_board=initial.board.tolist(),
                initial_ply=initial.move_count,initial_turn=initial.current_player,
                initial_history=[dict(board=np.frombuffer(board,dtype=np.int8).reshape(10,9).tolist(),
                                      occurrences=int(count)) for board,count in initial.history],
                initial_no_capture=initial.no_capture_count,
                initial_record=initial_record,model_step=player.step,simulations=player.simulations,
                noise=False,winner=result.winner,reason=result.reason,plies=len(trace),
                outcome='draw' if result.winner is None else 'win' if result.winner==candidate_side else 'loss',
                seconds=time.time()-started,moves=trace,final_board=state.board.tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',default='/mnt/d/project/Alphazero-data/policy-tactics/run')
    parser.add_argument('--data-root',default='/mnt/d/project/Alphazero-data/policy-tactics/data')
    parser.add_argument('--checkpoint',default='latest.pt')
    parser.add_argument('--simulations',type=int,default=400)
    parser.add_argument('--probe-per-category',type=int,default=32)
    parser.add_argument('--skip-standard',action='store_true')
    parser.add_argument('--output',default='coldstart-games.json')
    parser.add_argument('--dataset',default='curriculum',help='prefix of a verified validation dataset')
    parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();directory=Path(args.run_dir)
    torch.set_num_threads(1);player=ModelPlayer(directory/args.checkpoint,'cuda',args.simulations,exploration_noise=False)
    rng=np.random.default_rng(1907);selection_rng=np.random.default_rng(31907)
    rows=records(Path(args.data_root)/((args.dataset+'-' if args.dataset else '')+'validation.bin'))
    games=[]
    for cat in (() if args.probe_only else (0,1,3)):
        ids=np.flatnonzero(rows['category']==cat)
        chosen=selection_rng.choice(ids,2,replace=False)
        for i in chosen:
            state=state_from_record(rows[i]);side=state.current_player
            game=play(player,state,side,rng,f'{CATEGORIES[cat]}-{len(games)+1}',int(i))
            games.append(game);print(json.dumps({k:v for k,v in game.items() if k not in ('moves','initial_history','initial_board','final_board')}),flush=True)
            (directory/args.output).write_text(json.dumps(dict(status='running',games=games),indent=2))
    if not args.skip_standard and not args.probe_only:
        for side in (1,-1):
            game=play(player,GameState.new_game(player.config.max_game_ply),side,rng,'standard-red' if side==1 else 'standard-black')
            games.append(game);print(json.dumps({k:v for k,v in game.items() if k not in ('moves','initial_history','initial_board','final_board')}),flush=True)
            (directory/args.output).write_text(json.dumps(dict(status='running',games=games),indent=2))
    probes={}
    for cat in (0,1,3):
        available=np.flatnonzero(rows['category']==cat)
        ids=np.random.default_rng(1907+cat).choice(available,min(len(available),args.probe_per_category),replace=False);hit=0
        for i in ids:
            row=rows[i];state=state_from_record(row);move,_=player.choose(state)
            action=ActionEncoder.encode(move)
            if state.current_player<0:action=ActionEncoder.rotate_action(action)
            hit+=int(action in row['actions'][:int(row['count'])])
        probes[CATEGORIES[cat]]=dict(n=len(ids),hits=hit,rate=hit/max(len(ids),1),indices=ids.tolist())
        print(json.dumps(dict(event='search_probe',category=CATEGORIES[cat],metrics=probes[CATEGORIES[cat]])),flush=True)
    result=dict(status='finished',checkpoint=args.checkpoint,model_step=player.step,simulations=args.simulations,
                opponent='independent exact two-ply terminal/material minimax; seeded equal-score choices',
                dataset=args.dataset,
                noise=False,training_replay_written=False,games=games,search_probes=probes,
                outcomes=dict(Counter(g['outcome'] for g in games)),
                limitations='Few selected games, not strength statistics. Synthetic validation endgames and initial openings differ.')
    (directory/args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(event='matches_finished',outcomes=result['outcomes'],search_probes=probes)),flush=True)


if __name__=='__main__':main()
