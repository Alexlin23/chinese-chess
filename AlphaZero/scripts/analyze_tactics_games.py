"""Reconstruct saved cold-start games and audit immediate tactical opportunities."""
import argparse
import json
from pathlib import Path

from AlphaZero.engine import GameState,Move,ActionEncoder
from AlphaZero.tactics.data import records,state_from_record,mate1_moves


def analyze(report,rows,max_ply=180):
    findings=[]
    for game in report['games']:
        row=None if game['initial_record'] is None else rows[game['initial_record']]
        state=(GameState(game['initial_board'],game['initial_turn'],max_ply=max_ply)
               if row is None else state_from_record(row))
        missed=[];allowed=[];loss_audit=[];model_turns=0;first_correct=None
        for index,entry in enumerate(game['moves']):
            move=Move(*entry['move'])
            if entry['side']!=state.current_player or not state.is_legal(move):raise ValueError('broken match trace')
            if entry['agent']=='model':
                model_turns+=1;available=mate1_moves(state)
                if available and move not in available:missed.append(entry['ply'])
                if index==0 and row is not None:
                    action=ActionEncoder.encode(move)
                    if state.current_player<0:action=ActionEncoder.rotate_action(action)
                    first_correct=bool(action in row['actions'][:int(row['count'])])
                child=state.apply(move)
                if not child.is_terminal() and mate1_moves(child):
                    allowed.append(entry['ply']);safe=0
                    for alternative in state.legal_moves():
                        after=state.apply(alternative)
                        if after.is_terminal():
                            safe+=int(after.game_result().winner in (None,state.current_player))
                        elif not mate1_moves(after):safe+=1
                    loss_audit.append(dict(ply=entry['ply'],alternatives_avoiding_immediate_loss=safe))
            state=state.apply(move)
        result=state.game_result()
        if result is None or result.winner!=game['winner'] or result.reason!=game['reason']:
            raise ValueError('match outcome not reproduced')
        findings.append(dict(name=game['name'],outcome=game['outcome'],plies=game['plies'],
            first_target_hit=first_correct,model_turns=model_turns,immediate_win_not_taken=missed,
            moves_allowing_immediate_loss=allowed,immediate_loss_audit=loss_audit,result_independently_replayed=True))
    return dict(games=findings,limitations='Only exact immediate wins/losses are audited; not a complete optimal-move analysis.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',default='/mnt/d/project/Alphazero-data/policy-tactics/run')
    p.add_argument('--data-root',default='/mnt/d/project/Alphazero-data/policy-tactics/data')
    p.add_argument('--games',default='coldstart-games.json')
    args=p.parse_args();path=Path(args.run_dir)/args.games
    report=json.loads(path.read_text());prefix=report.get('dataset','')
    result=analyze(report,records(Path(args.data_root)/((prefix+'-' if prefix else '')+'validation.bin')))
    output=path.with_name(path.stem+'-analysis.json');output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
