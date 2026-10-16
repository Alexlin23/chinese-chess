"""Explicit manual smoke against a separately started Web API (never run on import)."""
import argparse
import json
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='http://127.0.0.1:8001')
    parser.add_argument('--game-id', type=int)
    args = parser.parse_args()
    def call(path, body=None):
        req = Request(args.base + path,
            data=json.dumps(body).encode() if body is not None else None,
            method='POST' if body is not None else 'GET',
            headers={'Content-Type': 'application/json'})
        with urlopen(req) as response:
            return json.load(response)
    game_id = args.game_id or call('/api/game/new', {})['game_id']
    game = call(f'/api/game/{game_id}')
    context = {key: game[key] for key in ('board', 'turn', 'game_id', 'revision')}
    for r, row in enumerate(game['board']):
        for c, piece in enumerate(row):
            if piece and piece['type'] == '馬' and piece['color'] == game['turn']:
                moves = call('/api/valid-moves', dict(context, row=r, col=c))['moves']
                if moves:
                    target = moves[0]
                    response = call('/api/move', dict(context, from_pos={'row': r, 'col': c},
                                    to_pos={'row': target['row'], 'col': target['col']}))
                    print(json.dumps(response, ensure_ascii=False))
                    return
    raise SystemExit('没有可走的马，或对局已结束')


if __name__ == '__main__':
    main()
