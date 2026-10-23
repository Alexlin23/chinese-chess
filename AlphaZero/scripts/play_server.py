"""Local browser play against a trained model, independent of training resources."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time
import uuid
import torch
from AlphaZero.engine import GameState, Move
from AlphaZero.model.player import ModelPlayer
from AlphaZero.scripts.paths import DEFAULT_CONFIG, run_paths


class PlaySession:
    def __init__(self, player, reload_player=None):
        self.player = player
        self.lock = threading.RLock()
        self.model_lock = threading.Lock()
        self.reload_player = reload_player
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.token = secrets.token_urlsafe(32)
        self.reset(1)

    def reset(self, human):
        if type(human) is not int or human not in (-1, 1):
            raise ValueError("请选择红方或黑方")
        with self.lock:
            if self.reload_player is not None:
                with self.model_lock:
                    self.player = self.reload_player(self.player)
            self.game_id = uuid.uuid4().hex
            self.state = GameState.new_game(self.player.config.max_game_ply)
            self.human, self.revision = human, 0
            self.last_move = None
            self.error = None
            self.thinking = False
            self._schedule()
            return self.snapshot()

    def _schedule(self):
        if self.state.is_terminal() or self.state.current_player == self.human:
            return
        self.thinking = True
        self.pool.submit(self._reply, self.game_id, self.revision, self.state, self.player)

    def _reply(self, game_id, revision, state, player):
        try:
            with self.model_lock:
                move, diagnostics = player.choose(state)
            if move is None or not state.is_legal(move):
                raise RuntimeError("模型未返回合法走法")
            with self.lock:
                if (self.game_id, self.revision) != (game_id, revision):
                    return
                self.state = state.apply(move)
                self.last_move = [move.from_row, move.from_col, move.to_row, move.to_col]
                self.revision += 1
                self.thinking = False
        except Exception as exc:
            with self.lock:
                if (self.game_id, self.revision) == (game_id, revision):
                    self.thinking = False
                    self.error = str(exc)

    def move(self, game_id, revision, coordinates):
        with self.lock:
            if (game_id, revision) != (self.game_id, self.revision):
                raise ValueError("棋局已更新，请重新落子")
            if self.thinking or self.error or self.state.is_terminal() or self.state.current_player != self.human:
                raise ValueError("现在不能落子")
            if not isinstance(coordinates, list) or len(coordinates) != 4 or any(type(v) is not int for v in coordinates):
                raise ValueError("走法格式错误")
            move = Move(*coordinates)
            if not self.state.is_legal(move):
                raise ValueError("这步棋不合法")
            self.state = self.state.apply(move)
            self.last_move = coordinates
            self.revision += 1
            self._schedule()
            return self.snapshot()

    def snapshot(self):
        with self.lock:
            result = self.state.game_result()
            legal = [] if result else [[m.from_row, m.from_col, m.to_row, m.to_col]
                                       for m in self.state.legal_moves()]
            return dict(game_id=self.game_id, revision=self.revision, board=self.state.board.tolist(),
                        turn=self.state.current_player, human=self.human, ply=self.state.move_count,
                        thinking=self.thinking, error=self.error, last_move=self.last_move,
                        legal_moves=legal, token=self.token, model_step=self.player.step,
                        simulations=self.player.simulations,
                        exploration_noise=getattr(self.player, "exploration_noise", False),
                        search_noise_depth=getattr(self.player, "search_noise_depth", 0),
                        natural_limit=self.state.natural_limit_status(),
                        result=None if result is None else dict(winner=result.winner, reason=result.reason))

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


def create_server(session, host="127.0.0.1", port=8767):
    root = Path(__file__).resolve().parents[2] / "frontend"
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def respond(self, payload, status=200):
            data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        def do_GET(self):
            if self.path == "/api/state":
                return self.respond(session.snapshot())
            name = {"/": "play.html", "/play.js": "play.js", "/training.css": "training.css"}.get(self.path)
            if not name:
                return self.send_error(404)
            data = (root / name).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", {"html":"text/html", "js":"text/javascript", "css":"text/css"}[name.rsplit(".",1)[1]] + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("请求过长或为空")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict) or not secrets.compare_digest(str(body.get("token", "")), session.token):
                    return self.respond(dict(error="请从本地对弈页面操作"), 403)
                if self.path == "/api/new":
                    return self.respond(session.reset(body.get("human", 1)))
                if self.path == "/api/move":
                    return self.respond(session.move(body.get("game_id"), body.get("revision"), body.get("move")))
                self.send_error(404)
            except (ValueError, TypeError, KeyError) as exc:
                self.respond(dict(error=str(exc)), 400)
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir")
    parser.add_argument("--checkpoint", help="explicit snapshot; otherwise use the configured run's latest.pt")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--simulations", type=int, help="default: num_simulations from the selected config")
    parser.add_argument("--exploration-noise", action=argparse.BooleanOptionalAction, default=False,
                        help="mix Dirichlet exploration noise into human-play MCTS")
    parser.add_argument("--search-noise-depth", type=int,
                        help="noisy search levels; default: selected config search_noise_depth")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--cuda-memory-fraction", type=float, default=.05)
    parser.add_argument("--wait-for-trained", action="store_true")
    parser.add_argument("--fixed-snapshot", action="store_true")
    args = parser.parse_args()
    config, directory = run_paths(args.config, args.run_dir)
    checkpoint = Path(args.checkpoint).expanduser() if args.checkpoint else directory / "latest.pt"
    simulations = args.simulations if args.simulations is not None else config.num_simulations
    player_options = dict(exploration_noise=args.exploration_noise,
                          search_noise_depth=(config.search_noise_depth if args.search_noise_depth is None
                                              else args.search_noise_depth))
    torch.set_num_threads(1)
    if args.device == "cuda":
        if not 0 < args.cuda_memory_fraction <= 1:
            raise ValueError("invalid CUDA memory fraction")
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
    while True:
        try:
            player = ModelPlayer(checkpoint, args.device, simulations, **player_options)
            break
        except (FileNotFoundError, ValueError) as exc:
            if not args.wait_for_trained or (isinstance(exc, ValueError) and "随机初始化" not in str(exc)):
                raise
            print("等待首个已训练模型发布…", flush=True)
            time.sleep(30)
    signature = player.path.stat().st_mtime_ns
    def reload_player(current):
        nonlocal signature
        next_signature = current.path.stat().st_mtime_ns
        if signature == next_signature:
            return current
        candidate = ModelPlayer(current.path, args.device, simulations, **player_options)
        differences = {key for key in candidate.config.spec.keys() | current.config.spec.keys()
                       if candidate.config.spec.get(key) != current.config.spec.get(key)}
        if differences - {"max_game_ply"} or candidate.step < current.step:
            raise ValueError("新模型的规格不一致或版本倒退")
        signature = next_signature
        return candidate
    session = PlaySession(player, None if args.fixed_snapshot else reload_player)
    server = create_server(session, port=args.port)
    print(f"对弈地址：http://localhost:{args.port} · 模型更新步数 {session.player.step}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        session.close()


if __name__ == "__main__":
    main()
