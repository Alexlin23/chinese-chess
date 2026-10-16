"""Web API using the same history-aware GameState as self-play and MCTS."""
import os
import sys
from pathlib import Path
import threading
from contextlib import asynccontextmanager
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "backend"

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from .models import (ValidMovesRequest, MoveRequest, MoveResponse, GameCreateResponse,
                     GameInfoResponse, PieceInfo, CheckWinRequest, UndoRequest)
from . import database as db
from .rule_adapter import numeric_board, web_board, status
from AlphaZero.engine import Move, player_id
from AlphaZero.engine.heuristic import heuristic_move
from AlphaZero.train.config import RULE_VERSION

_neural_mcts = None
_neural_lock = threading.Lock()


def _get_neural_mcts():
    global _neural_mcts
    if _neural_mcts is None:
        from AlphaZero.model.player import ModelPlayer
        root = Path(__file__).resolve().parents[1]
        best, latest = root / "AlphaZero/checkpoints/best.pt", root / "AlphaZero/checkpoints/latest.pt"
        path = Path(os.environ["XIANGQI_MODEL"]) if os.environ.get("XIANGQI_MODEL") else (best if best.exists() else latest)
        if not path.exists():
            return None
        # ModelPlayer rejects incompatible/step-zero checkpoints before searching.
        from AlphaZero.search import MCTS
        player = ModelPlayer(path, device="cpu")
        _neural_mcts = MCTS(player.evaluator, num_simulations=player.config.num_simulations,
                           c_puct=player.config.c_puct, pb_c_base=player.config.pb_c_base)
    return _neural_mcts


@asynccontextmanager
async def lifespan(app):
    db.init_db()
    yield


app = FastAPI(title="中国象棋 API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
FRONTEND_DIR = str(Path(__file__).resolve().parents[1] / "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.exception_handler(db.StateConflict)
async def state_conflict(request, exc):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


def _game(game_id):
    try:
        return db.get_rule_game(game_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="对局不存在") from exc


def _request_state(req):
    state, revision = _game(req.game_id)
    try:
        board = numeric_board(req.board)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if (req.revision != revision or player_id(req.turn) != state.current_player
            or not np.array_equal(board, state.board)):
        raise db.StateConflict("棋局已更新或提交棋盘不一致，请刷新后重试")
    return state, revision


def _metadata(state, revision):
    result = state.game_result()
    return dict(revision=revision, no_capture_count=state.no_capture_count,
                natural_limit=state.natural_limit_status(),
                rule_reason=result.reason if result else None, rule_version=RULE_VERSION)


@app.get("/")
def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.post("/api/valid-moves")
def api_valid_moves(req: ValidMovesRequest):
    state, revision = _request_state(req)
    moves = [] if state.is_terminal() else [
        dict(row=m.to_row, col=m.to_col, capture=bool(state.board[m.to_row, m.to_col]))
        for m in state.legal_moves() if (m.from_row, m.from_col) == (req.row, req.col)]
    return dict(game_id=req.game_id, moves=moves, **_metadata(state, revision))


@app.post("/api/move", response_model=MoveResponse)
def api_move(req: MoveRequest):
    _request_state(req)
    result = db.commit_move(req.game_id,
        Move(req.from_pos.row, req.from_pos.col, req.to_pos.row, req.to_pos.col), req.revision)
    if result is None:
        return MoveResponse(valid=False)
    state, revision, captured = result
    outcome = status(state)
    return MoveResponse(valid=True, captured=PieceInfo(**captured) if captured else None,
        check=state.is_in_check(), game_over=outcome if outcome != "ongoing" else None,
        new_board=web_board(state), turn="r" if state.turn else "b", **_metadata(state, revision))


@app.post("/api/check-win")
def api_check_win(req: CheckWinRequest):
    state, revision = _request_state(req)
    return dict(result=status(state), **_metadata(state, revision))


@app.post("/api/game/new", response_model=GameCreateResponse)
def api_new_game():
    game_id = db.create_game()
    state, revision = _game(game_id)
    return GameCreateResponse(game_id=game_id, board=web_board(state), turn="r",
                              **_metadata(state, revision))


@app.get("/api/game/{game_id}", response_model=GameInfoResponse)
def api_get_game(game_id: int):
    state, revision = _game(game_id)
    return GameInfoResponse(game_id=game_id, board=web_board(state), turn="r" if state.turn else "b",
        status=status(state), step_count=state.move_count, **_metadata(state, revision))


@app.get("/api/game/{game_id}/history")
def api_game_history(game_id: int):
    # Return history and revision from one read transaction.
    _game(game_id)
    with db.transaction() as conn:
        _, row, moves = db._load(conn, game_id)
        return dict(game_id=game_id, revision=row["revision"], moves=[dict(m) for m in moves])


@app.post("/api/game/{game_id}/undo")
def api_undo(game_id: int, req: UndoRequest):
    _game(game_id)
    result = db.undo_game(game_id, req.revision)
    if result is None:
        raise HTTPException(status_code=400, detail="没有可撤销的走棋")
    result["rule_version"] = RULE_VERSION
    return result


def _require_playable(state):
    if state.is_terminal():
        raise HTTPException(status_code=400, detail="对局已经结束")


@app.post("/api/ai-move")
def api_ai_move(req: MoveRequest):
    state, revision = _request_state(req)
    _require_playable(state)
    with _neural_lock:
        search = _get_neural_mcts()
        if search is None:
            raise HTTPException(status_code=503, detail="模型未加载，请先配置训练后的模型")
        move, _ = search.select_move(state, temperature=0.0, add_exploration_noise=False)
    # A concurrent move or undo invalidates an AI response too.
    if _game(req.game_id)[1] != revision:
        raise db.StateConflict("AI思考期间棋局已更新")
    if move is None or not state.is_legal(move):
        raise HTTPException(status_code=400, detail="AI找不到合法走法")
    return dict(game_id=req.game_id, from_pos=dict(row=move.from_row, col=move.from_col),
                to_pos=dict(row=move.to_row, col=move.to_col), **_metadata(state, revision))


@app.post("/api/heuristic-move")
def api_heuristic_move(req: MoveRequest):
    state, revision = _request_state(req)
    _require_playable(state)
    move = heuristic_move(state)
    if _game(req.game_id)[1] != revision:
        raise db.StateConflict("AI思考期间棋局已更新")
    if move is None:
        raise HTTPException(status_code=400, detail="启发式AI找不到合法走法")
    return dict(game_id=req.game_id, from_pos=dict(row=move.from_row, col=move.from_col),
                to_pos=dict(row=move.to_row, col=move.to_col), **_metadata(state, revision))


@app.websocket("/ws/watch/{game_id}")
async def ws_watch_game(ws: WebSocket, game_id: int):
    await ws.accept()
    try:
        while True:
            if await ws.receive_text() == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
