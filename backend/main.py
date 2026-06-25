"""中国象棋后端 API - FastAPI 入口"""
import os
import sys
# 确保能导入同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from models import (
    ValidMovesRequest, MoveRequest, MoveResponse,
    GameCreateResponse, GameInfoResponse, PieceInfo, Position,
    CheckWinRequest,
)
from chess_rules import (
    create_initial_board, get_valid_moves, is_valid_move,
    make_move, check_game_result, is_in_check,
)
from database import init_db, create_game, get_game, update_game, add_move, get_moves, record_step, get_step_count, undo_game

# AlphaZero AI（只读导入）
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent))
from AlphaZero.engine import GameState as _AlphaGameState
from AlphaZero.search import MCTS as _AlphaMCTS
from AlphaZero.model import PolicyWDLEncoder as _PolicyWDLNet, NeuralEvaluator as _NeuralEval

# 神经网络评估器（懒加载）
_neural_model = None
_neural_evaluator = None
_neural_mcts = None


def _get_neural_mcts():
    """懒加载：创建/获取神经网络 MCTS 实例"""
    global _neural_model, _neural_evaluator, _neural_mcts
    if _neural_mcts is None:
        ckpt_dir = _Path("AlphaZero/checkpoints")
        best = ckpt_dir / "best.pt"
        latest = ckpt_dir / "latest.pt"
        model_path = best if best.exists() else latest
        if model_path.exists():
            print(f"加载模型: {model_path}")
            state = __import__('torch').load(str(model_path),
                                             map_location='cpu', weights_only=False)
            cfg = state.get('config', {})
            _neural_model = _PolicyWDLNet(
                num_blocks=cfg.get('num_blocks', 8),
                num_filters=cfg.get('num_filters', 128))
            _neural_model.load_state_dict(state['model_state_dict'])
            _neural_evaluator = _NeuralEval(_neural_model)
            _neural_mcts = _AlphaMCTS(
                _neural_evaluator,
                num_simulations=cfg.get('num_simulations', 800),
                c_puct=cfg.get('c_puct', 1.5),
            )
        else:
            print("没有找到 checkpoint，请先训练模型")
            return None
    return _neural_mcts


# ── WebSocket 连接管理 ──
from fastapi import WebSocket, WebSocketDisconnect

class _ConnectionManager:
    """管理 WebSocket 连接，按 game_id 分组广播"""
    def __init__(self):
        self._connections: dict[int, list[WebSocket]] = {}

    async def connect(self, game_id: int, ws: WebSocket):
        await ws.accept()
        self._connections.setdefault(game_id, []).append(ws)

    def disconnect(self, game_id: int, ws: WebSocket):
        if game_id in self._connections:
            self._connections[game_id].remove(ws)

    async def broadcast(self, game_id: int, data: dict):
        """向关注某对局的所有客户端推送消息"""
        for ws in self._connections.get(game_id, []):
            try:
                await ws.send_json(data)
            except Exception:
                pass

ws_manager = _ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="中国象棋 API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 托管前端静态文件
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def serve_index():
    """返回前端页面"""
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.post("/api/valid-moves")
def api_valid_moves(req: ValidMovesRequest):
    """获取某棋子的所有合法走法"""
    board = _parse_board(req.board)
    moves = get_valid_moves(board, req.row, req.col, req.turn)
    return {"moves": moves}


@app.post("/api/move", response_model=MoveResponse)
def api_move(req: MoveRequest):
    """执行走棋。如果传入 game_id 则同步写入数据库。"""
    board = _parse_board(req.board)
    from_pos = {"row": req.from_pos.row, "col": req.from_pos.col}
    to_pos = {"row": req.to_pos.row, "col": req.to_pos.col}

    piece = board[req.from_pos.row][req.from_pos.col]
    if not is_valid_move(board, from_pos, to_pos, req.turn):
        return MoveResponse(valid=False)

    new_board, captured = make_move(board, from_pos, to_pos)
    next_turn = "b" if req.turn == "r" else "r"
    result = check_game_result(new_board, next_turn)
    in_check = is_in_check(new_board, next_turn)

    game_over = None
    game_status = "ongoing"
    if result in ("red_win", "black_win", "draw"):
        game_over = result
        game_status = result

    # 如果关联了对局，同步写入数据库
    if req.game_id is not None:
        step = get_step_count(req.game_id) + 1
        record_step(
            game_id=req.game_id,
            step=step,
            board=new_board,
            turn=next_turn,
            from_pos=from_pos,
            to_pos=to_pos,
            piece=piece,
            captured=captured,
            status=game_status,
        )

    return MoveResponse(
        valid=True,
        captured=PieceInfo(**captured) if captured else None,
        check=in_check,
        game_over=game_over,
        new_board=_serialize_board(new_board),
    )


@app.post("/api/check-win")
def api_check_win(req: CheckWinRequest):
    """检测胜负"""
    board = _parse_board(req.board)
    result = check_game_result(board, req.turn)
    return {"result": result}


@app.post("/api/game/new", response_model=GameCreateResponse)
def api_new_game():
    """创建新对局"""
    board = create_initial_board()
    game_id = create_game(board)
    return GameCreateResponse(
        game_id=game_id,
        board=_serialize_board(board),
        turn="r",
    )


@app.get("/api/game/{game_id}", response_model=GameInfoResponse)
def api_get_game(game_id: int):
    """获取对局信息"""
    game = get_game(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="对局不存在")
    return GameInfoResponse(
        game_id=game["id"],
        board=game["board"],
        turn=game["turn"],
        status=game["status"],
        step_count=get_step_count(game_id),
    )


@app.get("/api/game/{game_id}/history")
def api_game_history(game_id: int):
    """获取走棋历史"""
    game = get_game(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="对局不存在")
    moves = get_moves(game_id)
    return {"game_id": game_id, "moves": moves}


@app.post("/api/game/{game_id}/undo")
def api_undo(game_id: int):
    """悔棋：删除最后一步，回滚棋盘"""
    result = undo_game(game_id, create_initial_board)
    if not result:
        raise HTTPException(status_code=400, detail="无法悔棋（对局不存在或无走棋记录）")
    return result


@app.post("/api/ai-move")
def api_ai_move(req: MoveRequest):
    """AI走棋：使用 MCTS 搜索选择最优走法。

    请求体格式与 /api/move 一致（board + turn），
    但只需 from_pos/to_pos 中的棋盘和轮次信息。

    返回: {from_pos, to_pos} 可直接传递给 /api/move 走棋。
    """
    board = _parse_board(req.board)
    state = _board_to_gamestate(board, req.turn)

    neural_mcts = _get_neural_mcts()
    if neural_mcts is None:
        raise HTTPException(status_code=503, detail="模型未加载，请先训练")

    move, _ = neural_mcts.select_move(state, temperature=0.0)
    if move is None:
        raise HTTPException(status_code=400, detail="AI找不到合法走法")

    return {
        "from_pos": {"row": move.from_row, "col": move.from_col},
        "to_pos": {"row": move.to_row, "col": move.to_col},
    }


# ============================================================
#  WebSocket — 实时观战
# ============================================================

@app.websocket("/ws/watch/{game_id}")
async def ws_watch_game(ws: WebSocket, game_id: int):
    """WebSocket 端点：实时观看对局。

    客户端连接后，服务端在有新走棋时推送 JSON:
      {"type": "move", "step": N, "from": [r,c], "to": [r,c],
       "board": [...], "turn": "r"/"b", "check": bool, "game_over": str|None}
    """
    await ws_manager.connect(game_id, ws)
    try:
        # 保持连接，等待客户端消息（用于心跳/控制）
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        ws_manager.disconnect(game_id, ws)


# ============================================================
#  辅助
# ============================================================

def _parse_board(board_data):
    """将 Pydantic 模型转为 dict 棋盘"""
    return [
        [
            {"type": cell.type, "color": cell.color} if cell else None
            for cell in row
        ]
        for row in board_data
    ]


def _board_to_gamestate(board, turn):
    """将 dict 棋盘转为 AlphaZero GameState"""
    import numpy as np
    _type_to_num = {
        "帥": 1, "仕": 2, "相": 3, "馬": 4, "車": 5, "炮": 6, "兵": 7,
        "將": 1, "士": 2, "象": 3,           "砲": 6, "卒": 7,
    }
    arr = np.zeros((10, 9), dtype=np.int8)
    for r in range(10):
        for c in range(9):
            p = board[r][c]
            if p:
                sign = 1 if p["color"] == "r" else -1
                arr[r, c] = sign * _type_to_num[p["type"]]
    return _AlphaGameState(arr, turn)


def _serialize_board(board):
    """将 dict 棋盘转为可序列化格式"""
    return [
        [cell if cell else None for cell in row]
        for row in board
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
