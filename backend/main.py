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
from AlphaZero.search import MCTS as _AlphaMCTS, RandomEvaluator as _RandomEval
from AlphaZero.model import AlphaZeroNet as _AlphaNet, NeuralEvaluator as _NeuralEval
from AlphaZero.train.config import cpu_config as _cpu_config
from AlphaZero.train.self_play import SelfPlayGame as _SelfPlayGame

_ai_evaluator = _RandomEval(seed=42)
_ai_mcts = _AlphaMCTS(_ai_evaluator, num_simulations=200, c_puct=1.5)

# 神经网络评估器（懒加载）
_neural_model = None
_neural_evaluator = None
_neural_mcts = None


def _get_neural_mcts():
    """懒加载：创建/获取神经网络 MCTS 实例"""
    global _neural_model, _neural_evaluator, _neural_mcts
    if _neural_mcts is None:
        cfg = _cpu_config()
        _neural_model = _AlphaNet(num_blocks=cfg.num_blocks,
                                  num_filters=cfg.num_filters)
        # 尝试加载最新 checkpoint
        ckpt_dir = _Path(cfg.checkpoint_dir)
        ckpts = sorted(ckpt_dir.glob("model_iter*.pt")) if ckpt_dir.exists() else []
        if ckpts:
            print(f"加载最新模型: {ckpts[-1]}")
            state = __import__('torch').load(str(ckpts[-1]),
                                             map_location='cpu', weights_only=True)
            _neural_model.load_state_dict(state['model_state_dict'])
        _neural_evaluator = _NeuralEval(_neural_model)
        _neural_mcts = _AlphaMCTS(
            _neural_evaluator,
            num_simulations=cfg.num_simulations,
            c_puct=cfg.c_puct,
        )
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
    if result in ("red_win", "black_win"):
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

    move, _ = _ai_mcts.select_move(state, temperature=0.0)
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
#  自我对弈（Demo 观战 + 后台训练）
# ============================================================

@app.post("/api/self-play/start")
async def api_self_play_start():
    """启动一局神经网络自我对弈（用于前端实时观战）。

    返回 game_id，前端可通过 WebSocket /ws/watch/{game_id} 实时观看，
    或通过 /api/game/{game_id}/history 回放。
    """
    import asyncio

    # 创建对局记录
    board = create_initial_board()
    game_id = create_game(board)

    # 后台启动自我对弈
    asyncio.create_task(_run_self_play_demo(game_id))

    return {"game_id": game_id, "message": "自我对弈已启动，请通过 WebSocket 观看"}


async def _run_self_play_demo(game_id: int):
    """后台任务：运行一局神经网络自我对弈，每步推送到 WebSocket。

    与 bulk 训练不同，此函数:
      - 每步延迟 0.5s（方便人类观看）
      - 实时推送到 WebSocket
      - 写入数据库供回放
    """
    import time
    import asyncio
    import numpy as np

    try:
        mcts = _get_neural_mcts()
        cfg = _cpu_config()

        # 使用 SelfPlayGame 逐步执行
        game = _SelfPlayGame(mcts, cfg, game_id=game_id)

        while not game.is_terminal():
            # 执行一步
            move_record = game.step()
            if move_record is None:
                break

            # 获取当前棋盘状态
            board_dict = game.current_board_dict()
            turn_str = "r" if game.state.turn else "b"
            result = game.state.result()
            game_over = None
            if result is not None:
                if result > 0.5:
                    game_over = "red_win"
                elif result < -0.5:
                    game_over = "black_win"
                else:
                    game_over = "draw"

            # 写入数据库
            try:
                from database import record_step
                record_step(
                    game_id=game_id,
                    step=move_record["step"],
                    board=_serialize_board(board_dict),
                    turn=turn_str,
                    from_pos={"row": move_record["from_row"],
                              "col": move_record["from_col"]},
                    to_pos={"row": move_record["to_row"],
                            "col": move_record["to_col"]},
                    piece={"type": "", "color": ""},  # 简化
                    captured=None,
                    status=game_over or "ongoing",
                )
            except Exception as e:
                print(f"DB write error: {e}")

            # WebSocket 广播
            await ws_manager.broadcast(game_id, {
                "type": "move",
                "step": move_record["step"],
                "from": [move_record["from_row"], move_record["from_col"]],
                "to": [move_record["to_row"], move_record["to_col"]],
                "board": _serialize_board(board_dict),
                "turn": turn_str,
                "check": game.state.is_in_check(),
                "game_over": game_over,
            })

            # 延迟以便观看（终局前最后几步不减慢）
            delay = 0.8 if not game.is_terminal() else 0.3
            await asyncio.sleep(delay)

        # 对局结束
        final_result = game.result()
        result_str = "draw"
        if final_result and final_result > 0.5:
            result_str = "red_win"
        elif final_result and final_result < -0.5:
            result_str = "black_win"

        await ws_manager.broadcast(game_id, {
            "type": "game_over",
            "result": result_str,
            "total_steps": game.state.move_count,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        await ws_manager.broadcast(game_id, {
            "type": "error",
            "message": str(e),
        })


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
