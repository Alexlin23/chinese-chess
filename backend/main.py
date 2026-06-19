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


def _serialize_board(board):
    """将 dict 棋盘转为可序列化格式"""
    return [
        [cell if cell else None for cell in row]
        for row in board
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
