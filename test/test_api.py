"""
测试 /api/valid-moves 和 /api/move
"""
import json, sqlite3, os
from urllib.request import Request, urlopen

BASE = "http://127.0.0.1:8001"

def post(path, body):
    data = json.dumps(body).encode()
    req = Request(BASE + path, data=data, method="POST",
                  headers={"Content-Type": "application/json"})
    return json.loads(urlopen(req).read())

# 从DB直接拿最新对局
db = os.path.join(os.path.dirname(__file__), "..", "backend", "chess.db")
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row
row = conn.execute("SELECT id, board_state, current_turn FROM games WHERE status='ongoing' ORDER BY id DESC LIMIT 1").fetchone()
conn.close()

if not row:
    print("没有ongoing的对局，先去网页走几步")
    exit(1)

gid = row["id"]
board = json.loads(row["board_state"])
turn = row["current_turn"]
print(f"对局ID={gid}, turn={turn}")

# 找马、查可走
horse_positions = []
for r in range(10):
    for c in range(9):
        p = board[r][c]
        if p and p["type"] == "馬" and p["color"] == turn:
            resp = post("/api/valid-moves", {"board": board, "row": r, "col": c, "turn": turn})
            moves = resp["moves"]
            print(f"马({r},{c})可走: {[(m['row'],m['col']) for m in moves]}")
            if moves:
                horse_positions.append((r, c, moves))

# ============================================================
#  走棋：把第一匹马走到首个可选位置
# ============================================================
if not horse_positions:
    print("没有可走的马")
    exit(1)

r, c, moves = horse_positions[0]
to_row, to_col = moves[0]["row"], moves[0]["col"]
print(f"\n=== 走棋：马({r},{c}) -> ({to_row},{to_col}) ===")

move_resp = post("/api/move", {
    "from_pos": {"row": r, "col": c},
    "to_pos": {"row": to_row, "col": to_col},
    "board": board,
    "turn": turn,
    "game_id": gid,
})

print(f"valid={move_resp['valid']}")
if move_resp.get("captured"):
    print(f"吃掉: {move_resp['captured']}")
if move_resp.get("check"):
    print("将军！")
if move_resp.get("game_over"):
    print(f"对局结束: {move_resp['game_over']}")
if move_resp.get("new_board"):
    print("new_board 已返回")
