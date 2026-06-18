"""SQLite 数据库操作"""
import sqlite3
import json
import os
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chess.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            board_state TEXT NOT NULL,
            current_turn TEXT DEFAULT 'r',
            status TEXT DEFAULT 'ongoing'
        );
        CREATE TABLE IF NOT EXISTS moves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER NOT NULL,
            step INTEGER NOT NULL,
            from_row INTEGER, from_col INTEGER,
            to_row INTEGER, to_col INTEGER,
            piece_type TEXT, piece_color TEXT,
            captured_type TEXT, captured_color TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (game_id) REFERENCES games(id)
        );
    """)
    conn.commit()
    conn.close()


def create_game(board: list, turn: str = "r") -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO games (board_state, current_turn) VALUES (?, ?)",
        (json.dumps(board), turn)
    )
    game_id = cur.lastrowid
    conn.commit()
    conn.close()
    return game_id


def get_game(game_id: int) -> Optional[dict]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row["id"],
        "board": json.loads(row["board_state"]),
        "turn": row["current_turn"],
        "status": row["status"],
    }


def update_game(game_id: int, board: list, turn: str, status: str = "ongoing",
                conn: Optional[sqlite3.Connection] = None):
    """更新对局状态。传入 conn 可参与外部事务。"""
    close_conn = conn is None
    if conn is None:
        conn = get_conn()
    conn.execute(
        "UPDATE games SET board_state = ?, current_turn = ?, status = ? WHERE id = ?",
        (json.dumps(board), turn, status, game_id)
    )
    if close_conn:
        conn.commit()
        conn.close()


def add_move(game_id: int, step: int, from_pos: dict, to_pos: dict,
             piece: dict, captured: Optional[dict],
             conn: Optional[sqlite3.Connection] = None):
    """记录一步走棋。传入 conn 可参与外部事务。"""
    close_conn = conn is None
    if conn is None:
        conn = get_conn()
    conn.execute(
        """INSERT INTO moves
           (game_id, step, from_row, from_col, to_row, to_col,
            piece_type, piece_color, captured_type, captured_color)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (game_id, step, from_pos["row"], from_pos["col"],
         to_pos["row"], to_pos["col"],
         piece["type"], piece["color"],
         captured["type"] if captured else None,
         captured["color"] if captured else None)
    )
    if close_conn:
        conn.commit()
        conn.close()


def get_moves(game_id: int) -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM moves WHERE game_id = ? ORDER BY step",
        (game_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def record_step(game_id: int, step: int, board: list, turn: str,
                from_pos: dict, to_pos: dict, piece: dict,
                captured: Optional[dict], status: str = "ongoing"):
    """在一笔事务中同时写入走棋记录和更新棋盘（原子操作）。"""
    conn = get_conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        update_game(game_id, board, turn, status, conn=conn)
        add_move(game_id, step, from_pos, to_pos, piece, captured, conn=conn)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_step_count(game_id: int) -> int:
    """获取对局当前步数"""
    conn = get_conn()
    row = conn.execute(
        "SELECT MAX(step) as cnt FROM moves WHERE game_id = ?",
        (game_id,)
    ).fetchone()
    conn.close()
    return row["cnt"] or 0
