"""Transactional Web games reconstructed through the shared GameState engine."""
from contextlib import contextmanager
import json
import os
import sqlite3
import numpy as np
from AlphaZero.engine import GameState, Move
from AlphaZero.train.config import RULE_VERSION
from .rule_adapter import initial_state, numeric_board, web_board, status

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chess.db")

class StateConflict(ValueError):
    pass


def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def transaction(write=False):
    conn = get_conn()
    try:
        conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with transaction(write=True) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            board_state TEXT NOT NULL, current_turn TEXT DEFAULT 'r',
            status TEXT DEFAULT 'ongoing', origin TEXT, revision INTEGER NOT NULL DEFAULT 0)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS moves (
            id INTEGER PRIMARY KEY AUTOINCREMENT, game_id INTEGER NOT NULL,
            step INTEGER NOT NULL, from_row INTEGER, from_col INTEGER,
            to_row INTEGER, to_col INTEGER, piece_type TEXT, piece_color TEXT,
            captured_type TEXT, captured_color TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (game_id) REFERENCES games(id))""")
        columns = {row[1] for row in conn.execute("PRAGMA table_info(games)")}
        if "origin" not in columns:
            conn.execute("ALTER TABLE games ADD COLUMN origin TEXT")
        if "revision" not in columns:
            conn.execute("ALTER TABLE games ADD COLUMN revision INTEGER NOT NULL DEFAULT 0")


def _origin_payload(state):
    return json.dumps(dict(board=web_board(state), turn="r" if state.turn else "b",
                           max_ply=state._max_ply, rule_version=RULE_VERSION))


def create_game(board=None, turn="r", max_ply=None):
    state = GameState.new_game(max_ply) if board is None else initial_state(board, turn, max_ply)
    with transaction(write=True) as conn:
        cur = conn.execute("""INSERT INTO games
            (board_state,current_turn,status,origin) VALUES (?,?,?,?)""",
            (json.dumps(web_board(state)), "r" if state.turn else "b", status(state), _origin_payload(state)))
        return cur.lastrowid


def _origin(row):
    if row["origin"] is None:
        # Legacy games started from the standard position. Verify their full log
        # against the stored board instead of silently resetting the history.
        return GameState.new_game()
    origin = json.loads(row["origin"])
    if origin.get("rule_version") != RULE_VERSION:
        raise StateConflict("对局规则版本不兼容，需要明确迁移")
    return initial_state(origin["board"], origin["turn"], origin["max_ply"])


def _replay(state, moves):
    for expected, move in enumerate(moves, 1):
        if move["step"] != expected:
            raise StateConflict("对局棋谱步数不连续，无法恢复完整规则状态")
        board = web_board(state)
        fr, fc, tr, tc = (move[k] for k in ("from_row", "from_col", "to_row", "to_col"))
        if any(type(v) is not int for v in (fr, fc, tr, tc)) or not (0 <= fr < 10 and 0 <= tr < 10 and 0 <= fc < 9 and 0 <= tc < 9):
            raise StateConflict("对局棋谱坐标无效")
        piece, captured = board[fr][fc], board[tr][tc]
        recorded_piece = dict(type=move["piece_type"], color=move["piece_color"])
        recorded_capture = (dict(type=move["captured_type"], color=move["captured_color"])
                            if move["captured_type"] else None)
        if piece != recorded_piece or captured != recorded_capture:
            raise StateConflict("对局棋谱棋子记录不一致")
        try:
            state = state.apply(Move(fr, fc, tr, tc))
        except ValueError as exc:
            raise StateConflict("对局棋谱含非法着法或终局后的走棋") from exc
    return state


def _load(conn, game_id):
    row = conn.execute("SELECT * FROM games WHERE id=?", (game_id,)).fetchone()
    if row is None:
        raise KeyError(game_id)
    moves = conn.execute("SELECT * FROM moves WHERE game_id=? ORDER BY step,id", (game_id,)).fetchall()
    state = _replay(_origin(row), moves)
    if (not np.array_equal(state.board, numeric_board(json.loads(row["board_state"])))
            or ("r" if state.turn else "b") != row["current_turn"]):
        raise StateConflict("棋盘与完整棋谱不一致，不能将其当作新局面继续")
    return state, row, moves


def get_rule_game(game_id):
    with transaction() as conn:
        state, row, _ = _load(conn, game_id)
        return state, row["revision"]


def _public(game_id, state, revision):
    result = state.game_result()
    return dict(id=game_id, board=web_board(state), turn="r" if state.turn else "b",
                status=status(state), step_count=state.move_count, revision=revision,
                no_capture_count=state.no_capture_count,
                natural_limit=state.natural_limit_status(),
                rule_reason=result.reason if result else None)


def get_game(game_id):
    try:
        state, revision = get_rule_game(game_id)
    except KeyError:
        return None
    return _public(game_id, state, revision)


def get_moves(game_id):
    with transaction() as conn:
        return [dict(row) for row in conn.execute(
            "SELECT * FROM moves WHERE game_id=? ORDER BY step,id", (game_id,))]


def get_step_count(game_id):
    return len(get_moves(game_id))


def _store(conn, game_id, state, revision, origin):
    conn.execute("""UPDATE games SET board_state=?,current_turn=?,status=?,revision=?,
        origin=COALESCE(origin,?) WHERE id=?""",
        (json.dumps(web_board(state)), "r" if state.turn else "b", status(state),
         revision, origin, game_id))


def commit_move(game_id, move, expected_revision):
    with transaction(write=True) as conn:
        state, row, _ = _load(conn, game_id)
        if row["revision"] != expected_revision:
            raise StateConflict("棋局已更新，请刷新后重试")
        if state.is_terminal() or not state.is_legal(move):
            return None
        board = web_board(state)
        piece, captured = board[move.from_row][move.from_col], board[move.to_row][move.to_col]
        next_state = state.apply(move)
        conn.execute("""INSERT INTO moves
            (game_id,step,from_row,from_col,to_row,to_col,piece_type,piece_color,captured_type,captured_color)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (game_id, next_state.move_count, move.from_row, move.from_col, move.to_row, move.to_col,
             piece["type"], piece["color"], captured["type"] if captured else None,
             captured["color"] if captured else None))
        revision = row["revision"] + 1
        _store(conn, game_id, next_state, revision, row["origin"] or _origin_payload(_origin(row)))
        return next_state, revision, captured


def undo_game(game_id, expected_revision):
    with transaction(write=True) as conn:
        _, row, moves = _load(conn, game_id)
        if row["revision"] != expected_revision:
            raise StateConflict("棋局已更新，请刷新后重试")
        if not moves:
            return None
        origin = _origin(row)
        state = _replay(origin, moves[:-1])
        conn.execute("DELETE FROM moves WHERE id=?", (moves[-1]["id"],))
        revision = row["revision"] + 1
        _store(conn, game_id, state, revision, row["origin"] or _origin_payload(origin))
        return _public(game_id, state, revision)
