"""Shared colors and complete rule histories across Web, persistence and MCTS."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from fastapi.testclient import TestClient

from AlphaZero.engine import GameState, Move, RED, BLACK, player_id
from AlphaZero.engine.fast_chess import (
    create_initial_board, get_valid_moves, is_valid_move, is_in_check,
    has_any_valid_move, check_game_result)
from AlphaZero.engine.repetition import position_key
from AlphaZero.search import MCTS
from AlphaZero.tests.test_mcts_correctness import ConstantEvaluator
from backend import main, database as db
from backend.rule_adapter import web_board


@pytest.mark.parametrize("side,aliases,source,target", [
    (RED, [True, np.bool_(True), 1, np.int64(1), "r"], (6, 0), (5, 0)),
    (BLACK, [False, np.bool_(False), -1, np.int64(-1), "b"], (3, 0), (4, 0)),
])
def test_color_aliases_at_every_rule_boundary(side, aliases, source, target):
    board = create_initial_board()
    original = board.copy()
    for alias in aliases:
        assert player_id(alias) == side
        assert get_valid_moves(board, *source, alias) == get_valid_moves(board, *source, side == RED)
        assert is_valid_move(board, source, target, alias)
        assert has_any_valid_move(board, alias)
        assert not is_in_check(board, alias)
        assert check_game_result(board, alias) == "ongoing"
        assert GameState(board, alias).current_player == side
        assert position_key(board, alias) == position_key(board, side == RED)
    np.testing.assert_array_equal(board, original)


@pytest.mark.parametrize("invalid", [0, 2, -2, "black", "red", "", None, 1.0, np.float64(-1)])
def test_invalid_colors_fail_instead_of_silently_selecting_a_side(invalid):
    board = create_initial_board()
    for call in (lambda: player_id(invalid), lambda: is_in_check(board, invalid),
                 lambda: get_valid_moves(board, 3, 0, invalid),
                 lambda: is_valid_move(board, (3, 0), (4, 0), invalid),
                 lambda: has_any_valid_move(board, invalid),
                 lambda: check_game_result(board, invalid), lambda: position_key(board, invalid)):
        with pytest.raises(ValueError, match="invalid player"):
            call()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "web.sqlite"))
    with TestClient(main.app) as client:
        yield client


def same_state(actual, expected):
    for attr in ("turn", "move_count", "no_capture_count", "last_move", "history",
                 "position_history", "check_history", "no_capture_checks", "_max_ply"):
        assert getattr(actual, attr) == getattr(expected, attr), attr
    assert dict(actual.repetition_counts) == dict(expected.repetition_counts)
    assert actual.game_result() == expected.game_result()
    np.testing.assert_array_equal(actual.board, expected.board)
    np.testing.assert_array_equal(actual.encode(), expected.encode())
    np.testing.assert_array_equal(actual.legal_mask(), expected.legal_mask())


def request(state, game, revision, move=None):
    move = move or Move(0, 0, 0, 0)
    return dict(game_id=game, revision=revision, board=web_board(state),
        turn="r" if state.turn else "b", from_pos=dict(row=move.from_row, col=move.from_col),
        to_pos=dict(row=move.to_row, col=move.to_col))


HORSE_CYCLE = [Move(9, 1, 7, 2), Move(0, 1, 2, 2), Move(7, 2, 9, 1), Move(2, 2, 0, 1)]


def play(client, state, game, revision, moves):
    for move in moves:
        response = client.post("/api/move", json=request(state, game, revision, move))
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["valid"]
        state = state.apply(move)
        revision += 1
        assert data["revision"] == revision
        assert data["new_board"] == web_board(state)
        assert data["no_capture_count"] == state.no_capture_count
        assert data["natural_limit"] == state.natural_limit_status()
        stored, stored_revision = db.get_rule_game(game)
        assert stored_revision == revision
        same_state(stored, state)
    return state, revision


def test_web_repetition_and_undo_restore_every_history_field(client):
    game = client.post("/api/game/new").json()["game_id"]
    state, revision = play(client, GameState.new_game(), game, 0, HORSE_CYCLE * 2)
    assert state.game_result().reason == "repetition"
    response = client.post("/api/check-win", json=request(state, game, revision))
    assert response.json()["result"] == "draw"
    assert response.json()["rule_reason"] == "repetition"
    valid = request(state, game, revision)
    valid.update(row=9, col=1)
    assert client.post("/api/valid-moves", json=valid).json()["moves"] == []
    assert client.post("/api/ai-move", json=request(state, game, revision)).status_code == 400
    undone = client.post(f"/api/game/{game}/undo", json={"revision": revision}).json()
    assert undone["revision"] == revision + 1
    expected = GameState.new_game()
    for move in (HORSE_CYCLE * 2)[:-1]:
        expected = expected.apply(move)
    same_state(db.get_rule_game(game)[0], expected)
    assert undone["status"] == "ongoing"
    again, _ = play(client, expected, game, undone["revision"], [HORSE_CYCLE[-1]])
    assert again.game_result().reason == "repetition"


def test_search_receives_history_not_a_new_board_root(client, monkeypatch):
    game = client.post("/api/game/new").json()["game_id"]
    state, revision = play(client, GameState.new_game(), game, 0, HORSE_CYCLE + HORSE_CYCLE[:2])
    class RecordingSearch(MCTS):
        def select_move(self, received, **kwargs):
            same_state(received, state)
            assert kwargs["add_exploration_noise"] is False
            return super().select_move(received, **kwargs)
    monkeypatch.setattr(main, "_get_neural_mcts", lambda: RecordingSearch(ConstantEvaluator(), num_simulations=3))
    assert client.post("/api/ai-move", json=request(state, game, revision)).status_code == 200
    def heuristic(received):
        same_state(received, state)
        return received.legal_moves()[0]
    monkeypatch.setattr(main, "heuristic_move", heuristic)
    assert client.post("/api/heuristic-move", json=request(state, game, revision)).status_code == 200


def test_perpetual_check_survives_web_persistence_and_reload(client):
    board = np.zeros((10, 9), np.int8)
    board[0, 4], board[9, 3], board[1, 4] = -1, 1, 5
    state = GameState(board, BLACK)
    game = db.create_game(web_board(state), turn="b")
    cycle = [Move(0, 4, 0, 5), Move(1, 4, 1, 5), Move(0, 5, 0, 4), Move(1, 5, 1, 4)]
    state, revision = play(client, state, game, 0, cycle * 2)
    assert state.game_result().reason == "perpetual_check"
    assert client.get(f"/api/game/{game}").json()["status"] == "black_win"
    assert client.post("/api/check-win", json=request(state, game, revision)).json()["rule_reason"] == "perpetual_check"


def test_capture_counter_and_configured_ply_limit_are_shared(client):
    state = GameState.new_game(max_ply=4)
    game = db.create_game(web_board(state), max_ply=4)
    moves = [Move(6, 0, 5, 0), Move(3, 0, 4, 0), Move(5, 0, 4, 0)]
    state, revision = play(client, state, game, 0, moves)
    assert state.no_capture_count == 0
    state, revision = play(client, state, game, revision, [state.legal_moves()[0]])
    assert state.game_result().reason == "max_ply"
    assert client.get(f"/api/game/{game}").json()["rule_reason"] == "max_ply"


def test_stale_forged_and_historyless_requests_rejected(client):
    new = client.post("/api/game/new").json()
    state = GameState.new_game()
    game = new["game_id"]
    stale = request(state, game, 0, HORSE_CYCLE[0])
    state, revision = play(client, state, game, 0, [HORSE_CYCLE[0]])
    assert client.post("/api/move", json=stale).status_code == 409
    body = request(state, game, revision)
    body["board"][6][0] = None
    assert client.post("/api/ai-move", json=body).status_code == 409
    body = request(state, game, revision)
    del body["game_id"]
    assert client.post("/api/ai-move", json=body).status_code == 422
    assert len(db.get_moves(game)) == 1
    state, revision = play(client, state, game, revision, HORSE_CYCLE[1:])
    # Identical board and side to move can still have a different rule history.
    assert web_board(state) == stale["board"] and state.turn
    assert client.post("/api/move", json=stale).status_code == 409


def test_simultaneous_moves_commit_once(client):
    game = client.post("/api/game/new").json()["game_id"]
    def attempt(move):
        try:
            return db.commit_move(game, move, 0) is not None
        except db.StateConflict:
            return False
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(attempt, [Move(6, 0, 5, 0), Move(6, 2, 5, 2)]))
    assert sorted(results) == [False, True]
    assert len(db.get_moves(game)) == 1
    assert db.get_rule_game(game)[1] == 1


def test_legacy_log_replayed_and_corrupt_log_fails_without_deleting_moves(client):
    game = client.post("/api/game/new").json()["game_id"]
    state, _ = play(client, GameState.new_game(), game, 0, HORSE_CYCLE)
    with db.transaction(write=True) as conn:
        conn.execute("UPDATE games SET origin=NULL WHERE id=?", (game,))
    same_state(db.get_rule_game(game)[0], state)
    with db.transaction(write=True) as conn:
        conn.execute("UPDATE moves SET step=99 WHERE game_id=? AND step=2", (game,))
    response = client.post(f"/api/game/{game}/undo", json={"revision": 4})
    assert response.status_code == 409
    assert len(db.get_moves(game)) == 4


def test_ai_result_rejected_if_game_changes_during_search(client, monkeypatch):
    game = client.post("/api/game/new").json()["game_id"]
    state = GameState.new_game()
    class Interleaved:
        def select_move(self, received, **kwargs):
            move = received.legal_moves()[0]
            db.commit_move(game, move, 0)
            return move, None
    monkeypatch.setattr(main, "_get_neural_mcts", lambda: Interleaved())
    assert client.post("/api/ai-move", json=request(state, game, 0)).status_code == 409


def test_no_capture_limit_preserved_across_120_web_moves(client):
    state = GameState.new_game()
    game = client.post("/api/game/new").json()["game_id"]
    rng = np.random.default_rng(117)
    revision = 0
    for ply in range(120):
        choices = state.legal_moves()
        rng.shuffle(choices)
        for move in choices:
            if state.board[move.to_row, move.to_col] != 0:
                continue
            candidate = state.apply(move)
            if candidate.position_history[-1] in state.position_history:
                continue
            if ply < 119 and candidate.is_terminal():
                continue
            break
        else:
            pytest.fail("fixture cannot find a fresh non-capture continuation")
        state, revision = play(client, state, game, revision, [move])
    assert state.no_capture_count == 120
    assert state.game_result().reason == "no_capture"
    assert client.get(f"/api/game/{game}").json()["rule_reason"] == "no_capture"


def test_database_schema_upgrade_replays_legacy_initial_game(tmp_path, monkeypatch):
    import json
    import sqlite3
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "legacy.sqlite"))
    with sqlite3.connect(db.DB_PATH) as conn:
        conn.execute("""CREATE TABLE games (id INTEGER PRIMARY KEY,
            board_state TEXT NOT NULL, current_turn TEXT, status TEXT)""")
        conn.execute("INSERT INTO games VALUES (1,?,'r','ongoing')",
                     (json.dumps(web_board(GameState.new_game())),))
    db.init_db()
    state, revision = db.get_rule_game(1)
    assert revision == 0
    same_state(state, GameState.new_game())
    assert db.commit_move(1, HORSE_CYCLE[0], 0)[1] == 1


def test_real_neural_loader_uses_complete_state_and_checkpoint_budget(client, tmp_path, monkeypatch):
    import torch
    from AlphaZero.train.config import AlphaZeroConfig
    from AlphaZero.train.trainer import Trainer
    torch.set_num_threads(1)
    trainer = Trainer(AlphaZeroConfig(num_blocks=1, num_filters=4, batch_size=2,
                      num_simulations=3, device="cpu", precision="fp32"))
    state = GameState.new_game()
    class Batch:
        def __len__(self): return 2
        def sample(self, count):
            mask = state.legal_mask().astype(np.float32)
            return (np.stack([state.encode()] * count), np.stack([mask / mask.sum()] * count),
                    np.zeros(count, np.float32), np.ones(count))
    trainer.step(Batch())
    path = tmp_path / "actor.pt"
    trainer.export_actor(path)
    monkeypatch.setenv("XIANGQI_MODEL", str(path))
    monkeypatch.setattr(main, "_neural_mcts", None)
    game = client.post("/api/game/new").json()["game_id"]
    state, revision = play(client, state, game, 0, HORSE_CYCLE)
    response = client.post("/api/ai-move", json=request(state, game, revision))
    assert response.status_code == 200, response.text
    assert main._neural_mcts.diagnostics["simulations"] == 3
    assert db.get_rule_game(game)[1] == revision  # advice does not commit a move
