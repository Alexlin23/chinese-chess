"""All supported color representations use the authoritative engine."""
import numpy as np
import pytest
from AlphaZero.engine import GameState, RED, BLACK, player_id
from AlphaZero.engine.fast_chess import (
    create_initial_board, get_valid_moves, is_valid_move, is_in_check,
    has_any_valid_move, check_game_result)
from AlphaZero.engine.repetition import position_key


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
