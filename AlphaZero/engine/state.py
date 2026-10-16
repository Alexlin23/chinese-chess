"""Immutable rule state and eight-frame current-player observations.

Target: Chinese Xiangqi Competition Rules (2020). Chase adjudication is incomplete.
Do not describe this development rule version as fully compliant.
"""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional
import numpy as np
from .constants import ROWS, COLS, KING, RED, BLACK, MAX_GAME_PLY, INPUT_CHANNELS
from .fast_chess import create_initial_board, get_valid_moves, make_move, is_in_check
from .move import Move, ActionEncoder
from .repetition import position_key

def player_id(player):
    if isinstance(player, (bool, np.bool_)):
        return RED if player else BLACK
    if player in ("r", RED):
        return RED
    if player in ("b", BLACK):
        return BLACK
    raise ValueError(f"invalid player: {player!r}")

@dataclass(frozen=True)
class GameResult:
    winner: Optional[int]
    reason: str

    @property
    def is_draw(self):
        return self.winner is None

    def to_value(self, current_player):
        return 0.0 if self.is_draw else (1.0 if self.winner == player_id(current_player) else -1.0)

    def to_wdl(self, current_player):
        value = self.to_value(current_player)
        return np.array([value > 0, value == 0, value < 0], dtype=np.float32)

class GameState:
    __slots__ = ("board", "turn", "move_count", "no_capture_count", "repetition_counts",
                 "last_move", "_max_ply", "_encoding", "_legal_mask", "_moves",
                 "history", "position_history", "check_history", "_outcome", "_resolved")

    def __setattr__(self, name, value):
        if (hasattr(self, "_resolved") and name not in
                {"_encoding", "_legal_mask", "_moves", "_outcome", "_resolved"}):
            raise AttributeError("rule snapshots are immutable")
        object.__setattr__(self, name, value)

    def __init__(self, board, turn, move_count=0, no_capture_count=0,
                 repetition_counts=None, last_move=None, max_ply=None, *,
                 history=None, position_history=None, check_history=()):
        original = np.asarray(board)
        if (original.shape != (ROWS, COLS) or original.dtype.kind not in "iu"
                or np.any(original < -7) or np.any(original > 7)):
            raise ValueError("board must be a 10x9 matrix of pieces -7..7")
        data = np.frombuffer(original.astype(np.int8).tobytes(), dtype=np.int8).reshape(ROWS,COLS)
        self.board = data
        self.turn = player_id(turn) == RED
        self.move_count, self.no_capture_count = int(move_count), int(no_capture_count)
        self._max_ply = MAX_GAME_PLY if max_ply is None else int(max_ply)
        if min(self.move_count, self.no_capture_count) < 0 or self._max_ply < 1:
            raise ValueError("invalid rule counters")
        key = position_key(data, self.turn)
        counts = dict(repetition_counts) if repetition_counts is not None else {key: 1}
        self.repetition_counts = MappingProxyType(counts)
        self.history = tuple(history) if history is not None else ((data.tobytes(), counts.get(key, 1)),)
        self.position_history = tuple(position_history) if position_history is not None else (key,)
        self.check_history = tuple(check_history)
        self.last_move = last_move
        self._encoding = self._legal_mask = self._moves = self._outcome = None
        self._resolved = False

    @property
    def current_player(self):
        return RED if self.turn else BLACK

    def copy(self):
        return self

    @classmethod
    def new_game(cls, max_ply=None):
        return cls(create_initial_board(), RED, max_ply=max_ply)

    def legal_moves(self):
        if self._moves is None:
            # Move generator temporarily modifies this private workspace.
            board = self.board.copy()
            rows, cols = np.where(board * self.current_player > 0)
            self._moves = tuple(Move(int(r), int(c), int(m["row"]), int(m["col"]))
                                for r, c in zip(rows, cols)
                                for m in get_valid_moves(board, int(r), int(c), self.turn))
        return list(self._moves)

    def legal_actions(self):
        return ActionEncoder.legal_actions(self.legal_moves())

    def legal_mask(self):
        if self._legal_mask is None:
            self._legal_mask = ActionEncoder.legal_mask(self.legal_moves())
            self._legal_mask.flags.writeable = False
        return self._legal_mask

    def canonical_mask(self):
        mask = self.legal_mask()
        return mask if self.turn else mask[::-1].copy()

    def canonical_policy(self, policy):
        return np.asarray(policy) if self.turn else np.asarray(policy)[::-1].copy()

    def is_legal(self, move):
        return move in self.legal_moves()

    def apply(self, move):
        if self.is_terminal() or not self.is_legal(move):
            raise ValueError(f"illegal move or finished game: {move}")
        board = self.board.copy()
        captured = make_move(board, (move.from_row, move.from_col),
                             (move.to_row, move.to_col))["captured"]
        key = position_key(board, not self.turn)
        counts = dict(self.repetition_counts)
        counts[key] = counts.get(key, 0) + 1
        history = ((board.tobytes(), counts[key]),) + self.history[:7]
        return GameState(board, not self.turn, self.move_count + 1,
                         0 if captured else self.no_capture_count + 1,
                         counts, move, self._max_ply, history=history,
                         position_history=self.position_history + (key,),
                         check_history=self.check_history +
                         ((self.current_player, bool(is_in_check(board, not self.turn))),))

    def apply_action(self, action):
        return self.apply(ActionEncoder.decode(action))

    def game_result(self):
        if self._resolved:
            return self._outcome
        result = None
        if not np.any(self.board == KING):
            result = GameResult(BLACK, "king_captured")
        elif not np.any(self.board == -KING):
            result = GameResult(RED, "king_captured")
        elif not self.legal_moves():
            result = GameResult(-self.current_player, "no_legal_move")
        else:
            key = position_key(self.board, self.turn)
            if self.repetition_counts.get(key, 0) >= 3:
                occurrences = [i for i, k in enumerate(self.position_history) if k == key]
                checkers = []
                if len(occurrences) >= 3:
                    cycle = self.check_history[occurrences[-3]:]
                    for side in (RED, BLACK):
                        checks = [check for mover, check in cycle if mover == side]
                        if checks and all(checks):
                            checkers.append(side)
                result = (GameResult(-checkers[0], "perpetual_check") if len(checkers) == 1
                          else GameResult(None, "repetition"))
            elif self.no_capture_count >= 120:
                result = GameResult(None, "no_capture")
            elif self.move_count >= self._max_ply:
                result = GameResult(None, "max_ply")
        self._outcome, self._resolved = result, True
        return result

    def is_terminal(self):
        return self.game_result() is not None

    def result(self):
        result = self.game_result()
        return None if result is None else result.to_value(RED)

    def result_for_player(self):
        result = self.game_result()
        return None if result is None else result.to_value(self.current_player)

    def wdl_for_player(self):
        result = self.game_result()
        return None if result is None else result.to_wdl(self.current_player)

    def has_any_valid_move(self):
        return bool(self.legal_moves())

    def is_in_check(self):
        return bool(is_in_check(self.board, self.turn))

    def encode(self):
        if self._encoding is None:
            encoded = np.zeros((INPUT_CHANNELS, ROWS, COLS), dtype=np.float32)
            for frame, (raw, repetitions) in enumerate(self.history[:8]):
                board = np.frombuffer(raw, dtype=np.int8).reshape(ROWS, COLS)
                board = board if self.turn else board[::-1, ::-1]
                board = board * self.current_player
                for piece in range(1, 8):
                    encoded[frame * 16 + piece - 1] = board == piece
                    encoded[frame * 16 + piece + 6] = board == -piece
                encoded[frame * 16 + 14] = repetitions >= 2
                encoded[frame * 16 + 15] = repetitions >= 3
            encoded[128] = self.turn
            encoded[129] = self.move_count / self._max_ply
            encoded[130] = min(self.no_capture_count, 120) / 120
            encoded.flags.writeable = False
            self._encoding = encoded
        return self._encoding

    def __repr__(self):
        return f"GameState(player={self.current_player}, ply={self.move_count})"
