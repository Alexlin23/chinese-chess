"""Shared displacement planes, flattened in logical NCHW order.

action = plane * 90 + from_row * 9 + from_col
The final convolution emits [N, 50, 10, 9]. Plane order is versioned; changing
it requires a new action version. Off-board slots stay in the softmax space,
but never enter a legal mask. Piece-specific constraints belong to the rules.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .constants import ROWS, COLS, BOARD_SIZE, POLICY_PLANES, POLICY_SIZE


# 0..17 vertical, 18..33 horizontal, 34..41 horse,
# 42..45 elephant, 46..49 advisor. King and pawn share straight planes.
ACTION_DELTAS = (
    tuple((-distance, 0) for distance in range(1, ROWS))
    + tuple((distance, 0) for distance in range(1, ROWS))
    + tuple((0, -distance) for distance in range(1, COLS))
    + tuple((0, distance) for distance in range(1, COLS))
    + ((-2, -1), (-2, 1), (-1, -2), (-1, 2),
       (1, -2), (1, 2), (2, -1), (2, 1))
    + ((-2, -2), (-2, 2), (2, -2), (2, 2))
    + ((-1, -1), (-1, 1), (1, -1), (1, 1))
)
assert len(ACTION_DELTAS) == POLICY_PLANES
_PLANE_FOR_DELTA = {delta: plane for plane, delta in enumerate(ACTION_DELTAS)}
_NEGATIVE_PLANES = np.array([_PLANE_FOR_DELTA[(-dr, -dc)] for dr, dc in ACTION_DELTAS])
_ROTATED_ACTIONS = (
    _NEGATIVE_PLANES[:, None] * BOARD_SIZE
    + np.arange(BOARD_SIZE - 1, -1, -1)[None, :]
).reshape(POLICY_SIZE)
_ROTATED_ACTIONS.flags.writeable = False


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


@dataclass(frozen=True)
class Move:
    """Immutable board coordinates, independent of the policy representation."""
    from_row: int
    from_col: int
    to_row: int
    to_col: int

    @property
    def from_square(self) -> int:
        return self.from_row * COLS + self.from_col

    @property
    def to_square(self) -> int:
        return self.to_row * COLS + self.to_col

    def __repr__(self):
        return f"Move({self.from_row},{self.from_col}→{self.to_row},{self.to_col})"


class ActionEncoder:
    """Move <-> 50 displacement planes; invalid coordinates return -1/None."""

    @staticmethod
    def in_bounds(row: int, col: int) -> bool:
        return (_integer(row) and _integer(col)
                and 0 <= row < ROWS and 0 <= col < COLS)

    @staticmethod
    def encode(move: Move) -> int:
        if (not isinstance(move, Move)
                or not ActionEncoder.in_bounds(move.from_row, move.from_col)
                or not ActionEncoder.in_bounds(move.to_row, move.to_col)):
            return -1
        delta = (move.to_row - move.from_row, move.to_col - move.from_col)
        plane = _PLANE_FOR_DELTA.get(delta)
        return -1 if plane is None else int(plane * BOARD_SIZE + move.from_square)

    @staticmethod
    def decode(action: int) -> Optional[Move]:
        if not _integer(action) or not 0 <= action < POLICY_SIZE:
            return None
        plane, square = divmod(int(action), BOARD_SIZE)
        row, col = divmod(square, COLS)
        dr, dc = ACTION_DELTAS[plane]
        if not ActionEncoder.in_bounds(row + dr, col + dc):
            return None
        return Move(row, col, row + dr, col + dc)

    @staticmethod
    def rotate_action(action: int) -> int:
        """Rotate origin AND displacement by 180 degrees, including padding."""
        if not _integer(action) or not 0 <= action < POLICY_SIZE:
            return -1
        return int(_ROTATED_ACTIONS[action])

    @staticmethod
    def rotate_policy(policy: np.ndarray) -> np.ndarray:
        """Involution between absolute/current-player action coordinates.

        Supports any leading batch dimensions and preserves dtype. A simple
        reversal is incorrect: opposite displacement planes are not in reverse
        order. Taking this permutation twice recovers the original array.
        """
        policy = np.asarray(policy)
        if policy.ndim < 1 or policy.shape[-1] != POLICY_SIZE:
            raise ValueError(f"policy must have a final dimension of {POLICY_SIZE}")
        return np.take(policy, _ROTATED_ACTIONS, axis=-1)

    @staticmethod
    def legal_actions(legal_moves: list) -> np.ndarray:
        actions = [ActionEncoder.encode(move) for move in legal_moves]
        if any(action < 0 for action in actions):
            raise ValueError("legal move has no displacement-plane encoding")
        return np.asarray(actions, dtype=np.int32)

    @staticmethod
    def legal_mask(legal_moves: list) -> np.ndarray:
        mask = np.zeros(POLICY_SIZE, dtype=bool)
        mask[ActionEncoder.legal_actions(legal_moves)] = True
        return mask

    @staticmethod
    def mask_logits(logits: np.ndarray, legal_moves: list) -> np.ndarray:
        logits = np.asarray(logits)
        if logits.shape != (POLICY_SIZE,):
            raise ValueError(f"logits must have shape ({POLICY_SIZE},)")
        return np.where(ActionEncoder.legal_mask(legal_moves), logits, -np.inf)
