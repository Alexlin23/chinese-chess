"""Explicit conversion at rule API boundaries; bool(-1) is not a color."""
import numpy as np
from .constants import RED, BLACK


def player_id(player):
    if isinstance(player, (bool, np.bool_)):
        return RED if player else BLACK
    if isinstance(player, str):
        if player in ("r", "b"):
            return RED if player == "r" else BLACK
    elif isinstance(player, (int, np.integer)) and player in (RED, BLACK):
        return int(player)
    raise ValueError(f"invalid player: {player!r}")
