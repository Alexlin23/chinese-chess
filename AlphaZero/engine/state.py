"""棋盘状态封装 — GameState 不可变快照"""
from typing import Optional
import numpy as np

from .constants import ROWS, COLS, KING, ADVISOR, ELEPHANT, KNIGHT, ROOK, CANNON, PAWN
from .fast_chess import (
    create_initial_board, get_valid_moves, make_move,
    check_game_result, is_in_check, has_any_valid_move,
)
from .move import Move, ActionEncoder


class GameState:
    """不可变棋盘状态快照。

    内部持有:
      - board: numpy int8 (10,9)
      - turn: bool (True=红, False=黑)
      - move_count: int  已走步数
      - no_capture_count: int  连续无吃子步数（用于重复局面检测）

    走棋通过 apply() 创建新快照，不修改自身。
    """

    __slots__ = ('board', 'turn', 'move_count', 'no_capture_count')

    def __init__(self, board: np.ndarray, turn, move_count=0, no_capture_count=0):
        self.board = board
        if isinstance(turn, str):
            self.turn = (turn == "r")
        else:
            self.turn = bool(turn)
        self.move_count = move_count
        self.no_capture_count = no_capture_count

    @classmethod
    def new_game(cls) -> 'GameState':
        """创建初始棋盘状态。"""
        return cls(create_initial_board(), True)

    # ── 走法查询 ──

    def legal_moves(self) -> list[Move]:
        """当前局面所有合法走法。"""
        moves = []
        sign = 1 if self.turn else -1
        rows, cols = np.where(self.board * sign > 0)
        for r, c in zip(rows, cols):
            valid = get_valid_moves(self.board, int(r), int(c), self.turn)
            for m in valid:
                moves.append(Move(int(r), int(c), int(m["row"]), int(m["col"])))
        return moves

    def is_legal(self, move: Move) -> bool:
        """单步走法合法性校验。"""
        from .fast_chess import is_valid_move
        return is_valid_move(
            self.board,
            (move.from_row, move.from_col),
            (move.to_row, move.to_col),
            self.turn,
        )

    # ── 终局检测 ──

    def is_terminal(self) -> bool:
        return check_game_result(self.board, self.turn) != "ongoing"

    def result(self) -> Optional[float]:
        """终局结果: +1=红胜, -1=黑胜, 0=和棋, None=未终局。"""
        r = check_game_result(self.board, self.turn)
        if r == "red_win":
            return 1.0
        elif r == "black_win":
            return -1.0
        elif r == "draw":
            return 0.0
        return None

    def is_in_check(self) -> bool:
        return is_in_check(self.board, self.turn)

    # ── 状态迁移 ──

    def apply(self, move: Move) -> 'GameState':
        """执行走法，返回新状态（不修改自身）。"""
        new_board = self.board.copy()
        undo = make_move(new_board,
                         (move.from_row, move.from_col),
                         (move.to_row, move.to_col))
        captured = undo["captured"]
        return GameState(
            board=new_board,
            turn=not self.turn,
            move_count=self.move_count + 1,
            no_capture_count=0 if captured else self.no_capture_count + 1,
        )

    # ── 神经网络编码 ──

    def encode(self) -> np.ndarray:
        """编码为神经网络输入 (18, 10, 9) float32。

        通道:
          0-6:  己方棋子位 (王..兵)
          7-13: 对方棋子位
          14:   己方颜色标记 (全1)
          15:   总步数 / 200
          16:   无吃子步数 / 120
          17:   将军标记
        """
        encoded = np.zeros((18, ROWS, COLS), dtype=np.float32)

        my_sign = 1 if self.turn else -1
        my_mask = self.board * my_sign > 0
        opp_mask = self.board * my_sign < 0

        # 己方棋子通道
        for ptype in range(1, 8):
            piece_mask = np.abs(self.board) == ptype
            encoded[ptype - 1] = (piece_mask & my_mask).astype(np.float32)

        # 对方棋子通道
        for ptype in range(1, 8):
            piece_mask = np.abs(self.board) == ptype
            encoded[ptype + 6] = (piece_mask & opp_mask).astype(np.float32)

        # 己方颜色标记
        encoded[14] = 1.0

        # 步数归一化
        encoded[15] = min(self.move_count, 200) / 200.0
        encoded[16] = min(self.no_capture_count, 120) / 120.0

        # 将军标记
        if is_in_check(self.board, self.turn):
            encoded[17] = 1.0

        return encoded

    def __repr__(self):
        turn_str = "红" if self.turn else "黑"
        return f"GameState(turn={turn_str}, moves={self.move_count})"
