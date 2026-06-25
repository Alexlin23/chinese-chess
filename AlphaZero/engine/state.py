"""棋盘状态封装 — GameState 不可变快照

支持：
- 重复局面检测（三次重复平局）
- 最大步数限制
- WDL 结果输出
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .constants import (
    ROWS, COLS, KING, RED, BLACK,
    MAX_GAME_PLY, REPETITION_LIMIT, WDL_WIN, WDL_DRAW, WDL_LOSS
)
from .fast_chess import (
    create_initial_board, get_valid_moves, make_move, is_in_check
)
from .move import Move, ActionEncoder
from .repetition import position_key, update_repetition, is_repetition_draw


@dataclass(frozen=True)
class GameResult:
    """游戏结果"""
    winner: Optional[int]  # RED, BLACK, None(平局)
    reason: str            # king_captured | no_legal_move | max_ply | repetition

    @property
    def is_draw(self) -> bool:
        return self.winner is None

    def to_wdl(self, current_player: int) -> np.ndarray:
        """转换为当前玩家视角的 WDL 向量

        Args:
            current_player: RED 或 BLACK

        Returns:
            (3,) float32 [win, draw, loss]
        """
        if self.is_draw:
            return np.array([0, 1, 0], dtype=np.float32)
        elif self.winner == current_player:
            return np.array([1, 0, 0], dtype=np.float32)
        else:
            return np.array([0, 0, 1], dtype=np.float32)

    def to_value(self, current_player: int) -> float:
        """转换为当前玩家视角的标量值 [-1, +1]"""
        if self.is_draw:
            return 0.0
        elif self.winner == current_player:
            return 1.0
        else:
            return -1.0


class GameState:
    """不可变棋盘状态快照。

    内部持有:
      - board: numpy int8 (10,9)
      - turn: bool (True=红, False=黑)
      - move_count: int 已走步数
      - no_capture_count: int 连续无吃子步数
      - repetition_counts: dict 局面重复计数
      - last_move: Move | None
    """

    __slots__ = ('board', 'turn', 'move_count', 'no_capture_count',
                 'repetition_counts', 'last_move')

    def __init__(self, board: np.ndarray, turn,
                 move_count=0, no_capture_count=0,
                 repetition_counts=None, last_move=None):
        self.board = board
        if isinstance(turn, str):
            self.turn = (turn == "r")
        else:
            self.turn = bool(turn)
        self.move_count = move_count
        self.no_capture_count = no_capture_count
        self.repetition_counts = repetition_counts if repetition_counts is not None else {}
        self.last_move = last_move

    def copy(self) -> 'GameState':
        """浅拷贝（board 是 numpy 数组，apply 时会 copy）"""
        return GameState(
            self.board, self.turn, self.move_count,
            self.no_capture_count, self.repetition_counts, self.last_move
        )

    @classmethod
    def new_game(cls) -> 'GameState':
        """创建初始棋盘状态"""
        board = create_initial_board()
        key = position_key(board, True)
        counts = {key: 1}
        return cls(board, True, 0, 0, counts, None)

    # ── 走法查询 ──

    def legal_moves(self) -> list[Move]:
        """当前局面所有合法走法"""
        moves = []
        sign = 1 if self.turn else -1
        rows, cols = np.where(self.board * sign > 0)
        for r, c in zip(rows, cols):
            valid = get_valid_moves(self.board, int(r), int(c), self.turn)
            for m in valid:
                moves.append(Move(int(r), int(c), int(m["row"]), int(m["col"])))
        return moves

    def legal_actions(self) -> np.ndarray:
        """当前局面所有合法动作索引"""
        return ActionEncoder.legal_actions(self.legal_moves())

    def legal_mask(self) -> np.ndarray:
        """返回 (8100,) bool 合法动作掩码"""
        return ActionEncoder.legal_mask(self.legal_moves())

    def is_legal(self, move: Move) -> bool:
        """单步走法合法性校验"""
        from .fast_chess import is_valid_move
        return is_valid_move(
            self.board,
            (move.from_row, move.from_col),
            (move.to_row, move.to_col),
            self.turn,
        )

    # ── 走法执行 ──

    def apply(self, move: Move) -> 'GameState':
        """执行走法，返回新状态（不修改自身）"""
        action = ActionEncoder.encode(move)
        return self.apply_action(action)

    def apply_action(self, action: int) -> 'GameState':
        """执行动作，返回新状态

        Args:
            action: 0..8099 动作索引

        Returns:
            新的 GameState

        Raises:
            ValueError: 动作无效或非法
        """
        move = ActionEncoder.decode(action)
        if move is None:
            raise ValueError(f"Invalid action: {action}")
        if not self.is_legal(move):
            raise ValueError(f"Illegal move: {move}")

        new_board = self.board.copy()
        undo = make_move(new_board,
                         (move.from_row, move.from_col),
                         (move.to_row, move.to_col))
        captured = undo["captured"]

        new_turn = not self.turn
        new_move_count = self.move_count + 1
        new_no_capture = 0 if captured else self.no_capture_count + 1

        # 更新重复计数
        new_key = position_key(new_board, new_turn)
        new_counts = update_repetition(self.repetition_counts, new_key)

        return GameState(
            board=new_board,
            turn=new_turn,
            move_count=new_move_count,
            no_capture_count=new_no_capture,
            repetition_counts=new_counts,
            last_move=move,
        )

    # ── 终局检测 ──

    def game_result(self) -> Optional[GameResult]:
        """检测游戏结果

        Returns:
            None: 未结束
            GameResult: 红胜/黑胜/平局
        """
        # 1. 检查将/帅是否存在
        has_red_king = np.any(self.board == KING)
        has_black_king = np.any(self.board == -KING)

        if not has_red_king:
            return GameResult(BLACK, "king_captured")
        if not has_black_king:
            return GameResult(RED, "king_captured")

        # 2. 检查重复局面
        current_key = position_key(self.board, self.turn)
        if is_repetition_draw(self.repetition_counts, current_key, REPETITION_LIMIT):
            return GameResult(None, "repetition")

        # 3. 检查最大步数
        if self.move_count >= MAX_GAME_PLY:
            return GameResult(None, "max_ply")

        # 4. 检查无合法走法（困毙）
        if not self.has_any_valid_move():
            # 无合法走法 = 当前方负
            winner = BLACK if self.turn else RED
            return GameResult(winner, "no_legal_move")

        # 5. 检查自然限着（60步无吃子）
        if self.no_capture_count >= 60:
            return GameResult(None, "no_capture")

        return None

    def is_terminal(self) -> bool:
        """是否终局"""
        return self.game_result() is not None

    def result(self) -> Optional[float]:
        """终局结果: +1=红胜, -1=黑胜, 0=和棋, None=未终局"""
        r = self.game_result()
        if r is None:
            return None
        if r.is_draw:
            return 0.0
        return 1.0 if r.winner == RED else -1.0

    def result_for_player(self) -> Optional[float]:
        """当前玩家视角的结果: +1=胜, -1=负, 0=平, None=未终局"""
        r = self.game_result()
        if r is None:
            return None
        return r.to_value(self.turn)

    def wdl_for_player(self) -> Optional[np.ndarray]:
        """当前玩家视角的 WDL 向量"""
        r = self.game_result()
        if r is None:
            return None
        return r.to_wdl(self.turn)

    def has_any_valid_move(self) -> bool:
        """检查当前方是否有任何合法走法"""
        sign = 1 if self.turn else -1
        rows, cols = np.where(self.board * sign > 0)
        for r, c in zip(rows, cols):
            if get_valid_moves(self.board, int(r), int(c), self.turn):
                return True
        return False

    def is_in_check(self) -> bool:
        """当前方是否被将军"""
        return is_in_check(self.board, self.turn)

    # ── 编码 ──

    def encode(self) -> np.ndarray:
        """编码为神经网络输入 (18, 10, 9) float32。

        通道:
          0-6:  己方棋子位 (王..兵)
          7-13: 对方棋子位
          14:   己方颜色标记 (全1)
          15:   总步数 / MAX_GAME_PLY
          16:   无吃子步数 / 60
          17:   将军标记
        """
        from .constants import MAX_GAME_PLY
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
        encoded[15] = min(self.move_count, MAX_GAME_PLY) / MAX_GAME_PLY
        encoded[16] = min(self.no_capture_count, 60) / 60.0

        # 将军标记
        if is_in_check(self.board, self.turn):
            encoded[17] = 1.0

        return encoded

    def __repr__(self):
        turn_str = "红" if self.turn else "黑"
        return f"GameState(turn={turn_str}, moves={self.move_count})"
