"""Pydantic 数据模型"""
from pydantic import BaseModel
from typing import Optional


class Position(BaseModel):
    row: int
    col: int


class PieceInfo(BaseModel):
    type: str
    color: str


class BoardState(BaseModel):
    board: list[list[Optional[PieceInfo]]]
    turn: str  # "r" or "b"


class ValidMovesRequest(BaseModel):
    row: int
    col: int
    board: list[list[Optional[PieceInfo]]]
    turn: str


class MoveRequest(BaseModel):
    from_pos: Position
    to_pos: Position
    board: list[list[Optional[PieceInfo]]]
    turn: str


class MoveResponse(BaseModel):
    valid: bool
    captured: Optional[PieceInfo] = None
    check: bool = False
    game_over: Optional[str] = None  # "red_win" / "black_win" / None
    new_board: Optional[list[list[Optional[PieceInfo]]]] = None


class GameCreateResponse(BaseModel):
    game_id: int
    board: list[list[Optional[PieceInfo]]]
    turn: str


class GameInfoResponse(BaseModel):
    game_id: int
    board: list[list[Optional[PieceInfo]]]
    turn: str
    status: str
