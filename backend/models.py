"""Pydantic 数据模型"""
from pydantic import BaseModel, Field
from typing import Optional, Literal
from AlphaZero.train.config import RULE_VERSION


class Position(BaseModel):
    row: int = Field(ge=0, le=9, strict=True)
    col: int = Field(ge=0, le=8, strict=True)


class PieceInfo(BaseModel):
    type: str
    color: Literal["r", "b"]


class RuleStateRequest(BaseModel):
    board: list[list[Optional[PieceInfo]]]
    turn: Literal["r", "b"]
    game_id: int = Field(ge=1, strict=True)
    revision: int = Field(ge=0, strict=True)


class ValidMovesRequest(RuleStateRequest):
    row: int = Field(ge=0, le=9, strict=True)
    col: int = Field(ge=0, le=8, strict=True)


class MoveRequest(RuleStateRequest):
    from_pos: Position
    to_pos: Position


class RuleMetadata(BaseModel):
    revision: int = 0
    no_capture_count: int = 0
    natural_limit: dict = Field(default_factory=dict)
    rule_reason: Optional[str] = None
    rule_version: str = RULE_VERSION


class MoveResponse(RuleMetadata):
    valid: bool
    captured: Optional[PieceInfo] = None
    check: bool = False
    game_over: Optional[str] = None
    new_board: Optional[list[list[Optional[PieceInfo]]]] = None
    turn: Optional[str] = None


class GameCreateResponse(RuleMetadata):
    game_id: int
    board: list[list[Optional[PieceInfo]]]
    turn: str


class CheckWinRequest(RuleStateRequest):
    pass


class UndoRequest(BaseModel):
    revision: int = Field(ge=0, strict=True)


class GameInfoResponse(RuleMetadata):
    game_id: int
    board: list[list[Optional[PieceInfo]]]
    turn: str
    status: str
    step_count: int = 0
