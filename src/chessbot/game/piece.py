from __future__ import annotations

import string
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Tuple


class Colour(Enum):
    WHITE = 'WHITE'
    BLACK = 'BLACK'

    def other(self):
        return self.BLACK if self == self.WHITE else self.WHITE


class PieceType(IntEnum):
    KING = 1
    QUEEN = 2
    ROOK = 3
    BISHOP = 4
    KNIGHT = 5
    PAWN = 6

    @staticmethod
    def from_char(char):
        return _ASCII_CHAR_TO_PIECE_TYPE[char]


_PIECE_TYPE_TO_ASCII_CHAR = {
    PieceType.KING: 'K',
    PieceType.QUEEN: 'Q',
    PieceType.ROOK: 'R',
    PieceType.BISHOP: 'B',
    PieceType.KNIGHT: 'N',
    PieceType.PAWN: 'P'
}
_ASCII_CHAR_TO_PIECE_TYPE = {v: k for k, v in _PIECE_TYPE_TO_ASCII_CHAR.items()}
_UNICODE_PIECE_LOOKUP = {
    Colour.WHITE: {
        PieceType.KING: '♔',
        PieceType.QUEEN: '♕',
        PieceType.ROOK: '♖',
        PieceType.BISHOP: '♗',
        PieceType.KNIGHT: '♘',
        PieceType.PAWN: '♙'
    },
    Colour.BLACK: {
        PieceType.KING: '♚',
        PieceType.QUEEN: '♛',
        PieceType.ROOK: '♜',
        PieceType.BISHOP: '♝',
        PieceType.KNIGHT: '♞',
        PieceType.PAWN: '♟'
    }
}


@dataclass
class Piece:
    type: PieceType
    colour: Colour


@dataclass
class PieceOnSquare:
    piece: Piece
    square: RankAndFile


@dataclass
class RankAndFile:
    """
    Represents a chessboard position. The convention adopted here is that the bottom-left corner is rank=0, file=0.
    """
    rank: int
    file: int

    def is_in_board(self) -> bool:
        return 0 <= self.rank <= 7 and 0 <= self.file <= 7

    def as_tuple(self) -> Tuple[int, int]:
        return self.rank, self.file

    @classmethod
    def from_algebraic(cls, algebraic_pos: str):
        """
        Construct from standard algebraic notation for board positions.
        :param algebraic_pos: Algebraic position, e.g. 'e6', 'h8', etc.
        :return: Corresponding RankAndFile.
        """
        file, rank = algebraic_pos
        return cls(file=string.ascii_lowercase.index(file), rank=int(rank)-1)

    @classmethod
    def from_tuple(cls, rank_and_file: Tuple[int, int]):
        rank, file = rank_and_file
        return cls(rank=rank, file=file)

    def __add__(self, other: RankAndFile) -> RankAndFile:
        return RankAndFile(rank=self.rank + other.rank,
                           file=self.file + other.file)

    def __sub__(self, other: RankAndFile) -> RankAndFile:
        return RankAndFile(rank=self.rank - other.rank,
                           file=self.file - other.file)

    def __neg__(self):
        return RankAndFile(rank=-self.rank, file=-self.file)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'{string.ascii_lowercase[self.file]}{self.rank+1}'

    def __hash__(self):
        return hash(self.as_tuple())