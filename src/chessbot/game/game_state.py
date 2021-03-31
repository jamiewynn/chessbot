from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Set

from chessbot.game.board import Board, RankAndFile, Colour


#@dataclass
#class Piece:
#    # NB here 'piece' is used in a more generic sense to include pawns, whereas usually in chess terminology pawns are
#    # not considered pieces.
#    pass
#
#    def generate_moves(self, square: Square, state: GameState) -> Iterable[Move]:
#        raise NotImplementedError
#
#
#@dataclass
#class King(Piece):
#    has_moved: bool
#    has_castled: bool
#
#
#
#@dataclass
#class Rook(Piece):
#    has_moved: bool
#
#
#@dataclass
#class Queen(Piece):
#    pass
#
#
#@dataclass
#class Bishop(Piece):
#    pass
#
#
#@dataclass
#class Knight(Piece):
#    pass
#
#
#@dataclass
#class Pawn(Piece):
#    # Pawns that have just moved forward 2 spaces can be captured en passant, so we need to track this state
#    has_just_double_moved: bool


class CastleType(Enum):
    KINGSIDE = 'KINGSIDE'
    QUEENSIDE = 'QUEENSIDE'


#@dataclass
#class Board:
#    """
#    Represents the locations of the pieces on the board (with none of the associated state about whose turn it is etc.)
#    """
#    pieces: Dict[Location, Piece]
#
#    def


@dataclass
class GameState:
    board: Board
    player_to_move: Colour
    double_moved_pawn_square: Optional[RankAndFile]
    castling_rights: Dict[Colour, Set[CastleType]]

    @classmethod
    def make_initial(cls):
        castling_rights = {Colour.WHITE: {CastleType.KINGSIDE, CastleType.QUEENSIDE},
                           Colour.BLACK: {CastleType.KINGSIDE, CastleType.QUEENSIDE}}
        return GameState(board=Board.standard_starting_position(), player_to_move=Colour.WHITE,
                         double_moved_pawn_square=None, castling_rights=castling_rights)
