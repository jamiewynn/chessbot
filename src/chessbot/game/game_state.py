from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Set

from chessbot.game.board import Board, RankAndFile, Colour


class CastleType(Enum):
    KINGSIDE = 'KINGSIDE'
    QUEENSIDE = 'QUEENSIDE'


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
