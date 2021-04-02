from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Set

from chessbot.game.board import Board
from chessbot.game.piece import Colour, RankAndFile


class CastleType(Enum):
    KINGSIDE = 'KINGSIDE'
    QUEENSIDE = 'QUEENSIDE'


@dataclass
class GameState:
    """
    Represents the current game state at any given time, including both the board itself and other 'metadata' like
    castling rights.
    """
    board: Board

    # Player to make the next move from this position
    player_to_move: Colour

    # If a pawn has just made a double move, this will be the square it ended up on; otherwise it will be empty
    double_moved_pawn_square: Optional[RankAndFile]

    # Dictionary from player type to the set of castling moves that are still permitted
    castling_rights: Dict[Colour, Set[CastleType]]

    @classmethod
    def make_initial(cls):
        """
        :return: The starting state for a standard game of chess
        """
        castling_rights = {Colour.WHITE: {CastleType.KINGSIDE, CastleType.QUEENSIDE},
                           Colour.BLACK: {CastleType.KINGSIDE, CastleType.QUEENSIDE}}
        return GameState(board=Board.standard_starting_position(), player_to_move=Colour.WHITE,
                         double_moved_pawn_square=None, castling_rights=castling_rights)
