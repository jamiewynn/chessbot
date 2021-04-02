from abc import ABC, abstractmethod
from typing import Iterable

from chessbot.game.board import Board
from chessbot.game.move import Move
from chessbot.game.move_generation.position_analyser_interface import PositionAnalyserInterface
from chessbot.game.piece import RankAndFile


class MoveGenerator(ABC):
    """
    Interface for a class that can calculate moves and attacks by a given piece.
    In practice this class is implemented for the different possible movement patterns, and the implementations are
    stored in the lookup PIECE_TYPE_TO_MOVE_GENERATOR.
    """

    @abstractmethod
    def get_moves(self, original_square: RankAndFile, position_analyser: PositionAnalyserInterface) -> Iterable[Move]:
        """
        Get all valid moves for a particular piece.
        :param original_square: Square that the piece is currently on.
        :param position_analyser: Calculation cache for the game state of interest.
        :return: Valid moves for this piece. NB this doesn't have to filter out moves that would be illegal due to
        leaving the player in check - those are filtered out at a higher level when
        BoardCalculationCache.get_valid_moves is called.
        """
        raise NotImplementedError

    @abstractmethod
    def get_attacks(self, square: RankAndFile, board: Board) -> Iterable[RankAndFile]:
        """
        Get all attacks by a given piece.
        :param square: Square that the piece is currently on.
        :param board: Board that the piece is on.
        :return: Squares attacked by the specified piece. Note that this is defined to include squares occupied by
        pieces of the same colour (i.e. pieces that this piece is 'defending'). The piece is not considered to be
        'attacking' its own square.
        """
        raise NotImplementedError
