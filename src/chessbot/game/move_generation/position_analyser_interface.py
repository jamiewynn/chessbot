from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional

from chessbot.game.game_state import GameState
from chessbot.game.move import Move
from chessbot.game.piece import Colour, RankAndFile, Piece


class GameResult(Enum):
    BLACK_WIN = 'BLACK_WINS'
    WHITE_WIN = 'WHITE_WINS'
    DRAW = 'DRAW'  # Including both stalemate and draw by agreement


class PositionAnalyserInterface(ABC):
    """
    NB this separate interface class exists solely to break an otherwise circular import between the move generator
    classes and the PositionAnalyserInterface.
    """
    @property
    @abstractmethod
    def state(self) -> GameState:
        pass

    @abstractmethod
    def get_attacks(self, attacking_side: Colour) -> List[RankAndFile]:
        pass

    @abstractmethod
    def locate_pieces(self, piece: Piece) -> List[RankAndFile]:
        pass

    @abstractmethod
    def get_game_result(self) -> Optional[GameResult]:
        pass

    @abstractmethod
    def is_check(self) -> bool:
        pass

    @abstractmethod
    def is_checkmate(self) -> bool:
        pass

    @abstractmethod
    def is_stalemate(self) -> bool:
        pass

    @abstractmethod
    def validate_move(self, move: Move) -> bool:
        pass

    @abstractmethod
    def get_valid_moves(self) -> List[Move]:
        pass
