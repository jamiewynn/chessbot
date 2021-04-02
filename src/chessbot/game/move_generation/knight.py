from typing import Iterable

from chessbot.game.board import Board
from chessbot.game.helpers import _square_is_occupied_by_player
from chessbot.game.move import Move
from chessbot.game.move_generation.position_analyser_interface import PositionAnalyserInterface
from chessbot.game.move_generation.move_generator_interface import MoveGenerator
from chessbot.game.piece import RankAndFile


class KnightMoveGenerator(MoveGenerator):
    def __init__(self):
        self._knight_move_stencil = [
            RankAndFile(1, 2), RankAndFile(1, -2), RankAndFile(-1, 2), RankAndFile(-1, -2),
            RankAndFile(2, 1), RankAndFile(2, -1), RankAndFile(-2, 1), RankAndFile(-2, -1)
        ]

    def get_moves(self, original_square: RankAndFile, position_analyser: PositionAnalyserInterface) -> Iterable[Move]:
        state = position_analyser.state
        for attacked_square in self.get_attacks(original_square, state.board):
            if not _square_is_occupied_by_player(attacked_square, state.player_to_move, state):
                yield Move(original_square=original_square, target_square=attacked_square)

    def get_attacks(self, square: RankAndFile, board: Board) -> Iterable[RankAndFile]:
        for target_square_relative in self._knight_move_stencil:
            target_square = square + target_square_relative
            if target_square.is_in_board():
                yield target_square