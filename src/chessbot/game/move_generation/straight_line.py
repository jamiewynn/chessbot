from functools import cached_property
from typing import List, Iterable, Set

from chessbot.game.board import Board
from chessbot.game.helpers import _walk_along_direction
from chessbot.game.move import Move
from chessbot.game.move_generation.position_analyser_interface import PositionAnalyserInterface
from chessbot.game.move_generation.move_generator_interface import MoveGenerator
from chessbot.game.piece import RankAndFile


class StraightLineMoveGenerator(MoveGenerator):
    """ MoveGenerator that is sufficiently generic to work for the queen, bishop, or rook. """
    def __init__(self, moves_diagonally: bool, moves_up_and_right: bool):
        self._move_directions: List[RankAndFile] = []
        if moves_diagonally:
            self._move_directions += [RankAndFile(1, 1), RankAndFile(1, -1), RankAndFile(-1, 1), RankAndFile(-1, -1)]
        if moves_up_and_right:
            self._move_directions += [RankAndFile(1, 0), RankAndFile(0, 1), RankAndFile(-1, 0), RankAndFile(0, -1)]

    def get_moves(self, original_square: RankAndFile, position_analyser: PositionAnalyserInterface) -> Iterable[Move]:
        state = position_analyser.state
        for move_direction in self._move_directions:
            attacks = list(self._get_attacks_in_direction(original_square, state.board, move_direction))

            # The final attacked square along each direction may be of the same colour as us - this square does not
            # form a valid square to move to
            if len(attacks) > 0:
                contents_of_last_square_along_path = state.board[attacks[-1]]
                if contents_of_last_square_along_path is not None \
                        and contents_of_last_square_along_path.colour == state.player_to_move:
                    attacks.pop(-1)

            yield from [Move(original_square=original_square, target_square=attack) for attack in attacks]

    def get_attacks(self, square: RankAndFile, board: Board) -> Iterable[RankAndFile]:
        for move_direction in self._move_directions:
            yield from self._get_attacks_in_direction(square, board, move_direction)

    @cached_property
    def move_directions(self) -> Set[RankAndFile]:
        return set(self._move_directions)

    @staticmethod
    def _get_attacks_in_direction(start_square: RankAndFile, board: Board,
                                  direction: RankAndFile) -> Iterable[RankAndFile]:
        squares_along_direction = iter(_walk_along_direction(start_square, direction))
        next(squares_along_direction)
        for target_square in squares_along_direction:
            # We are attacking this square regardless of its occupancy
            yield target_square

            # We can't step any further if this square is occupied (by either player)
            target_square_contents = board[target_square]
            if target_square_contents is not None:
                break