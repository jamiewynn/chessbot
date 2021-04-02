from typing import Iterable

from chessbot.game.board import Board
from chessbot.game.game_state import CastleType
from chessbot.game.helpers import _square_is_occupied_by_player
from chessbot.game.move import Move, Castle
from chessbot.game.move_generation.position_analyser_interface import PositionAnalyserInterface
from chessbot.game.move_generation.move_generator_interface import MoveGenerator
from chessbot.game.piece import RankAndFile


class KingMoveGenerator(MoveGenerator):
    def __init__(self):
        self._king_move_stencil = [
            RankAndFile(0, 1), RankAndFile(0, -1), RankAndFile(1, 0), RankAndFile(-1, 0),
            RankAndFile(1, 1), RankAndFile(-1, -1), RankAndFile(1, -1), RankAndFile(-1, 1)
        ]

    def get_moves(self, original_square: RankAndFile, position_analyser: PositionAnalyserInterface) -> Iterable[Move]:
        state = position_analyser.state
        squares_attacked_by_opposing_side = position_analyser.get_attacks(state.player_to_move.other())

        # Normal moves
        for attack in self.get_attacks(original_square, state.board):
            if not _square_is_occupied_by_player(attack, state.player_to_move, state):
                if attack not in squares_attacked_by_opposing_side:
                    yield Move(original_square=original_square, target_square=attack)

        # Castling
        # Verify preconditions for legality:
        # 1) the king is not in check
        if original_square in squares_attacked_by_opposing_side:
            return

        # 2) castling rights for the appropriate side are intact
        for castle_type in state.castling_rights[state.player_to_move]:
            # 3) castling would not put the king in check or through check.
            # The king slides along the below files during this move, and the corresponding squares must not be attacked
            # or castling is illegal:
            files_touched_by_king = [5, 6] if castle_type == CastleType.KINGSIDE else [3, 2]
            squares_touched_by_king = [RankAndFile(rank=original_square.rank, file=file)
                                       for file in files_touched_by_king]

            if any(touched_square in squares_attacked_by_opposing_side for touched_square in squares_touched_by_king):
                continue

            # 4) None of the squares passed through by either king or rook may be occupied (by either side)
            files_touched_by_king_or_rook = [5, 6] if castle_type == CastleType.KINGSIDE else [1, 2, 3]
            squares_touched_by_king_or_rook = [
                RankAndFile(rank=original_square.rank, file=file) for file in files_touched_by_king_or_rook
            ]

            if any(state.board[touched_square] is not None for touched_square in squares_touched_by_king_or_rook):
                continue

            # If we get this far then this castling move is valid
            yield Castle.make(colour=state.player_to_move, castle_type=castle_type)

    def get_attacks(self, square: RankAndFile, board: Board) -> Iterable[RankAndFile]:
        for target_square_relative in self._king_move_stencil:
            target_square = square + target_square_relative
            if target_square.is_in_board():
                yield target_square