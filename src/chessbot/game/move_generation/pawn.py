from typing import Iterable

from chessbot.game.board import NUM_RANKS, Board
from chessbot.game.game_state import GameState
from chessbot.game.helpers import _square_is_occupied_by_player
from chessbot.game.move import Move, Promotion
from chessbot.game.move_generation.position_analyser_interface import PositionAnalyserInterface
from chessbot.game.move_generation.move_generator_interface import MoveGenerator
from chessbot.game.piece import RankAndFile, Colour, PieceType


class PawnMoveGenerator(MoveGenerator):
    def get_moves(self, original_square: RankAndFile, position_analyser: PositionAnalyserInterface) -> Iterable[Move]:
        state = position_analyser.state
        colour = state.board[original_square].colour
        colour_adjusted_rank = original_square.rank if colour == Colour.WHITE else NUM_RANKS-1 - original_square.rank

        moves = list(self._get_moves_ignoring_promotion(original_square, colour, state))

        # If the pawn is currently at the second to last rank, then all valid moves will take it to the end of the end
        # of the board and promotion is then obligatory
        if colour_adjusted_rank == NUM_RANKS - 2:
            for move in moves:
                for promotable_piece in [PieceType.KNIGHT, PieceType.ROOK, PieceType.BISHOP, PieceType.QUEEN]:
                    yield Promotion(original_square=move.original_square, target_square=move.target_square,
                                    new_piece_type=promotable_piece)
        else:
            yield from moves

    def _get_moves_ignoring_promotion(self, square: RankAndFile, colour: Colour, state: GameState) -> Iterable[Move]:
        # Check whether we can push (either a single or double push)
        forward_direction = RankAndFile(rank=1 if colour == Colour.WHITE else -1,
                                        file=0)
        colour_adjusted_rank = square.rank if colour == Colour.WHITE else NUM_RANKS-1 - square.rank
        target_square = square + forward_direction
        if target_square.is_in_board() and state.board[target_square] is None:
            yield Move(original_square=square, target_square=target_square)
            target_square = target_square + forward_direction

            # Double push is only permissible if we are on the starting rank!
            if target_square.is_in_board() and state.board[target_square] is None and colour_adjusted_rank == 1:
                yield Move(original_square=square, target_square=target_square)

        # Check whether captures are possible
        # Handle en passant special case
        en_passant_square = None
        if state.double_moved_pawn_square is not None:
            double_moved_pawn = state.board[state.double_moved_pawn_square]
            if double_moved_pawn.colour == Colour.WHITE:
                en_passant_square = state.double_moved_pawn_square - RankAndFile(rank=1, file=0)
            else:
                en_passant_square = state.double_moved_pawn_square + RankAndFile(rank=1, file=0)

        for attack in self.get_attacks(square, state.board):
            if _square_is_occupied_by_player(attack, state.player_to_move.other(), state):
                yield Move(original_square=square, target_square=attack)
            elif en_passant_square is not None and attack == en_passant_square:
                yield Move(original_square=square, target_square=attack)

    def get_attacks(self, square: RankAndFile, board: Board) -> Iterable[RankAndFile]:
        colour = board[square].colour

        forward_direction = RankAndFile(rank=1 if colour == Colour.WHITE else -1,
                                        file=0)
        forward_left = forward_direction + RankAndFile(rank=0, file=-1)
        forward_right = forward_direction + RankAndFile(rank=0, file=1)
        attacks = [forward_left, forward_right]
        for attack in attacks:
            target_square = square + attack
            if target_square.is_in_board():
                yield target_square