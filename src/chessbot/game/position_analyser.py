import logging
from functools import lru_cache
from typing import Iterable, List, Optional

from chessbot.game.board import Board
from chessbot.game.game_state import GameState
from chessbot.game.helpers import (_iterator_non_empty,
                                   _squares_same_diagonal, _squares_same_rank_or_file,
                                   _squares_are_knights_move_away)
from chessbot.game.move import Move
from chessbot.game.move_generation.king import KingMoveGenerator
from chessbot.game.move_generation.knight import KnightMoveGenerator
from chessbot.game.move_generation.pawn import PawnMoveGenerator
from chessbot.game.move_generation.straight_line import StraightLineMoveGenerator
from chessbot.game.piece import Colour, Piece, PieceType, RankAndFile
from chessbot.game.move_generation.position_analyser_interface import GameResult, PositionAnalyserInterface

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


PIECE_TYPE_TO_MOVE_GENERATOR = {
    PieceType.KNIGHT: KnightMoveGenerator(),
    PieceType.QUEEN: StraightLineMoveGenerator(moves_diagonally=True, moves_up_and_right=True),
    PieceType.ROOK: StraightLineMoveGenerator(moves_diagonally=False, moves_up_and_right=True),
    PieceType.BISHOP: StraightLineMoveGenerator(moves_diagonally=True, moves_up_and_right=False),
    PieceType.PAWN: PawnMoveGenerator(),
    PieceType.KING: KingMoveGenerator(),
}


class PositionAnalyser(PositionAnalyserInterface):
    """
    Computes properties of interest for a given state, and caches them since often these properties are needed multiple
    times. NB this isn't for evaluating how good a position is via heuristics - all that is in the 'engine' submodule.
    """

    def __init__(self, state: GameState):
        self._state = state

    @property
    def state(self) -> GameState:
        return self._state

    @lru_cache
    def get_attacks(self, attacking_side: Colour) -> List[RankAndFile]:
        return list(_get_squares_attacked_by_side(self._state.board, attacking_side))

    @lru_cache
    def locate_pieces(self, piece: Piece) -> List[RankAndFile]:
        return list(self._state.board.locate_piece(piece))

    @lru_cache
    def get_game_result(self) -> Optional[GameResult]:
        if self.is_stalemate():
            return GameResult.DRAW
        if self.is_checkmate():
            return GameResult.WHITE_WIN if self._state.player_to_move == Colour.BLACK else GameResult.BLACK_WIN

    @lru_cache
    def is_check(self):
        try:
            king_location = next(iter(self._state.board.locate_piece(Piece(PieceType.KING,
                                                                           self._state.player_to_move))))
        except StopIteration:
            # Should never hit this in ordinary play since it signifies there is no king, but could in principle set the
            # board up that way from the start
            _logger.warning('Missing king!')
            return False

        attacks = _get_attacks_on_square(self._state.board, king_location, self._state.player_to_move.other())
        return _iterator_non_empty(iter(attacks))

    def is_checkmate(self):
        return len(self.get_valid_moves()) == 0 and self.is_check()

    def is_stalemate(self):
        return len(self.get_valid_moves()) == 0 and not self.is_check()

    def validate_move(self, move: Move) -> bool:
        return move in self.get_valid_moves()

    @lru_cache
    def get_valid_moves(self) -> List[Move]:
        moves = []
        for square in self._state.board.get_occupied_squares():
            piece = self._state.board[square]
            if piece.colour != self._state.player_to_move:
                continue

            move_generator = PIECE_TYPE_TO_MOVE_GENERATOR[piece.type]
            candidate_moves = move_generator.get_moves(square, self)

            # Only moves that do not leave the king in check are permitted.
            # Check this by executing the move and then flipping the player to move, so that the right king is tested
            # for check
            # TODO: only moves whose starting or ending squares lie on the same diagonal or rank or file or a knight's
            #  move from the king could possibly affect check status! Can use this to speed things up...
            for move in candidate_moves:
                child_state = move.execute(self._state)
                child_state.player_to_move = self._state.player_to_move
                if not PositionAnalyser(child_state).is_check():
                    moves.append(move)
        return moves


def _get_squares_attacked_by_side(board: Board, colour: Colour) -> Iterable[RankAndFile]:
    """
    Get all attacks by a specified side.
    :param board: Board to get attacks for
    :param colour: Side to get attacks by
    :return: Iterable of squares that the specified side is attacking. Squares will be returned once per attack, so the
    same square may appear multiple times.
    """
    for square in board.get_occupied_squares():
        piece = board[square]
        if piece.colour != colour:
            continue

        attacks = PIECE_TYPE_TO_MOVE_GENERATOR[piece.type].get_attacks(square, board)
        for attack in attacks:
            yield attack


def _get_attacks_on_square(board: Board, target_square: RankAndFile, attacking_colour: Colour) -> Iterable[RankAndFile]:
    """
    Get all attacks on a particular square.
    :param board: Board to get attacks for
    :param target_square: Square to get attacks on
    :param attacking_colour: Colour to get attacks by
    :return: Iterable of squares on which there are pieces attacking the specified square.
    """
    # NB a lot of time is spent in this function!
    for square in board.get_occupied_squares():
        piece = board[square]
        if piece.colour != attacking_colour:
            continue

        same_diagonal = _squares_same_diagonal(square, target_square)
        same_rank_or_file = _squares_same_rank_or_file(square, target_square)

        # Pre-filters for a few pieces to avoid unnecessarily computing attacks
        if piece.type == PieceType.QUEEN:
            if not same_diagonal and not same_rank_or_file:
                continue
        elif piece.type == PieceType.BISHOP:
            if not same_diagonal:
                continue
        elif piece.type == PieceType.ROOK:
            if not same_rank_or_file:
                continue
        elif piece.type == PieceType.KNIGHT:
            if not _squares_are_knights_move_away(square, target_square):
                continue
        elif piece.type in (PieceType.KING, PieceType.PAWN):
            if max([square.rank - target_square.rank, square.file - target_square.file]) > 1:
                continue

        # Compute all squares attacked by this piece and return the square that the piece is on if it turns out
        # to be attacking the target square
        attacks = PIECE_TYPE_TO_MOVE_GENERATOR[piece.type].get_attacks(square, board)
        if target_square in attacks:
            yield square
