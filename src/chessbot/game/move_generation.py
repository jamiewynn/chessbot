from enum import Enum
from functools import cached_property, lru_cache
from typing import Iterable, List, Optional, Set, Iterator, Any
from abc import ABC, abstractmethod

from chessbot.game.board import RankAndFile, Colour, PieceType, Piece, PieceOnSquare, NUM_RANKS, Board
from chessbot.game.game_state import GameState, CastleType
from chessbot.game.move import Move, Promotion, Castle

import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


# TODO To avoid recalculating stuff like attacked squares!
# Will expose various member functions like is_check, is_checkmate, get_attacked_squares, etc etc
# Then higher-level functions like generate_moves can go elsewhere and this file can contain lower-level calculational
# stuff
CACHE_SIZES = 64


class GameResult(Enum):
    BLACK_WIN = 'BLACK_WINS'
    WHITE_WIN = 'WHITE_WIN'
    DRAW = 'DRAW'  # Including both stalemate and draw by agreement


class BoardCalculationCache:
    def __init__(self, state: GameState):
        self._state = state

    @property
    def state(self):
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
                if not BoardCalculationCache(child_state).is_check():
                    moves.append(move)
        return moves


class MoveGenerator(ABC):
    @abstractmethod
    def get_moves(self, original_square: RankAndFile, calculation_cache: BoardCalculationCache) -> Iterable[Move]:
        raise NotImplementedError

    @abstractmethod
    def get_attacks(self, square: RankAndFile, board: Board) -> Iterable[RankAndFile]:
        raise NotImplementedError


class PawnMoveGenerator(MoveGenerator):
    def get_moves(self, original_square: RankAndFile, calculation_cache: BoardCalculationCache) -> Iterable[Move]:
        state = calculation_cache.state
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


class KnightMoveGenerator(MoveGenerator):
    def __init__(self):
        self._knight_move_stencil = [
            RankAndFile(1, 2), RankAndFile(1, -2), RankAndFile(-1, 2), RankAndFile(-1, -2),
            RankAndFile(2, 1), RankAndFile(2, -1), RankAndFile(-2, 1), RankAndFile(-2, -1)
        ]

    def get_moves(self, original_square: RankAndFile, calculation_cache: BoardCalculationCache) -> Iterable[Move]:
        state = calculation_cache.state
        for attacked_square in self.get_attacks(original_square, state.board):
            if not _square_is_occupied_by_player(attacked_square, state.player_to_move, state):
                yield Move(original_square=original_square, target_square=attacked_square)

    def get_attacks(self, square: RankAndFile, board: Board) -> Iterable[RankAndFile]:
        for target_square_relative in self._knight_move_stencil:
            target_square = square + target_square_relative
            if target_square.is_in_board():
                yield target_square


class KingMoveGenerator(MoveGenerator):
    def __init__(self):
        self._king_move_stencil = [
            RankAndFile(0, 1), RankAndFile(0, -1), RankAndFile(1, 0), RankAndFile(-1, 0),
            RankAndFile(1, 1), RankAndFile(-1, -1), RankAndFile(1, -1), RankAndFile(-1, 1)
        ]

    def get_moves(self, original_square: RankAndFile, calculation_cache: BoardCalculationCache) -> Iterable[Move]:
        state = calculation_cache.state
        squares_attacked_by_opposing_side = calculation_cache.get_attacks(state.player_to_move.other())

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


class StraightLineMoveGenerator(MoveGenerator):
    def __init__(self, moves_diagonally: bool, moves_up_and_right: bool):
        self._move_directions: List[RankAndFile] = []
        if moves_diagonally:
            self._move_directions += [RankAndFile(1, 1), RankAndFile(1, -1), RankAndFile(-1, 1), RankAndFile(-1, -1)]
        if moves_up_and_right:
            self._move_directions += [RankAndFile(1, 0), RankAndFile(0, 1), RankAndFile(-1, 0), RankAndFile(0, -1)]

    def get_moves(self, original_square: RankAndFile, calculation_cache: BoardCalculationCache) -> Iterable[Move]:
        state = calculation_cache.state
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


PIECE_TYPE_TO_MOVE_GENERATOR = {
    PieceType.KNIGHT: KnightMoveGenerator(),
    PieceType.QUEEN: StraightLineMoveGenerator(moves_diagonally=True, moves_up_and_right=True),
    PieceType.ROOK: StraightLineMoveGenerator(moves_diagonally=False, moves_up_and_right=True),
    PieceType.BISHOP: StraightLineMoveGenerator(moves_diagonally=True, moves_up_and_right=False),
    PieceType.PAWN: PawnMoveGenerator(),
    PieceType.KING: KingMoveGenerator(),
}


def _square_is_occupied_by_player(square: RankAndFile, player: Colour, state: GameState) -> bool:
    target_square_contents = state.board[square]
    return target_square_contents is not None and target_square_contents.colour == player


def _is_valid_target_square(target_square: RankAndFile, player: Colour, state: GameState) -> bool:
    # Check if a square can in principle be moved to by the specified player, i.e. is on the board and is not occupied
    # by one of that player's own pieces.
    return target_square.is_in_board() and not _square_is_occupied_by_player(target_square, player, state)


def _walk_along_direction(start_square: RankAndFile, direction: RankAndFile) -> Iterable[RankAndFile]:
    """
    Walk along the board until the end is reached.
    :param start_square: Square to start walking from (will be included in output)
    :param direction: Direction to walk in - steps of this size will be taken
    :return: Iterable of squares reached before going off the end of the board
    """
    target_square = start_square
    while target_square.is_in_board():
        yield target_square
        target_square = target_square + direction


def _find_pieces_pinned_to_king(king_square: RankAndFile, state: GameState) -> Iterable[PieceOnSquare]:
    # NB Only rooks, queens and bishops can pin!
    move_directions = PIECE_TYPE_TO_MOVE_GENERATOR[PieceType.QUEEN].move_directions
    king_colour = state.board[king_square].colour

    def iter_until_hitting_piece(squares: Iterable[RankAndFile]) -> Optional[PieceOnSquare]:
        for square in squares:
            square_contents = state.board[square]
            if square_contents is not None:
                return PieceOnSquare(piece=square_contents, square=square)

    # Condition for a pin is that walking from the king in one of 8 possible directions intersects with a piece of the
    # same colour before intersecting a piece of the opposite colour that can attack along that direction.
    for direction in move_directions:
        squares_along_direction = iter(_walk_along_direction(king_square, direction))

        # Pinned piece must of same colour as king
        first_piece_hit = iter_until_hitting_piece(squares_along_direction)
        if first_piece_hit is None or first_piece_hit.piece.colour != king_colour:
            continue

        # Pinning piece must of opposite colour as king
        second_piece_hit = iter_until_hitting_piece(squares_along_direction)
        if second_piece_hit is None or second_piece_hit.piece.colour == king_colour:
            continue

        # Pinning piece must be capable of attacking along this direction
        # (minus sign because we walked from the king from the attacking piece)
        if -direction not in PIECE_TYPE_TO_MOVE_GENERATOR[second_piece_hit.piece.type].move_directions:
            continue

        # If we get this far then we have a pin, so return the pinned piece (the first one we hit)
        yield first_piece_hit


def _get_squares_attacked_by_side(board: Board, colour: Colour) -> Iterable[RankAndFile]:
    for square in board.get_occupied_squares():
        piece = board[square]
        if piece.colour != colour:
            continue

        attacks = PIECE_TYPE_TO_MOVE_GENERATOR[piece.type].get_attacks(square, board)
        for attack in attacks:
            yield attack


def _get_attacks_on_square(board: Board, target_square: RankAndFile, attacking_colour: Colour) -> Iterable[RankAndFile]:
    # NOTE: a lot of time is spent in this function!
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


def _taxicab_distance(lhs: RankAndFile, rhs: RankAndFile) -> int:
    return abs(lhs.rank - rhs.rank) + abs(lhs.file - rhs.file)


def _squares_are_knights_move_away(lhs: RankAndFile, rhs: RankAndFile) -> bool:
    return {abs(lhs.rank - rhs.rank), abs(lhs.file - rhs.file)} == {1, 2}


def _squares_same_rank_or_file(lhs: RankAndFile, rhs: RankAndFile) -> bool:
    return lhs.rank == rhs.rank or lhs.file == rhs.file


def _squares_same_diagonal(lhs: RankAndFile, rhs: RankAndFile) -> bool:
    # Checks whether one can get from lhs to rhs by travelling diagonally
    delta = rhs - lhs
    return abs(delta.rank) == abs(delta.file)


def _iterator_non_empty(x: Iterator[Any]) -> bool:
    # Helper fn to lazily check whether an iterator has any elements
    return any(True for _ in x)
