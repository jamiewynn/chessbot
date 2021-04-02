from typing import Any, Iterable, Iterator

from chessbot.game.game_state import GameState
from chessbot.game.piece import Colour, RankAndFile


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


def _square_is_occupied_by_player(square: RankAndFile, player: Colour, state: GameState) -> bool:
    target_square_contents = state.board[square]
    return target_square_contents is not None and target_square_contents.colour == player


def _is_valid_target_square(target_square: RankAndFile, player: Colour, state: GameState) -> bool:
    # Check if a square can in principle be moved to by the specified player, i.e. is on the board and is not occupied
    # by one of that player's own pieces.
    return target_square.is_in_board() and not _square_is_occupied_by_player(target_square, player, state)


