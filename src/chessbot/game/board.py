from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

from chessbot.game.piece import (_UNICODE_PIECE_LOOKUP, Colour, Piece,
                                 PieceType, RankAndFile)


def make_coloured_char(piece: Optional[Piece], square: RankAndFile):
    dark_square = (square.rank + square.file) % 2 == 0
    background_colour = 47 if dark_square else 40
    if piece is not None:
        foreground_colour = 34 if piece.colour == Colour.WHITE else 31
        piece_char = _UNICODE_PIECE_LOOKUP[Colour.BLACK][piece.type] + ' '
    else:
        foreground_colour = 30
        piece_char = '  '
    colour_code = f'\033[{background_colour};{foreground_colour}m{piece_char}'
    return colour_code + '\033[0m'


NUM_RANKS = 8
NUM_FILES = 8


class Board:
    """
    Represents the locations of the pieces on the board (with none of the associated state about whose turn it is etc.)
    Uses a numpy array as internal representation for efficiency.
    """
    def __init__(self, board: np.array):
        assert board.shape == (8, 8)
        board = board.astype(np.int8)
        self._board = board

    @classmethod
    def make_empty(cls):
        board = np.zeros((NUM_RANKS, NUM_FILES), dtype=np.int8)
        return cls(board)

    @classmethod
    def standard_starting_position(cls):
        board = np.zeros((NUM_RANKS, NUM_FILES), dtype=np.int8)
        piece_row_initial = [
            PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN,
            PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK
        ]
        board[0, :] = piece_row_initial.copy()
        board[1, :] = PieceType.PAWN

        board[-1, :] = piece_row_initial.copy()
        board[-1, :] *= -1
        board[-2, :] = PieceType.PAWN
        board[-2, :] *= -1
        return cls(board)

    def __getitem__(self, square: RankAndFile) -> Optional[Piece]:
        assert square.is_in_board()
        return self._int_to_piece(self._board[square.as_tuple()])

    def __setitem__(self, square: RankAndFile, piece: Optional[Piece]) -> None:
        self._board[square.as_tuple()] = self._piece_to_int(piece)

    def __str__(self):
        board_str = ''
        for rank in reversed(range(NUM_RANKS)):
            board_str += f'{rank+1} '
            for file in range(NUM_FILES):
                square = RankAndFile(rank=rank, file=file)
                piece = self[square]
                board_str += make_coloured_char(piece, square)
            board_str += '\n'
        board_str += '  ＡＢＣＤＥＦＧＨ'

        return board_str

    def get_occupied_squares(self) -> Iterable[RankAndFile]:
        occupied_squares = np.where(self._board != 0)
        for square in zip(*occupied_squares):
            yield RankAndFile.from_tuple(square)

    def locate_piece(self, piece: Piece) -> List[RankAndFile]:
        piece_locations = np.where(self._board == self._piece_to_int(piece))
        for square in zip(*piece_locations):
            yield RankAndFile.from_tuple(square)

    @staticmethod
    def _piece_to_int(piece: Optional[Piece]) -> int:
        if piece is None:
            return 0
        piece_idx = piece.type.value
        return piece_idx if piece.colour == Colour.WHITE else -piece_idx

    @staticmethod
    def _int_to_piece(piece_int: int) -> Optional[Piece]:
        if piece_int == 0:
            return None
        piece_type = PieceType(abs(piece_int))
        colour = Colour.WHITE if piece_int > 0 else Colour.BLACK
        return Piece(type=piece_type, colour=colour)


