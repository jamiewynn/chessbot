from __future__ import annotations

import copy
from dataclasses import dataclass

from chessbot.game.game_state import CastleType, GameState
from chessbot.game.piece import Colour, Piece, PieceType, RankAndFile

QUEENSIDE_ROOK_STARTING_FILE = 0
KINGSIDE_ROOK_STARTING_FILE = 7


@dataclass
class Move:
    """
    Represents a single move. Instances of this base class corresponding to ordinary moves or captures; there are
    special subclasses below for castling and promotion.
    """
    original_square: RankAndFile
    target_square: RankAndFile

    def execute(self, state: GameState) -> GameState:
        """
        Execute this move. The validity of the move is not verified here except for some simple checks.
        :param state: State to execute the move from.
        :return: New state resulting by making this move.
        """
        state = copy.deepcopy(state)
        piece = state.board[self.original_square]
        assert piece is not None
        target_square_contents = state.board[self.target_square]
        assert target_square_contents is None or target_square_contents.colour == state.player_to_move.other()

        # Delete the double moved pawn if the move is an en passant capture
        if self._is_en_passant(state):
            assert state.board[self.target_square] is None
            assert state.double_moved_pawn_square is not None
            state.board[state.double_moved_pawn_square] = None

        # Check if the move is a pawn double move, in which case en passant is legal next turn
        if abs(self.target_square.rank - self.original_square.rank) == 2 and piece.type == PieceType.PAWN:
            state.double_moved_pawn_square = self.target_square
        else:
            state.double_moved_pawn_square = None

        # Check if the move is a king move, in which case all castling rights are lost
        if piece.type == PieceType.KING:
            state.castling_rights[state.player_to_move] = set()

        # Check if the move is a rook move, in which case castling rights for the associated side are lost
        back_rank = 0 if state.player_to_move == Colour.WHITE else 7
        if piece.type == PieceType.ROOK:
            if self.original_square == RankAndFile(rank=back_rank, file=QUEENSIDE_ROOK_STARTING_FILE):
                state.castling_rights[state.player_to_move].discard(CastleType.QUEENSIDE)
            elif self.original_square == RankAndFile(rank=back_rank, file=KINGSIDE_ROOK_STARTING_FILE):
                state.castling_rights[state.player_to_move].discard(CastleType.KINGSIDE)

        state.player_to_move = state.player_to_move.other()
        state.board[self.original_square] = None
        state.board[self.target_square] = piece
        return state

    def _is_en_passant(self, state: GameState) -> bool:
        # Check if the move is an en passant capture, in which case we have to delete the pawn from the square it ended
        # up on.
        target_square_contents = state.board[self.target_square]
        piece = state.board[self.original_square]

        # Can tell if move is en passant by verifying that:
        # 1) The move is a pawn move
        if piece.type != PieceType.PAWN:
            return False
        # 2) The move is diagonal
        if self.target_square.file == self.original_square.file:
            return False
        # 3) The target square is empty
        if target_square_contents is not None:
            return False
        return True

    def __hash__(self):
        return hash((self.original_square, self.target_square))


@dataclass
class Promotion(Move):
    new_piece_type: PieceType

    def execute(self, state: GameState) -> GameState:
        player_to_move = state.player_to_move
        state = super().execute(state)
        state.board[self.target_square] = Piece(type=self.new_piece_type, colour=player_to_move)
        return state

    def __hash__(self):
        return hash((self.original_square, self.target_square, self.new_piece_type))


@dataclass
class Castle(Move):
    castling_type: CastleType

    @classmethod
    def make(cls, colour: Colour, castle_type: CastleType):
        back_rank = 0 if colour == Colour.WHITE else 7
        king_start_square = RankAndFile(rank=back_rank, file=4)

        final_file = 6 if castle_type == castle_type.KINGSIDE else 2
        king_end_square = RankAndFile(rank=back_rank, file=final_file)
        return cls(original_square=king_start_square, target_square=king_end_square, castling_type=castle_type)

    def execute(self, state: GameState) -> GameState:
        state = copy.deepcopy(state)
        state.castling_rights[state.player_to_move] = set()
        back_rank = 0 if state.player_to_move == Colour.WHITE else 7

        # Also need to move the rook!
        if self.castling_type == CastleType.KINGSIDE:
            rook_start_position = RankAndFile(rank=back_rank, file=KINGSIDE_ROOK_STARTING_FILE)
            rook_end_position = RankAndFile(rank=back_rank, file=5)
        else:
            rook_start_position = RankAndFile(rank=back_rank, file=QUEENSIDE_ROOK_STARTING_FILE)
            rook_end_position = RankAndFile(rank=back_rank, file=3)

        assert state.board[rook_start_position].colour == state.player_to_move
        assert state.board[rook_end_position] is None
        state.board[rook_start_position] = None
        state.board[rook_end_position] = Piece(type=PieceType.ROOK, colour=state.player_to_move)

        state = super().execute(state)

        return state

    def __hash__(self):
        return hash((self.original_square, self.target_square, self.castling_type))

