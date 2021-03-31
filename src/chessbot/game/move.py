from __future__ import annotations
import copy
from dataclasses import dataclass

from chessbot.game.board import PieceType, RankAndFile, Piece, Colour
from chessbot.game.game_state import GameState, CastleType

QUEENSIDE_ROOK_STARTING_FILE = 0
KINGSIDE_ROOK_STARTING_FILE = 7


@dataclass
class Move:
    original_square: RankAndFile
    target_square: RankAndFile

    def execute(self, state: GameState) -> GameState:
        state = copy.deepcopy(state)
        piece = state.board[self.original_square]
        assert piece is not None
        target_square_contents = state.board[self.target_square]
        assert target_square_contents is None or target_square_contents.colour == state.player_to_move.other()

        # Check if the move is a pawn double move, in which case en passant is legal next turn
        if abs(self.target_square.rank - self.original_square.rank) == 2 and piece.type == PieceType.PAWN:
            #en_passant_square_rank = (self.original_square.rank + self.target_square.rank) // 2
            state.double_moved_pawn_square = self.target_square
        else:
            state.double_moved_pawn_square = None

        # Check if the move is a king move, in which case all castling rights are lost
        if piece.type == PieceType.KING:
            state.castling_rights[state.player_to_move] = set()

        # Check if the move is a rook move, in which case castling rights for the associated side are lost
        queenside_rook_starting_file = 7
        kingside_rook_starting_file = 0
        back_rank = 0 if state.player_to_move == Colour.WHITE else 7
        if piece.type == PieceType.ROOK:
            if self.original_square == RankAndFile(rank=back_rank, file=queenside_rook_starting_file):
                state.castling_rights[state.player_to_move].discard(CastleType.QUEENSIDE)
            elif self.original_square == RankAndFile(rank=back_rank, file=kingside_rook_starting_file):
                state.castling_rights[state.player_to_move].discard(CastleType.KINGSIDE)

        state.player_to_move = state.player_to_move.other()
        state.board[self.original_square] = None
        state.board[self.target_square] = piece
        return state

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

