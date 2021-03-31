
""" Tests to verify that the rules of chess have been correctly implemented in the move generation code """
import unittest
from chessbot.game.board import Board, RankAndFile, PieceType, Piece, Colour
from chessbot.game.game_state import GameState
from chessbot.game.move import Move, Promotion
from chessbot.game.move_generation import BoardCalculationCache, GameResult


class MoveGenerationTests(unittest.TestCase):
    @staticmethod
    def make_game_state(board: Board, player_to_move: Colour):
        return GameState(board=board, player_to_move=player_to_move, double_moved_pawn_square=None,
                         castling_rights={Colour.WHITE: set(), Colour.BLACK: set()})

    def test_piece_pinned_to_king_cannot_escape_pin(self):
        # Arrange
        board = Board.make_empty()
        board[RankAndFile.from_algebraic('e1')] = Piece(type=PieceType.ROOK, colour=Colour.WHITE)
        board[RankAndFile.from_algebraic('d1')] = Piece(type=PieceType.ROOK, colour=Colour.WHITE)
        board[RankAndFile.from_algebraic('f1')] = Piece(type=PieceType.ROOK, colour=Colour.WHITE)
        board[RankAndFile.from_algebraic('a7')] = Piece(type=PieceType.ROOK, colour=Colour.WHITE)
        board[RankAndFile.from_algebraic('e8')] = Piece(type=PieceType.KING, colour=Colour.BLACK)
        board[RankAndFile.from_algebraic('e4')] = Piece(type=PieceType.QUEEN, colour=Colour.BLACK)
        state = self.make_game_state(board=board, player_to_move=Colour.BLACK)
        # Act
        valid_moves = BoardCalculationCache(state).get_valid_moves()

        # Assert
        # Only valid queen moves are those where the queen moves up or down the file
        valid_moves_expected = {
            Move(original_square=RankAndFile.from_algebraic('e4'), target_square=RankAndFile.from_algebraic(f'e{rank}'))
            for rank in range(1, 8)
            if rank != 4
        }
        print(state.board)
        self.assertSetEqual(set(valid_moves), valid_moves_expected)

    def test_promotions_handled_correctly(self):
        # Arrange
        board = Board.make_empty()
        board[RankAndFile.from_algebraic('b2')] = Piece(type=PieceType.PAWN, colour=Colour.BLACK)
        board[RankAndFile.from_algebraic('a1')] = Piece(type=PieceType.BISHOP, colour=Colour.WHITE)
        state = self.make_game_state(board=board, player_to_move=Colour.BLACK)

        # Act
        calculation_cache = BoardCalculationCache(state)
        valid_moves = calculation_cache.get_valid_moves()

        # Assert
        # Pawn can promote either by moving forward once or by capturing the bishop
        valid_moves_expected = {
            Promotion(original_square=RankAndFile.from_algebraic('b2'), target_square=target_square,
                      new_piece_type=new_piece_type)
            for target_square in [RankAndFile.from_algebraic('a1'), RankAndFile.from_algebraic('b1')]
            for new_piece_type in [PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN]
        }
        self.assertSetEqual(set(valid_moves), valid_moves_expected)

    def test_en_passant_capture_permitted(self):
        # Arrange
        board = Board.make_empty()
        board[RankAndFile.from_algebraic('d5')] = Piece(type=PieceType.PAWN, colour=Colour.WHITE)
        board[RankAndFile.from_algebraic('e7')] = Piece(type=PieceType.PAWN, colour=Colour.BLACK)
        state = self.make_game_state(board=board, player_to_move=Colour.BLACK)

        # Act
        double_pawn_move = Move(original_square=RankAndFile.from_algebraic('e7'),
                                target_square=RankAndFile.from_algebraic('e5'))
        state = double_pawn_move.execute(state)

        # Assert
        valid_moves = BoardCalculationCache(state).get_valid_moves()
        # White pawn should be able to either capture the black pawn en passant or just push normally
        valid_moves_expected = {
            Move(original_square=RankAndFile.from_algebraic('d5'), target_square=RankAndFile.from_algebraic('e6')),
            Move(original_square=RankAndFile.from_algebraic('d5'), target_square=RankAndFile.from_algebraic('d6')),
        }
        assert state.double_moved_pawn_square == RankAndFile.from_algebraic('e5')
        self.assertSetEqual(set(valid_moves), valid_moves_expected)

    def test_can_identify_checkmate(self):
        # Arrange
        board = Board.make_empty()
        board[RankAndFile.from_algebraic('a1')] = Piece(type=PieceType.ROOK, colour=Colour.WHITE)
        board[RankAndFile.from_algebraic('b1')] = Piece(type=PieceType.ROOK, colour=Colour.WHITE)
        board[RankAndFile.from_algebraic('a8')] = Piece(type=PieceType.KING, colour=Colour.BLACK)
        state = self.make_game_state(board=board, player_to_move=Colour.BLACK)

        # Act
        move_generation_cache = BoardCalculationCache(state)

        # Assert
        assert move_generation_cache.is_check()
        assert move_generation_cache.is_checkmate()
        assert len(move_generation_cache.get_valid_moves()) == 0
        assert move_generation_cache.get_game_result() == GameResult.WHITE_WIN
