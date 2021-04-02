#from .position_analyser_interface import PositionAnalyserInterface
#from .king import KingMoveGenerator
#from .knight import KnightMoveGenerator
#from .straight_line import StraightLineMoveGenerator
#from .pawn import PawnMoveGenerator
#from .move_generator_interface import MoveGenerator
#
#from chessbot.game.piece import PieceType
#
#PIECE_TYPE_TO_MOVE_GENERATOR = {
#    PieceType.KNIGHT: KnightMoveGenerator(),
#    PieceType.QUEEN: StraightLineMoveGenerator(moves_diagonally=True, moves_up_and_right=True),
#    PieceType.ROOK: StraightLineMoveGenerator(moves_diagonally=False, moves_up_and_right=True),
#    PieceType.BISHOP: StraightLineMoveGenerator(moves_diagonally=True, moves_up_and_right=False),
#    PieceType.PAWN: PawnMoveGenerator(),
#    PieceType.KING: KingMoveGenerator(),
#}
#
#