from collections import Counter

from chessbot.game.board import PieceType, Colour, NUM_RANKS
from chessbot.game.game_state import GameState
import numpy as np
from chessbot.game.move_generation import BoardCalculationCache
import logging
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

PIECE_VALUES = {
    PieceType.QUEEN: 9.,
    PieceType.ROOK: 5.,
    PieceType.BISHOP: 3.4,
    PieceType.KNIGHT: 3.,
    PieceType.PAWN: 1.,
    PieceType.KING: 0.,  # Kings can never come off the board, so their assigned value here is irrelevant
}

EXTRA_PAWN_VALUE_PER_STEP_FORWARD = 0.1


class HeuristicEvaluator:
    def evaluate(self, position: GameState) -> float:
        move_generation_cache = BoardCalculationCache(position)

        # If the game is over, this makes the evaluation trivial
        result = move_generation_cache.get_game_result()
        _logger.debug(f'Result {result}')
        if result is not None:
            if result == result.WHITE_WIN:
                return np.inf
            elif result == result.BLACK_WIN:
                return -np.inf
            elif result == result.DRAW:
                return 0.
            else:
                raise ValueError

        score = 0.

        score += self._material_score(position)
        score += self._attacked_squares_score(move_generation_cache)
        score += self._hanging_pieces_score(position, move_generation_cache)
        score += self._pawn_structure_score(position)

        _logger.debug(f'Heuristic evaluation for position\n{position.board}\n')
        _logger.debug(f'Score: {score}')

        return score

    @staticmethod
    def _material_score(position: GameState) -> float:
        score = 0.
        for occupied_square in position.board.get_occupied_squares():
            piece = position.board[occupied_square]
            if piece.colour == Colour.WHITE:
                score += PIECE_VALUES[piece.type]
            else:
                score -= PIECE_VALUES[piece.type]
        return score

    @staticmethod
    def _attacked_squares_score(move_generation_cache: BoardCalculationCache) -> float:
        score = 0.

        # Attacking more squares is better, and central squares in particular are better targets
        for player in Colour:
            attacks = move_generation_cache.get_attacks(attacking_side=player)
            for attack in attacks:
                # rank=3.5, file=3.5 corresponds to the centre of the board
                taxicab_distance_from_centre = abs(attack.rank - 3.5) + abs(attack.file - 3.5)
                # Inner four squares
                if taxicab_distance_from_centre <= 1.5:
                    attack_score = 0.04
                # Ring of squares around the inner four
                elif taxicab_distance_from_centre <= 2.5:
                    attack_score = 0.03
                # More distant squares
                else:
                    attack_score = 0.02

                if player == Colour.WHITE:
                    score += attack_score
                else:
                    score -= attack_score
        return score

    @staticmethod
    def _hanging_pieces_score(position: GameState, move_generation_cache: BoardCalculationCache) -> float:
        # Pieces attacked more times than they are defended are considered lost for the purposes of heuristic evaluation
        attacks_by_white = move_generation_cache.get_attacks(attacking_side=Colour.WHITE)
        attacks_by_black = move_generation_cache.get_attacks(attacking_side=Colour.BLACK)
        attack_counters = {
            Colour.WHITE: Counter(attacks_by_white),
            Colour.BLACK: Counter(attacks_by_black),
        }

        score = 0.

        for square in position.board.get_occupied_squares():
            piece = position.board[square]
            defence_count = attack_counters[piece.colour][square]
            attack_count = attack_counters[piece.colour.other()][square]
            if attack_count > defence_count:
                if piece.colour == Colour.WHITE:
                    score -= PIECE_VALUES[piece.type]
                if piece.colour == Colour.BLACK:
                    score += PIECE_VALUES[piece.type]

        return score

    @staticmethod
    def _pawn_structure_score(position: GameState):
        score = 0.
        for square in position.board.get_occupied_squares():
            piece = position.board[square]
            if piece.type != PieceType.PAWN:
                continue

            distance_from_back_rank = square.rank if piece.colour == Colour.WHITE else NUM_RANKS-1 - square.rank
            pawn_advancement_value = distance_from_back_rank * EXTRA_PAWN_VALUE_PER_STEP_FORWARD
            if piece.colour == Colour.BLACK:
                pawn_advancement_value = -pawn_advancement_value
            score += pawn_advancement_value
        return score
