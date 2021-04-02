from unittest import TestCase

from chessbot.engine.ai_player import AIPlayer, EngineParams
from chessbot.game.game_state import GameState
from chessbot.game.move import Move
from chessbot.game.piece import RankAndFile


class EngineTest(TestCase):
    def test_engine_finds_mate_in_one(self):
        # Verifies that the engine finds the mate in one in Scholar's mate

        # Arrange
        state = GameState.make_initial()
        moves = [
            Move(original_square=RankAndFile.from_algebraic('e2'), target_square=RankAndFile.from_algebraic('e4')),
            Move(original_square=RankAndFile.from_algebraic('e7'), target_square=RankAndFile.from_algebraic('e5')),
            Move(original_square=RankAndFile.from_algebraic('d1'), target_square=RankAndFile.from_algebraic('h5')),
            Move(original_square=RankAndFile.from_algebraic('g8'), target_square=RankAndFile.from_algebraic('f6')),
            Move(original_square=RankAndFile.from_algebraic('f1'), target_square=RankAndFile.from_algebraic('c4')),
            Move(original_square=RankAndFile.from_algebraic('a7'), target_square=RankAndFile.from_algebraic('a6')),
        ]
        for move in moves:
            state = move.execute(state)

        engine = AIPlayer(EngineParams(target_thinking_time_secs=0.01))

        # Act
        move = engine.get_move(state)

        # Assert
        expected_move = Move(original_square=RankAndFile.from_algebraic('h5'),
                             target_square=RankAndFile.from_algebraic('f7'))
        self.assertEqual(move, expected_move)
