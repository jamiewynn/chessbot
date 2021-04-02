import random
from abc import ABC, abstractmethod
from typing import List

from chessbot.game.game_state import CastleType, GameState
from chessbot.game.move import Castle, Move, Promotion
from chessbot.game.move_generation import PositionAnalyser
from chessbot.game.piece import Colour, PieceType, RankAndFile


class Player(ABC):
    """ Interface that must be implemented to satisfy the requirements of the GameManager. """

    @abstractmethod
    def get_move(self, state: GameState) -> Move:
        """ Choose a move to play for a given state. """
        pass


class HumanPlayer(Player):
    """
    Implementation of Player interface that prompts for a move from the terminal.
    """
    def __init__(self, colour: Colour):
        self._colour = colour
        print('Enter moves when prompted in one of the following formats:\n'
              '- Castling: O-O or O-O-O\n'
              '- Normal move or capture: e4-e5, h1-a8, etc\n'
              '- Promotion: h7-h8Q, b2-b1N etc\n')

    def get_move(self, state: GameState) -> Move:
        print(f'Enter move...')
        assert self._colour == state.player_to_move
        move_str = input()
        return self._parse_move(move_str)

    def _parse_move(self, move_str: str) -> Move:
        move_str = move_str.rstrip()
        if move_str == 'O-O':
            return Castle.make(colour=self._colour, castle_type=CastleType.KINGSIDE)
        if move_str == 'O-O-O':
            return Castle.make(colour=self._colour, castle_type=CastleType.QUEENSIDE)

        start_square, end_square = move_str.split('-')
        start_square = RankAndFile.from_algebraic(start_square)

        # For a promotion, end square may indicate name of piece
        if len(end_square) == 3:
            end_square = end_square[:2]
            end_square = RankAndFile.from_algebraic(end_square)
            promoted_piece = PieceType.from_char(end_square[-1])
            return Promotion(original_square=start_square, target_square=end_square, new_piece_type=promoted_piece)

        # Otherwise we just have a normal move
        return Move(original_square=start_square, target_square=RankAndFile.from_algebraic(end_square))


class HardcodedPlayer(Player):
    """ Implementation of Player interface that plays a predetermined list of moves. Useful for testing. """
    def __init__(self, moves: List[Move]):
        self._moves_iter = iter(moves)

    def get_move(self, state: GameState) -> Move:
        return next(self._moves_iter)


class RandomPlayer(Player):
    """ Just chooses a valid random move each time. """
    def get_move(self, state: GameState) -> Move:
        valid_moves = PositionAnalyser(state).get_valid_moves()
        return random.choice(valid_moves)
