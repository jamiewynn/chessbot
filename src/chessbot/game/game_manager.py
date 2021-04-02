from chessbot.game.game_state import GameState
from chessbot.game.move import Move
from chessbot.game.move_generation import PositionAnalyser
from chessbot.game.piece import Colour
from chessbot.game.player import Player


class GameManager:
    """ Handles getting moves from the players, validating and executing them. """

    def __init__(self, initial_state: GameState, white_player: Player, black_player: Player):
        self._state = initial_state
        self._players = {Colour.WHITE: white_player, Colour.BLACK: black_player}

    def run(self) -> None:
        """  Repeatedly get moves from the two players until the game is over. """
        while True:
            position_analyser = PositionAnalyser(self._state)
            result = position_analyser.get_game_result()
            if result is not None:
                print('Game ended with result', result)
                return
            print(self._state.board)
            player_to_move = self._players[self._state.player_to_move]
            move = self._get_valid_move(player_to_move)
            print('Received move:', move)
            self._state = move.execute(self._state)

    def _get_valid_move(self, player: Player) -> Move:
        position_analyser = PositionAnalyser(self._state)
        valid_moves = position_analyser.get_valid_moves()
        while not position_analyser.validate_move(move := player.get_move(self._state)):
            print(f'Provided move {move} is invalid! Valid moves are: {valid_moves}')
        return move
