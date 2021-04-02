import logging

from chessbot.engine.ai_player import AIPlayer, EngineParams
from chessbot.game.board import Colour
from chessbot.game.game_manager import GameManager
from chessbot.game.game_state import GameState
from chessbot.game.player import HumanPlayer

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    state = GameState.make_initial()

    white_player = HumanPlayer(colour=Colour.WHITE)
    black_player = AIPlayer(EngineParams(target_thinking_time_secs=10.))

    game_manager = GameManager(initial_state=state, white_player=white_player, black_player=black_player)
    game_manager.run()
