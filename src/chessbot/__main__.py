from chessbot.engine.ai_player import AIPlayer, EngineParams
from chessbot.game.board import Colour
from chessbot.game.game_manager import GameManager
from chessbot.game.game_state import GameState
from chessbot.game.move import Move
from chessbot.game.player import HumanPlayer, HardcodedPlayer, RandomPlayer
import logging
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    state = GameState.make_initial()

    #human_player = HumanPlayer(Colour.WHITE)
    #white_player = HardcodedPlayer(moves=Move(original_square=RankAndFile()))

    #white_player = AIPlayer(EngineParams(num_plies=1))
    #black_player = AIPlayer(EngineParams(num_plies=1))

    #white_player = AIPlayer(EngineParams(target_thinking_time_secs=10.))
    white_player = HumanPlayer(colour=Colour.WHITE)
    black_player = AIPlayer(EngineParams(target_thinking_time_secs=10.))

    #white_player = RandomPlayer()
    #black_player = RandomPlayer()


    game_manager = GameManager(initial_state=state, white_player=white_player, black_player=black_player)
    game_manager.run()
