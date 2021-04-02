import logging
import multiprocessing
import random
import time
from dataclasses import dataclass

import numpy as np

from chessbot.engine.evaluation import HeuristicEvaluator
from chessbot.game.board import Colour
from chessbot.game.game_state import GameState
from chessbot.game.move import Move
from chessbot.game.move_generation import BoardCalculationCache
from chessbot.game.player import Player

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


@dataclass
class EngineParams:
    target_thinking_time_secs: float
    num_threads: int = multiprocessing.cpu_count()


class AIPlayer(Player):
    def __init__(self, params: EngineParams):
        self._params = params
        self._evaluator = HeuristicEvaluator()
        self._pool = multiprocessing.Pool(processes=self._params.num_threads)

    def get_move(self, state: GameState) -> Move:
        total_start_time = time.time()
        depth = 1
        while True:
            _logger.info(f'Searching to depth {depth}...')
            start_time = time.time()
            best_move = self._get_move_to_specified_depth(state, depth)
            end_time = time.time()
            thinking_time = end_time - start_time
            _logger.info(f'Search took {thinking_time:.2f} secs.')
            depth += 1

            # Only go one level deeper if an increase of 10x in our thinking time would not go past the target thinking
            # time:
            if end_time + thinking_time*10. > total_start_time + self._params.target_thinking_time_secs:
                _logger.info('Terminating search')
                break

        return best_move

    def _get_move_to_specified_depth(self, state: GameState, depth: int):
        # Minimax
        move_generation_cache = BoardCalculationCache(state)
        valid_moves = move_generation_cache.get_valid_moves()

        # So that the bot doesn't always pick the same move when there are multiple best moves
        random.shuffle(valid_moves)

        # Compute evaluations in parallel, doing alpha-beta pruning on each thread separately
        alpha_beta_arg_tuples = [
            (move.execute(state), depth-1, self._evaluator, -np.inf, np.inf) for move in valid_moves
        ]
        if self._params.num_threads > 1:
            move_values = list(self._pool.imap(alpha_beta_multiprocessing_kernel, alpha_beta_arg_tuples, chunksize=1))
        else:
            move_values = list(map(alpha_beta_multiprocessing_kernel, alpha_beta_arg_tuples))

        if state.player_to_move == Colour.WHITE:
            best_move_idx = np.argmax(move_values)
        else:
            best_move_idx = np.argmin(move_values)
        best_move = valid_moves[best_move_idx]
        _logger.info(f'Best move is {best_move} with value {move_values[best_move_idx]:.2f}')
        return best_move

    # TODO: try LRU cache below, as a transposition table. Does it speed things up?


def minimax_multiprocessing_kernel(args):
    # This just a kernel function that passes its args through to minimax; multiprocessing is easier to work with when
    # passing args as a single tuple in this way
    return minimax(*args)


def minimax(state: GameState, depth: int, evaluator: HeuristicEvaluator) -> float:
    _logger.debug(f'Evaluating:\n{state.board}\nto depth {depth}')
    move_generation_cache = BoardCalculationCache(state)
    valid_moves = move_generation_cache.get_valid_moves()
    # At a depth of zero, or if there are no valid moves (i.e. game is over), we evaluate using the heuristic
    if depth == 0 or len(valid_moves) == 0:
        return evaluator.evaluate(state)

    # At a depth > 0, we evaluate recursively
    move_values = [
        minimax(state=move.execute(state), depth=depth-1, evaluator=evaluator)
        for move in valid_moves
    ]

    # White wants to maximise the evaluation; black wants to minimise it
    return max(move_values) if state.player_to_move == Colour.WHITE else min(move_values)


def alpha_beta_multiprocessing_kernel(args):
    return alpha_beta_search(*args)


def alpha_beta_search(state: GameState, depth: int, evaluator: HeuristicEvaluator, alpha: float, beta: float) -> float:
    # See https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
    move_generation_cache = BoardCalculationCache(state)
    valid_moves = move_generation_cache.get_valid_moves()
    if depth == 0 or len(valid_moves) == 0:
        return evaluator.evaluate(state)

    if state.player_to_move == Colour.WHITE:
        value = -np.inf
        for move in valid_moves:
            value = max(value, alpha_beta_search(move.execute(state), depth-1, evaluator, alpha, beta))
            alpha = max(alpha, value)
            if alpha >= beta:
                # Beta cutoff
                break
        return value
    else:
        value = np.inf
        for move in valid_moves:
            value = min(value, alpha_beta_search(move.execute(state), depth-1, evaluator, alpha, beta))
            beta = min(beta, value)
            if beta <= alpha:
                # Beta cutoff
                break
        return value
