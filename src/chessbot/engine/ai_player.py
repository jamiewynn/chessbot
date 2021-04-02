import logging
import multiprocessing
import random
import time
from dataclasses import dataclass

import numpy as np

from chessbot.engine.evaluation import HeuristicEvaluator
from chessbot.game.game_state import GameState
from chessbot.game.move import Move
from chessbot.game.position_analyser import PositionAnalyser
from chessbot.game.piece import Colour
from chessbot.game.player import Player

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


@dataclass
class EngineParams:
    # Engine will try to think for about this number of seconds; it might exceed it, i.e. it's not a hard limit.
    target_thinking_time_secs: float

    # Number of threads to parallelise over using multithreading.
    num_threads: int = multiprocessing.cpu_count()


class AIPlayer(Player):
    """
    Implementation of Player interface that finds a move via multithreaded alpha-beta pruning with a heuristic function.
    """
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
        position_analyser = PositionAnalyser(state)
        valid_moves = position_analyser.get_valid_moves()

        # So that the bot doesn't always pick the same move when there are multiple best moves
        random.shuffle(valid_moves)

        # Compute evaluations in parallel, doing alpha-beta pruning on each thread separately
        alpha_beta_arg_tuples = [
            (move.execute(state), depth-1, self._evaluator, -np.inf, np.inf) for move in valid_moves
        ]

        # We only need to run via multiprocessing if more than 1 thread has been requested
        if self._params.num_threads > 1:
            move_values = list(self._pool.imap(alpha_beta_multiprocessing_kernel, alpha_beta_arg_tuples, chunksize=1))
        else:
            move_values = list(map(alpha_beta_multiprocessing_kernel, alpha_beta_arg_tuples))

        # Choose the best move whoever is the current player to move
        if state.player_to_move == Colour.WHITE:
            best_move_idx = np.argmax(move_values)
        else:
            best_move_idx = np.argmin(move_values)
        best_move = valid_moves[best_move_idx]
        _logger.info(f'Best move is {best_move} with value {move_values[best_move_idx]:.2f}')
        return best_move


def minimax_multiprocessing_kernel(args):
    # This just a kernel function that passes its args through to minimax; multiprocessing is easier to work with when
    # passing args as a single tuple in this way
    return minimax(*args)


def minimax(state: GameState, depth: int, evaluator: HeuristicEvaluator) -> float:
    _logger.debug(f'Evaluating:\n{state.board}\nto depth {depth}')
    position_analyser = PositionAnalyser(state)
    valid_moves = position_analyser.get_valid_moves()
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
    # This just a kernel function that passes its args through to alpha_beta_search; multiprocessing is easier to work
    # with when passing args as a single tuple in this way
    return alpha_beta_search(*args)


def alpha_beta_search(state: GameState, depth: int, evaluator: HeuristicEvaluator, alpha: float, beta: float) -> float:
    # For algorithmic details see https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
    position_analyser = PositionAnalyser(state)
    valid_moves = position_analyser.get_valid_moves()
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
                # Alpha cutoff
                break
        return value
