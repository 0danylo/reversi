import random
import reversi
from copy import deepcopy
import engine


class Strategy:
    """Base class for strategies. Subclasses implement choose_move(board, me, opp)."""
    name = "base"

    def choose_move(self, board, me=1, opp=2):
        raise NotImplementedError()


class RandomStrategy(Strategy):
    name = "random"

    def choose_move(self, board, me=1, opp=2):
        moves = reversi.get_legal_moves(board, me, opp)
        if not moves:
            return None
        return random.choice(moves)


class GreedyStrategy(Strategy):
    name = "greedy"

    def choose_move(self, board, me=1, opp=2):
        moves = reversi.get_legal_moves(board, me, opp)
        if not moves:
            return None

        best = None
        best_score = -10**9
        reversi.board_global = board
        for m in moves:
            s = reversi.score_move(board, m, me, opp)
            if s > best_score:
                best_score = s
                best = m
        return best


class CornerFirstStrategy(Strategy):
    name = "corner_first"

    def choose_move(self, board, me=1, opp=2):
        moves = reversi.get_legal_moves(board, me, opp)
        if not moves:
            return None

        # prefer corners first, otherwise fallback to greedy
        max_r = len(board) - 1
        corners = {(0, 0), (0, len(board[0]) - 1), (max_r, 0), (max_r, len(board[-1]) - 1)}
        for m in moves:
            r, c, _ = m
            if (r, c) in corners:
                return m

        # fallback greedy
        return GreedyStrategy().choose_move(board, me, opp)


class AlphaBetaStrategy(Strategy):
    """Alpha-beta pruning minimax strategy.

    Parameters:
      depth: search depth (plies)
    """
    name = "ab"

    def __init__(self, depth=3):
        self.depth = depth

    def choose_move(self, board, me=1, opp=2):
        reversi.board_global = board

        moves = reversi.get_legal_moves(board, me, opp)
        if not moves:
            return None

        def evaluate(b):
            # simple heuristic: disk difference + corner control
            counts = engine.count_disks(b)
            my = counts[me]
            other = counts[opp]
            score = (my - other) * 10

            max_r = len(b) - 1
            corners = [(0, 0), (0, len(b[0]) - 1), (max_r, 0), (max_r, len(b[-1]) - 1)]
            for cr, cc in corners:
                if reversi.is_on_board(cr, cc):
                    if b[cr][cc] == me:
                        score += 100
                    elif b[cr][cc] == opp:
                        score -= 100
            return score

        def alphabeta(b, depth, alpha, beta, current_player):
            # current_player is whose turn it is (1 or 2)
            other = 1 if current_player == 2 else 2

            # terminal or depth
            reversi.board_global = b
            moves_here = reversi.get_legal_moves(b, current_player, other)
            
            if depth == 0 or not moves_here:
                # if no moves for both, terminal
                if not moves_here:
                    moves_other = reversi.get_legal_moves(b, other, current_player)
                    if not moves_other:
                        return evaluate(b)
                    # pass turn
                    return alphabeta(b, depth, alpha, beta, other)
                return evaluate(b)

            if current_player == me:
                value = -10**9
                for mv in moves_here:
                    nb = deepcopy(b)
                    engine.apply_move(nb, mv, current_player)
                    val = alphabeta(nb, depth - 1, alpha, beta, other)
                    if val > value:
                        value = val
                    if value > alpha:
                        alpha = value
                    if alpha >= beta:
                        break
                return value
            else:
                value = 10**9
                for mv in moves_here:
                    nb = deepcopy(b)
                    engine.apply_move(nb, mv, current_player)
                    val = alphabeta(nb, depth - 1, alpha, beta, other)
                    if val < value:
                        value = val
                    if value < beta:
                        beta = value
                    if alpha >= beta:
                        break
                return value

        # Top-level: choose best move by running alphabeta for each move
        best_move = None
        best_val = -10**9
        alpha = -10**9
        beta = 10**9

        for mv in moves:
            nb = deepcopy(board)
            engine.apply_move(nb, mv, me)
            val = alphabeta(nb, self.depth - 1, alpha, beta, opp)
            if val > best_val:
                best_val = val
                best_move = mv
            if val > alpha:
                alpha = val

        return best_move
