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


class AlphaBetaImprovedStrategy(AlphaBetaStrategy):
    """Improved Alpha-Beta that uses a stronger evaluation (includes greedy's
    corner-adjacent penalties and edge heuristics) and simple move ordering.
    """
    name = "ab2"

    def __init__(self, depth=3):
        super().__init__(depth=depth)

    def evaluate(self, b, me, opp):
        # disk difference
        counts = engine.count_disks(b)
        my = counts[me]
        other = counts[opp]
        score = (my - other) * 10

        # corners (strong)
        max_r = len(b) - 1
        corners = [(0, 0), (0, len(b[0]) - 1), (max_r, 0), (max_r, len(b[-1]) - 1)]
        for cr, cc in corners:
            if reversi.is_on_board(cr, cc):
                if b[cr][cc] == me:
                    score += 1000
                elif b[cr][cc] == opp:
                    score -= 1000

        # edge heuristics: small bonus for occupying edges
        for r in range(len(b)):
            for c in range(len(b[r])):
                if r == 0 or r == len(b) - 1 or c == 0 or c == len(b[r]) - 1:
                    if b[r][c] == me:
                        score += 50
                    elif b[r][c] == opp:
                        score -= 50

        # corner-adjacent penalty to avoid giving up corners
        corner_adjacent = set()
        for cr, cc in corners:
            for dr, dc in reversi.DELTA:
                nr, nc = cr + dr, cc + dc
                if reversi.is_on_board(nr, nc):
                    corner_adjacent.add((nr, nc))

        for (ar, ac) in corner_adjacent:
            if reversi.is_on_board(ar, ac):
                # if we occupy an adj cell while the corner is empty, penalize
                for cr, cc in corners:
                    if abs(cr - ar) <= 1 and abs(cc - ac) <= 1:
                        if reversi.is_on_board(cr, cc) and b[cr][cc] == 0:
                            if b[ar][ac] == me:
                                score -= 800
                            elif b[ar][ac] == opp:
                                score += 800

        # mobility: prefer positions with more legal moves
        reversi.board_global = b
        my_moves = len(reversi.get_legal_moves(b, me, opp))
        opp_moves = len(reversi.get_legal_moves(b, opp, me))
        score += (my_moves - opp_moves) * 20

        return score

    def choose_move(self, board, me=1, opp=2):
        # override to use improved evaluation and move ordering
        reversi.board_global = board

        moves = reversi.get_legal_moves(board, me, opp)
        if not moves:
            return None

        # simple move ordering: sort by reversi.score_move (greedy heuristic)
        moves = sorted(moves, key=lambda mv: reversi.score_move(board, mv, me, opp), reverse=True)

        # copy-paste alphabeta from base but call improved evaluate and order children
        from copy import deepcopy

        def alphabeta(b, depth, alpha, beta, current_player):
            other = 1 if current_player == 2 else 2
            reversi.board_global = b
            moves_here = reversi.get_legal_moves(b, current_player, other)

            if depth == 0 or not moves_here:
                if not moves_here:
                    moves_other = reversi.get_legal_moves(b, other, current_player)
                    if not moves_other:
                        return self.evaluate(b, me, opp)
                    return alphabeta(b, depth, alpha, beta, other)
                return self.evaluate(b, me, opp)

            # order children using greedy heuristic for better pruning
            moves_here = sorted(moves_here, key=lambda mv: reversi.score_move(b, mv, current_player, other), reverse=True)

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

        # top-level search
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
