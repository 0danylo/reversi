import random
import reversi


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
