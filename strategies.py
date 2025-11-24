import random
import time
import multiprocessing
from multiprocessing import Pipe, Process
from copy import deepcopy as _deepcopy
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

    def __init__(self, depth=3, max_time=3.0):
        self.depth = depth
        # per-move time budget (seconds); if None or <=0, no time limit
        self.max_time = max_time

    def choose_move(self, board, me=1, opp=2):
        # If a strict per-move time limit is set, run iterative deepening in
        # a separate process and collect intermediate results (one per
        # completed depth) via a pipe; parent will terminate the child when
        # the budget expires and use the last completed iteration's result.
        reversi.board_global = board
        moves = reversi.get_legal_moves(board, me, opp)
        if not moves:
            return None

        

        def _ab_worker(conn, board_arg, me_arg, opp_arg, max_depth):
            # perform iterative deepening without time checks; after each
            # completed depth send the best_move via conn
            try:
                # local copies
                bd = _deepcopy(board_arg)
                for depth_limit in range(1, max(1, max_depth) + 1):
                    # simple alphabeta for this worker
                    def evaluate_local(b):
                        counts = engine.count_disks(b)
                        my = counts[me_arg]
                        other = counts[opp_arg]
                        score = (my - other) * 10
                        max_r = len(b) - 1
                        corners = [(0, 0), (0, len(b[0]) - 1), (max_r, 0), (max_r, len(b[-1]) - 1)]
                        for cr, cc in corners:
                            if reversi.is_on_board(cr, cc):
                                if b[cr][cc] == me_arg:
                                    score += 100
                                elif b[cr][cc] == opp_arg:
                                    score -= 100
                        return score

                    def alphabeta_local(b, depth, alpha, beta, current_player):
                        other = 1 if current_player == 2 else 2
                        reversi.board_global = b
                        moves_here = reversi.get_legal_moves(b, current_player, other)
                        if depth == 0 or not moves_here:
                            if not moves_here:
                                moves_other = reversi.get_legal_moves(b, other, current_player)
                                if not moves_other:
                                    return evaluate_local(b)
                                return alphabeta_local(b, depth, alpha, beta, other)
                            return evaluate_local(b)

                        if current_player == me_arg:
                            value = -10**9
                            for mv in moves_here:
                                nb = _deepcopy(b)
                                engine.apply_move(nb, mv, current_player)
                                val = alphabeta_local(nb, depth - 1, alpha, beta, other)
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
                                nb = _deepcopy(b)
                                engine.apply_move(nb, mv, current_player)
                                val = alphabeta_local(nb, depth - 1, alpha, beta, other)
                                if val < value:
                                    value = val
                                if value < beta:
                                    beta = value
                                if alpha >= beta:
                                    break
                            return value

                    # order candidate moves with greedy heuristic
                    ordered_moves = sorted(reversi.get_legal_moves(bd, me_arg, opp_arg), key=lambda mv: reversi.score_move(bd, mv, me_arg, opp_arg), reverse=True)
                    best_move = None
                    best_val = -10**9
                    alpha = -10**9
                    beta = 10**9
                    for mv in ordered_moves:
                        nb = _deepcopy(bd)
                        engine.apply_move(nb, mv, me_arg)
                        val = alphabeta_local(nb, depth_limit - 1, alpha, beta, opp_arg)
                        if val > best_val:
                            best_val = val
                            best_move = mv
                        if val > alpha:
                            alpha = val

                    # send intermediate result to parent
                    try:
                        conn.send(best_move)
                    except Exception:
                        pass
                conn.close()
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass

        # If max_time specified, run worker and poll for intermediate results
        if self.max_time and self.max_time > 0:
            parent_conn, child_conn = Pipe(duplex=False)
            p = Process(target=_ab_worker, args=(child_conn, board, me, opp, self.depth))
            p.start()
            last_move = None
            start_t = time.time()
            remaining = self.max_time
            try:
                while True:
                    if parent_conn.poll(timeout=remaining):
                        try:
                            last_move = parent_conn.recv()
                        except EOFError:
                            pass
                    if not p.is_alive():
                        break
                    elapsed = time.time() - start_t
                    remaining = self.max_time - elapsed
                    if remaining <= 0:
                        break
                # time's up or process finished; ensure process terminated
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=0.1)
                # try to receive any remaining message
                while parent_conn.poll(timeout=0.01):
                    try:
                        last_move = parent_conn.recv()
                    except EOFError:
                        break
                parent_conn.close()
            finally:
                if p.is_alive():
                    p.terminate()
                    p.join()
            return last_move

        # no time limit: run worker synchronously and collect last sent move
        parent_conn, child_conn = Pipe(duplex=False)
        p = Process(target=_ab_worker, args=(child_conn, board, me, opp, self.depth))
        p.start()
        last_move = None
        while True:
            if parent_conn.poll(timeout=0.1):
                try:
                    last_move = parent_conn.recv()
                except EOFError:
                    pass
            if not p.is_alive():
                break
        p.join()
        parent_conn.close()
        return last_move


def ab_improved_worker(conn, board_arg, me_arg, opp_arg, max_depth):
    # Module-level improved worker (iterative deepening + improved eval).
    try:
        bd = _deepcopy(board_arg)

        def evaluate_local(b):
            counts = engine.count_disks(b)
            my = counts[me_arg]
            other = counts[opp_arg]
            score = (my - other) * 10

            max_r = len(b) - 1
            corners = [(0, 0), (0, len(b[0]) - 1), (max_r, 0), (max_r, len(b[-1]) - 1)]
            for cr, cc in corners:
                if reversi.is_on_board(cr, cc):
                    if b[cr][cc] == me_arg:
                        score += 1000
                    elif b[cr][cc] == opp_arg:
                        score -= 1000

            # edge heuristics
            for r in range(len(b)):
                for c in range(len(b[r])):
                    if r == 0 or r == len(b) - 1 or c == 0 or c == len(b[r]) - 1:
                        if b[r][c] == me_arg:
                            score += 50
                        elif b[r][c] == opp_arg:
                            score -= 50

            # corner-adjacent penalty
            corner_adjacent = set()
            for cr, cc in corners:
                for dr, dc in reversi.DELTA:
                    nr, nc = cr + dr, cc + dc
                    if reversi.is_on_board(nr, nc):
                        corner_adjacent.add((nr, nc))

            for (ar, ac) in corner_adjacent:
                if reversi.is_on_board(ar, ac):
                    for cr, cc in corners:
                        if abs(cr - ar) <= 1 and abs(cc - ac) <= 1:
                            if reversi.is_on_board(cr, cc) and b[cr][cc] == 0:
                                if b[ar][ac] == me_arg:
                                    score -= 800
                                elif b[ar][ac] == opp_arg:
                                    score += 800

            reversi.board_global = b
            my_moves = len(reversi.get_legal_moves(b, me_arg, opp_arg))
            opp_moves = len(reversi.get_legal_moves(b, opp_arg, me_arg))
            score += (my_moves - opp_moves) * 20

            return score

        def alphabeta_local(b, depth, alpha, beta, current_player):
            other = 1 if current_player == 2 else 2
            reversi.board_global = b
            moves_here = reversi.get_legal_moves(b, current_player, other)
            if depth == 0 or not moves_here:
                if not moves_here:
                    moves_other = reversi.get_legal_moves(b, other, current_player)
                    if not moves_other:
                        return evaluate_local(b)
                    return alphabeta_local(b, depth, alpha, beta, other)
                return evaluate_local(b)

            # order moves by greedy heuristic
            moves_here = sorted(moves_here, key=lambda mv: reversi.score_move(b, mv, current_player, other), reverse=True)

            if current_player == me_arg:
                value = -10**9
                for mv in moves_here:
                    nb = _deepcopy(b)
                    engine.apply_move(nb, mv, current_player)
                    val = alphabeta_local(nb, depth - 1, alpha, beta, other)
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
                    nb = _deepcopy(b)
                    engine.apply_move(nb, mv, current_player)
                    val = alphabeta_local(nb, depth - 1, alpha, beta, other)
                    if val < value:
                        value = val
                    if value < beta:
                        beta = value
                    if alpha >= beta:
                        break
                return value

        for depth_limit in range(1, max(1, max_depth) + 1):
            ordered_moves = sorted(reversi.get_legal_moves(bd, me_arg, opp_arg), key=lambda mv: reversi.score_move(bd, mv, me_arg, opp_arg), reverse=True)
            best_move = None
            best_val = -10**9
            alpha = -10**9
            beta = 10**9
            for mv in ordered_moves:
                nb = _deepcopy(bd)
                engine.apply_move(nb, mv, me_arg)
                val = alphabeta_local(nb, depth_limit - 1, alpha, beta, opp_arg)
                if val > best_val:
                    best_val = val
                    best_move = mv
                if val > alpha:
                    alpha = val

            try:
                # send (depth, move) so callers can profile depth reached
                conn.send((depth_limit, best_move))
            except Exception:
                pass

        try:
            conn.close()
        except Exception:
            pass
    except Exception:
        try:
            conn.close()
        except Exception:
            pass


class AlphaBetaImprovedStrategy(AlphaBetaStrategy):
    """Improved Alpha-Beta that uses a stronger evaluation (includes greedy's
    corner-adjacent penalties and edge heuristics) and simple move ordering.
    """
    name = "ab2"

    def __init__(self, depth=3, max_time=3.0):
        super().__init__(depth=depth, max_time=max_time)

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
        # Use the module-level improved worker and strict polling so we
        # never return a move that was produced after the time budget.
        reversi.board_global = board
        moves = reversi.get_legal_moves(board, me, opp)
        if not moves:
            return None

        # run worker process that sends one best-move per completed depth
        if self.max_time and self.max_time > 0:
            parent_conn, child_conn = Pipe(duplex=False)
            p = Process(target=ab_improved_worker, args=(child_conn, board, me, opp, self.depth))
            p.start()
            last_move = None
            start_t = time.time()
            remaining = self.max_time
            try:
                while True:
                    if parent_conn.poll(timeout=remaining):
                        try:
                            res = parent_conn.recv()
                            if isinstance(res, tuple) and len(res) == 2:
                                depth_val, mv = res
                                last_move = mv
                                try:
                                    self.last_depth = depth_val
                                except Exception:
                                    pass
                            else:
                                last_move = res
                                try:
                                    self.last_depth = None
                                except Exception:
                                    pass
                        except EOFError:
                            pass
                    if not p.is_alive():
                        break
                    elapsed = time.time() - start_t
                    remaining = self.max_time - elapsed
                    if remaining <= 0:
                        break
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=0.1)
                while parent_conn.poll(timeout=0.01):
                    try:
                        res = parent_conn.recv()
                        if isinstance(res, tuple) and len(res) == 2:
                            depth_val, mv = res
                            last_move = mv
                            try:
                                self.last_depth = depth_val
                            except Exception:
                                pass
                        else:
                            last_move = res
                            try:
                                self.last_depth = None
                            except Exception:
                                pass
                    except EOFError:
                        break
                parent_conn.close()
            finally:
                if p.is_alive():
                    p.terminate()
                    p.join()
            return last_move

        # no time limit: just run worker and collect last move
        parent_conn, child_conn = Pipe(duplex=False)
        p = Process(target=ab_improved_worker, args=(child_conn, board, me, opp, self.depth))
        p.start()
        last_move = None
        while True:
            if parent_conn.poll(timeout=0.1):
                try:
                    res = parent_conn.recv()
                    if isinstance(res, tuple) and len(res) == 2:
                        depth_val, mv = res
                        last_move = mv
                        try:
                            self.last_depth = depth_val
                        except Exception:
                            pass
                    else:
                        last_move = res
                        try:
                            self.last_depth = None
                        except Exception:
                            pass
                except EOFError:
                    pass
            if not p.is_alive():
                break
        p.join()
        parent_conn.close()
        return last_move
