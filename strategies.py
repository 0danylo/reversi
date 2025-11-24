import random
import time
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
        reversi.board_global = board
        moves = reversi.get_legal_moves(board, me, opp)
        if not moves:
            return None

        start_time = time.time()
        best_move = None

        def evaluate_local(b):
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

        def alphabeta_local(b, depth, alpha, beta, current_player):
            if self.max_time and self.max_time > 0:
                if time.time() - start_time > self.max_time:
                    raise TimeoutError()

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

            if current_player == me:
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

        try:
            for depth_limit in range(1, max(1, self.depth) + 1):
                ordered_moves = sorted(reversi.get_legal_moves(board, me, opp), key=lambda mv: reversi.score_move(board, mv, me, opp), reverse=True)
                current_best_move = None
                best_val = -10**9
                alpha = -10**9
                beta = 10**9
                
                for mv in ordered_moves:
                    nb = _deepcopy(board)
                    engine.apply_move(nb, mv, me)
                    val = alphabeta_local(nb, depth_limit - 1, alpha, beta, opp)
                    if val > best_val:
                        best_val = val
                        current_best_move = mv
                    if val > alpha:
                        alpha = val
                
                best_move = current_best_move
                self.last_depth = depth_limit
        except TimeoutError:
            pass

        return best_move if best_move else moves[0]





class AlphaBetaImprovedStrategy(AlphaBetaStrategy):
    """Improved Alpha-Beta that uses a stronger evaluation (includes greedy's
    corner-adjacent penalties and edge heuristics) and simple move ordering.
    """
    name = "ab2"

    def __init__(self, depth=100, max_time=3.0):
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
        reversi.board_global = board
        moves = reversi.get_legal_moves(board, me, opp)
        if not moves:
            return None

        start_time = time.time()
        best_move = None

        def alphabeta_local(b, depth, alpha, beta, current_player):
            if self.max_time and self.max_time > 0:
                if time.time() - start_time > self.max_time:
                    raise TimeoutError()

            other = 1 if current_player == 2 else 2
            reversi.board_global = b
            moves_here = reversi.get_legal_moves(b, current_player, other)
            if depth == 0 or not moves_here:
                if not moves_here:
                    moves_other = reversi.get_legal_moves(b, other, current_player)
                    if not moves_other:
                        return self.evaluate(b, me, opp)
                    return alphabeta_local(b, depth, alpha, beta, other)
                return self.evaluate(b, me, opp)

            # order moves by greedy heuristic
            moves_here = sorted(moves_here, key=lambda mv: reversi.score_move(b, mv, current_player, other), reverse=True)

            if current_player == me:
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

        try:
            # Calculate max possible depth based on empty squares
            empty_squares = sum(row.count(0) for row in board)
            effective_max_depth = min(self.depth, empty_squares)

            for depth_limit in range(1, max(1, effective_max_depth) + 1):
                ordered_moves = sorted(reversi.get_legal_moves(board, me, opp), key=lambda mv: reversi.score_move(board, mv, me, opp), reverse=True)
                current_best_move = None
                best_val = -10**9
                alpha = -10**9
                beta = 10**9
                
                for mv in ordered_moves:
                    nb = _deepcopy(board)
                    engine.apply_move(nb, mv, me)
                    val = alphabeta_local(nb, depth_limit - 1, alpha, beta, opp)
                    if val > best_val:
                        best_val = val
                        current_best_move = mv
                    if val > alpha:
                        alpha = val
                
                best_move = current_best_move
                self.last_depth = depth_limit
        except TimeoutError:
            pass

        return best_move if best_move else moves[0]


class AlphaBetaOptimizedStrategy(AlphaBetaImprovedStrategy):
    """Optimized Alpha-Beta that avoids deepcopy by using make/unmake move.
    """
    name = "ab_opt"

    def __init__(self, depth=3, max_time=3.0):
        super().__init__(depth=depth, max_time=max_time)

    def choose_move(self, board, me=1, opp=2):
        reversi.board_global = board
        moves = reversi.get_legal_moves(board, me, opp)
        if not moves:
            return None

        start_time = time.time()
        best_move = None

        def undo_move(b, move, player):
            r, c, flips = move
            b[r][c] = 0
            other = 1 if player == 2 else 2
            for fr, fc in flips:
                b[fr][fc] = other

        def alphabeta_local(b, depth, alpha, beta, current_player):
            if self.max_time and self.max_time > 0:
                if time.time() - start_time > self.max_time:
                    raise TimeoutError()

            other = 1 if current_player == 2 else 2
            reversi.board_global = b
            moves_here = reversi.get_legal_moves(b, current_player, other)
            if depth == 0 or not moves_here:
                if not moves_here:
                    moves_other = reversi.get_legal_moves(b, other, current_player)
                    if not moves_other:
                        return self.evaluate(b, me, opp)
                    return alphabeta_local(b, depth, alpha, beta, other)
                return self.evaluate(b, me, opp)

            # order moves by greedy heuristic
            moves_here = sorted(moves_here, key=lambda mv: reversi.score_move(b, mv, current_player, other), reverse=True)

            if current_player == me:
                value = -10**9
                for mv in moves_here:
                    engine.apply_move(b, mv, current_player)
                    val = alphabeta_local(b, depth - 1, alpha, beta, other)
                    undo_move(b, mv, current_player)
                    
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
                    engine.apply_move(b, mv, current_player)
                    val = alphabeta_local(b, depth - 1, alpha, beta, other)
                    undo_move(b, mv, current_player)
                    
                    if val < value:
                        value = val
                    if value < beta:
                        beta = value
                    if alpha >= beta:
                        break
                return value

        try:
            # Use a copy of the board for the search to avoid modifying the original board passed by the caller
            search_board = _deepcopy(board)
            
            # Calculate max possible depth based on empty squares
            empty_squares = sum(row.count(0) for row in board)
            effective_max_depth = min(self.depth, empty_squares)

            for depth_limit in range(1, max(1, effective_max_depth) + 1):
                ordered_moves = sorted(reversi.get_legal_moves(search_board, me, opp), key=lambda mv: reversi.score_move(search_board, mv, me, opp), reverse=True)
                current_best_move = None
                best_val = -10**9
                alpha = -10**9
                beta = 10**9
                
                for mv in ordered_moves:
                    engine.apply_move(search_board, mv, me)
                    val = alphabeta_local(search_board, depth_limit - 1, alpha, beta, opp)
                    undo_move(search_board, mv, me)
                    
                    if val > best_val:
                        best_val = val
                        current_best_move = mv
                    if val > alpha:
                        alpha = val
                
                best_move = current_best_move
                self.last_depth = depth_limit
        except TimeoutError:
            pass

        return best_move if best_move else moves[0]


class AlphaBetaBitboardStrategy(AlphaBetaImprovedStrategy):
    """Alpha-Beta strategy using bitboards for performance.
    Assumes the standard 88-square board structure defined in reversi.py.
    """
    name = "ab_bit"

    def __init__(self, depth=100, max_time=3.0):
        super().__init__(depth=depth, max_time=max_time)
        # Precompute bitboard constants
        self.ROW_OFFSETS = [3, 2, 1, 0, 0, 1, 2, 3]
        self.WIDTH = 14
        self.HEIGHT = 8
        self.TOTAL_BITS = self.WIDTH * self.HEIGHT
        
        # Valid mask (1 where board exists)
        self.VALID_MASK = 0
        self.ROW_MASKS = []
        for r in range(self.HEIGHT):
            row_len = [8, 10, 12, 14, 14, 12, 10, 8][r]
            offset = self.ROW_OFFSETS[r]
            row_mask = 0
            for c in range(row_len):
                idx = r * self.WIDTH + (c + offset)
                self.VALID_MASK |= (1 << idx)
                row_mask |= (1 << idx)
            self.ROW_MASKS.append(row_mask)

        # Directions shifts in the aligned 8x14 grid
        # (dr, dc) -> shift
        # (0, 1) -> +1
        # (0, -1) -> -1
        # (1, 0) -> +14
        # (-1, 0) -> -14
        # (1, 1) -> +15
        # (1, -1) -> +13
        # (-1, 1) -> -13
        # (-1, -1) -> -15
        self.SHIFTS = [1, -1, 14, -14, 15, 13, -13, -15]

        # Precompute evaluation masks
        self.CORNER_MASK = 0
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        for r, c in corners:
            idx = r * self.WIDTH + (c + self.ROW_OFFSETS[r])
            self.CORNER_MASK |= (1 << idx)

        self.EDGE_MASK = 0
        for r in range(self.HEIGHT):
            row_len = [8, 10, 12, 14, 14, 12, 10, 8][r]
            offset = self.ROW_OFFSETS[r]
            for c in range(row_len):
                if r == 0 or r == 7 or c == 0 or c == row_len - 1:
                    idx = r * self.WIDTH + (c + offset)
                    self.EDGE_MASK |= (1 << idx)
        
        # Corner adjacent mask
        self.CORNER_ADJ_MASK = 0
        # Map corner index to its adjacent indices
        self.CORNER_TO_ADJ = {} 
        # Map adjacent index to corner index (for penalty check)
        self.ADJ_TO_CORNER = {}
        
        for r, c in corners:
            c_idx = r * self.WIDTH + (c + self.ROW_OFFSETS[r])
            adj_indices = []
            for shift in self.SHIFTS:
                adj = c_idx + shift
                if 0 <= adj < self.TOTAL_BITS and (self.VALID_MASK & (1 << adj)):
                    self.CORNER_ADJ_MASK |= (1 << adj)
                    adj_indices.append(adj)
                    self.ADJ_TO_CORNER[adj] = c_idx
            self.CORNER_TO_ADJ[c_idx] = adj_indices

    def to_bitboard(self, board, me, opp):
        me_bb = 0
        opp_bb = 0
        for r in range(len(board)):
            offset = self.ROW_OFFSETS[r]
            for c in range(len(board[r])):
                val = board[r][c]
                if val == 0: continue
                idx = r * self.WIDTH + (c + offset)
                if val == me:
                    me_bb |= (1 << idx)
                elif val == opp:
                    opp_bb |= (1 << idx)
        return me_bb, opp_bb

    def get_moves_bb(self, me_bb, opp_bb):
        empty = self.VALID_MASK & ~(me_bb | opp_bb)
        moves = 0
        
        # For each direction, find candidates
        for shift in self.SHIFTS:
            candidates = 0
            # Shift my pieces to potential opponent pieces
            if shift > 0:
                mask = (me_bb << shift) & opp_bb
                # Keep shifting while we see opponent pieces
                while mask:
                    candidates |= mask
                    mask = (mask << shift) & opp_bb
                # One more shift to land on empty
                moves |= (candidates << shift) & empty
            else:
                s = -shift
                mask = (me_bb >> s) & opp_bb
                while mask:
                    candidates |= mask
                    mask = (mask >> s) & opp_bb
                moves |= (candidates >> s) & empty
                
        # Filter out moves that wrapped around rows (if any)
        # Since we use 14 width and rows are separated by invalid cells (padding),
        # horizontal wraps are blocked by 0s in the padding if we are careful.
        # However, our VALID_MASK handles the shape.
        # But a shift could jump from end of row R to start of row R+1 if we are not careful.
        # The padding in 8x14 grid:
        # Row 0: 3 pad, 8 valid, 3 pad.
        # Row 1: 2 pad, 10 valid, 2 pad.
        # ...
        # A horizontal shift (+1) from the last valid cell of a row lands in padding (0).
        # Since opp_bb has 0 in padding, the propagation stops.
        # So wrapping is naturally handled by the padding zeros!
        
        return moves

    def get_flips_bb(self, move_idx, me_bb, opp_bb):
        flips = 0
        move_bit = (1 << move_idx)
        for shift in self.SHIFTS:
            potential_flips = 0
            mask = move_bit
            while True:
                if shift > 0:
                    mask = (mask << shift)
                else:
                    mask = (mask >> -shift)
                
                if not (mask & self.VALID_MASK):
                    break
                
                if mask & opp_bb:
                    potential_flips |= mask
                elif mask & me_bb:
                    flips |= potential_flips
                    break
                else:
                    break
        return flips

    def evaluate_bb(self, me_bb, opp_bb):
        # Disk count
        my_count = bin(me_bb).count('1')
        opp_count = bin(opp_bb).count('1')
        score = (my_count - opp_count) * 10

        # Corners
        my_corners = me_bb & self.CORNER_MASK
        opp_corners = opp_bb & self.CORNER_MASK
        score += bin(my_corners).count('1') * 1000
        score -= bin(opp_corners).count('1') * 1000

        # Edges
        my_edges = me_bb & self.EDGE_MASK
        opp_edges = opp_bb & self.EDGE_MASK
        score += bin(my_edges).count('1') * 50
        score -= bin(opp_edges).count('1') * 50

        # Corner adjacent penalty
        # If I have a piece adjacent to a corner, and that corner is empty -> penalty
        empty_corners = self.CORNER_MASK & ~(me_bb | opp_bb)
        if empty_corners:
            # Iterate corners to check adjacency
            for c_idx, adj_list in self.CORNER_TO_ADJ.items():
                if (empty_corners & (1 << c_idx)):
                    # This corner is empty. Check my pieces around it
                    for adj in adj_list:
                        if (me_bb & (1 << adj)):
                            score -= 800
                        elif (opp_bb & (1 << adj)):
                            score += 800

        # Mobility
        my_moves = self.get_moves_bb(me_bb, opp_bb)
        opp_moves = self.get_moves_bb(opp_bb, me_bb)
        score += (bin(my_moves).count('1') - bin(opp_moves).count('1')) * 20
        
        return score

    def choose_move(self, board, me=1, opp=2):
        reversi.board_global = board
        # Get legal moves for final return value matching
        legal_moves_list = reversi.get_legal_moves(board, me, opp)
        if not legal_moves_list:
            return None

        start_time = time.time()
        
        # Convert to bitboards
        me_bb, opp_bb = self.to_bitboard(board, me, opp)
        
        best_move_idx = None

        def get_move_score(idx, my_b, op_b):
            mask = (1 << idx)
            s = 0
            
            # Corner
            if mask & self.CORNER_MASK:
                s += 10000
            # Edge
            elif mask & self.EDGE_MASK:
                s += 200
                
            # Flips (expensive? relative to deepcopy it is cheap)
            flips = self.get_flips_bb(idx, my_b, op_b)
            s += bin(flips).count('1') * 10
            
            # Corner Adjacency Penalty
            if mask & self.CORNER_ADJ_MASK:
                c_idx = self.ADJ_TO_CORNER.get(idx)
                if c_idx is not None:
                    # Check if that corner is empty
                    if not ((my_b | op_b) & (1 << c_idx)):
                        s -= 800
            
            # Tie-breaker (prefer top-left)
            # idx is roughly r*WIDTH + c. Higher idx is lower on board.
            # We want to subtract a small amount for higher idx.
            s -= idx * 0.001
            return s

        def alphabeta_bb(my_b, op_b, depth, alpha, beta, maximizing):
            if self.max_time and self.max_time > 0:
                if time.time() - start_time > self.max_time:
                    raise TimeoutError()

            moves_mask = self.get_moves_bb(my_b, op_b)
            
            if depth == 0 or moves_mask == 0:
                if moves_mask == 0:
                    # Check if opponent has moves
                    opp_moves_mask = self.get_moves_bb(op_b, my_b)
                    if opp_moves_mask == 0:
                        # Game over
                        return self.evaluate_bb(my_b if maximizing else op_b, op_b if maximizing else my_b)
                    # Pass
                    return alphabeta_bb(op_b, my_b, depth, alpha, beta, not maximizing)
                
                return self.evaluate_bb(my_b if maximizing else op_b, op_b if maximizing else my_b)

            # Extract moves from mask
            move_indices = []
            temp_mask = moves_mask
            while temp_mask:
                # Get lowest set bit
                lsb = temp_mask & -temp_mask
                idx = lsb.bit_length() - 1
                move_indices.append(idx)
                temp_mask ^= lsb

            # Sort moves for better pruning
            move_indices.sort(key=lambda idx: get_move_score(idx, my_b, op_b), reverse=True)
            
            if maximizing:
                value = -10**9
                for idx in move_indices:
                    flips = self.get_flips_bb(idx, my_b, op_b)
                    new_my = my_b | (1 << idx) | flips
                    new_op = op_b & ~flips
                    
                    val = alphabeta_bb(new_op, new_my, depth - 1, alpha, beta, False)
                    if val > value:
                        value = val
                    if value > alpha:
                        alpha = value
                    if alpha >= beta:
                        break
                return value
            else:
                value = 10**9
                for idx in move_indices:
                    flips = self.get_flips_bb(idx, my_b, op_b)
                    new_my = my_b | (1 << idx) | flips
                    new_op = op_b & ~flips
                    
                    val = alphabeta_bb(new_op, new_my, depth - 1, alpha, beta, True)
                    if val < value:
                        value = val
                    if value < beta:
                        beta = value
                    if alpha >= beta:
                        break
                return value

        try:
            # Calculate max possible depth based on empty squares
            empty_squares = sum(row.count(0) for row in board)
            effective_max_depth = min(self.depth, empty_squares)

            for depth_limit in range(1, max(1, effective_max_depth) + 1):
                moves_mask = self.get_moves_bb(me_bb, opp_bb)
                if not moves_mask:
                    break
                    
                move_indices = []
                temp_mask = moves_mask
                while temp_mask:
                    lsb = temp_mask & -temp_mask
                    idx = lsb.bit_length() - 1
                    move_indices.append(idx)
                    temp_mask ^= lsb
                
                move_indices.sort(key=lambda idx: get_move_score(idx, me_bb, opp_bb), reverse=True)

                current_best = None
                best_val = -10**9
                alpha = -10**9
                beta = 10**9
                
                for idx in move_indices:
                    flips = self.get_flips_bb(idx, me_bb, opp_bb)
                    new_my = me_bb | (1 << idx) | flips
                    new_op = opp_bb & ~flips
                    
                    val = alphabeta_bb(new_op, new_my, depth_limit - 1, alpha, beta, False)
                    
                    if val > best_val:
                        best_val = val
                        current_best = idx
                    if val > alpha:
                        alpha = val
                
                best_move_idx = current_best
                self.last_depth = depth_limit
                
        except TimeoutError:
            pass

        if best_move_idx is not None:
            # Convert back to (r, c)
            r = best_move_idx // self.WIDTH
            c_aligned = best_move_idx % self.WIDTH
            c = c_aligned - self.ROW_OFFSETS[r]
            
            # Find matching move in legal_moves_list to return full object
            for m in legal_moves_list:
                if m[0] == r and m[1] == c:
                    return m
            
        return legal_moves_list[0]


class AlphaBetaTTStrategy(AlphaBetaImprovedStrategy):
    """Alpha-Beta strategy with Transposition Table.
    Based on AlphaBetaImprovedStrategy (ab2).
    """
    name = "ab_tt"

    def __init__(self, depth=100, max_time=3.0):
        super().__init__(depth=depth, max_time=max_time)
        self.tt = {}  # Transposition table: hash -> (depth, value, flag)

    def choose_move(self, board, me=1, opp=2):
        reversi.board_global = board
        moves = reversi.get_legal_moves(board, me, opp)
        if not moves:
            return None

        # Calculate max possible depth based on empty squares
        empty_squares = sum(row.count(0) for row in board)
        effective_max_depth = min(self.depth, empty_squares)

        start_time = time.time()
        best_move = None
        
        # TT flags
        EXACT = 0
        LOWERBOUND = 1
        UPPERBOUND = 2

        def board_to_hash(b):
            # Convert list of lists to tuple of tuples for hashing
            return tuple(tuple(row) for row in b)

        def alphabeta_tt(b, depth, alpha, beta, current_player):
            if self.max_time and self.max_time > 0:
                if time.time() - start_time > self.max_time:
                    raise TimeoutError()

            alpha_orig = alpha
            b_hash = board_to_hash(b)
            
            # TT Lookup
            tt_entry = self.tt.get(b_hash)
            if tt_entry:
                tt_depth, tt_value, tt_flag = tt_entry
                if tt_depth >= depth:
                    if tt_flag == EXACT:
                        return tt_value
                    elif tt_flag == LOWERBOUND:
                        alpha = max(alpha, tt_value)
                    elif tt_flag == UPPERBOUND:
                        beta = min(beta, tt_value)
                    
                    if alpha >= beta:
                        return tt_value

            other = 1 if current_player == 2 else 2
            reversi.board_global = b
            moves_here = reversi.get_legal_moves(b, current_player, other)
            
            if depth == 0 or not moves_here:
                if not moves_here:
                    moves_other = reversi.get_legal_moves(b, other, current_player)
                    if not moves_other:
                        val = self.evaluate(b, me, opp)
                        # Store exact value for terminal node
                        self.tt[b_hash] = (depth, val, EXACT)
                        return val
                    # Pass
                    val = alphabeta_tt(b, depth, alpha, beta, other)
                    # Store whatever we got back
                    # Note: Passing doesn't change board, but depth might be different?
                    # Actually, passing preserves board state but changes player turn.
                    # Our hash only includes board, not player. 
                    # This is a potential issue if we don't include player in hash.
                    # But usually minimax alternates. 
                    # Let's include current_player in hash to be safe.
                    return val
                
                val = self.evaluate(b, me, opp)
                self.tt[b_hash] = (depth, val, EXACT)
                return val

            # order moves by greedy heuristic
            moves_here = sorted(moves_here, key=lambda mv: reversi.score_move(b, mv, current_player, other), reverse=True)

            if current_player == me:
                value = -10**9
                for mv in moves_here:
                    nb = _deepcopy(b)
                    engine.apply_move(nb, mv, current_player)
                    val = alphabeta_tt(nb, depth - 1, alpha, beta, other)
                    if val > value:
                        value = val
                    if value > alpha:
                        alpha = value
                    if alpha >= beta:
                        break
            else:
                value = 10**9
                for mv in moves_here:
                    nb = _deepcopy(b)
                    engine.apply_move(nb, mv, current_player)
                    val = alphabeta_tt(nb, depth - 1, alpha, beta, other)
                    if val < value:
                        value = val
                    if value < beta:
                        beta = value
                    if alpha >= beta:
                        break
            
            # TT Store
            tt_flag = EXACT
            if value <= alpha_orig:
                tt_flag = UPPERBOUND
            elif value >= beta:
                tt_flag = LOWERBOUND
            
            # We need to include player in hash key if we want to be correct for passes
            # But for now let's stick to board hash as per standard simple implementations,
            # assuming player is implicit from board state (count of pieces) usually,
            # but in Reversi passes make it ambiguous.
            # Let's update the hash function to include player.
            self.tt[b_hash] = (depth, value, tt_flag)
            
            return value

        # Redefine hash to include player
        def board_player_hash(b, p):
            return (tuple(tuple(row) for row in b), p)

        # Update the inner function to use the new hash
        def alphabeta_tt_fixed(b, depth, alpha, beta, current_player):
            if self.max_time and self.max_time > 0:
                if time.time() - start_time > self.max_time:
                    raise TimeoutError()

            alpha_orig = alpha
            bp_hash = board_player_hash(b, current_player)
            
            tt_entry = self.tt.get(bp_hash)
            if tt_entry:
                tt_depth, tt_value, tt_flag = tt_entry
                if tt_depth >= depth:
                    if tt_flag == EXACT:
                        return tt_value
                    elif tt_flag == LOWERBOUND:
                        alpha = max(alpha, tt_value)
                    elif tt_flag == UPPERBOUND:
                        beta = min(beta, tt_value)
                    
                    if alpha >= beta:
                        return tt_value

            other = 1 if current_player == 2 else 2
            reversi.board_global = b
            moves_here = reversi.get_legal_moves(b, current_player, other)
            
            if depth == 0 or not moves_here:
                if not moves_here:
                    moves_other = reversi.get_legal_moves(b, other, current_player)
                    if not moves_other:
                        val = self.evaluate(b, me, opp)
                        self.tt[bp_hash] = (1000, val, EXACT) # Terminal
                        return val
                    val = alphabeta_tt_fixed(b, depth, alpha, beta, other)
                    return val
                
                val = self.evaluate(b, me, opp)
                self.tt[bp_hash] = (depth, val, EXACT)
                return val

            moves_here = sorted(moves_here, key=lambda mv: reversi.score_move(b, mv, current_player, other), reverse=True)

            if current_player == me:
                value = -10**9
                for mv in moves_here:
                    nb = _deepcopy(b)
                    engine.apply_move(nb, mv, current_player)
                    val = alphabeta_tt_fixed(nb, depth - 1, alpha, beta, other)
                    if val > value:
                        value = val
                    if value > alpha:
                        alpha = value
                    if alpha >= beta:
                        break
            else:
                value = 10**9
                for mv in moves_here:
                    nb = _deepcopy(b)
                    engine.apply_move(nb, mv, current_player)
                    val = alphabeta_tt_fixed(nb, depth - 1, alpha, beta, other)
                    if val < value:
                        value = val
                    if value < beta:
                        beta = value
                    if alpha >= beta:
                        break
            
            tt_flag = EXACT
            if value <= alpha_orig:
                tt_flag = UPPERBOUND
            elif value >= beta:
                tt_flag = LOWERBOUND
            
            self.tt[bp_hash] = (depth, value, tt_flag)
            return value

        try:
            for depth_limit in range(1, max(1, effective_max_depth) + 1):
                ordered_moves = sorted(reversi.get_legal_moves(board, me, opp), key=lambda mv: reversi.score_move(board, mv, me, opp), reverse=True)
                current_best_move = None
                best_val = -10**9
                alpha = -10**9
                beta = 10**9
                
                for mv in ordered_moves:
                    nb = _deepcopy(board)
                    engine.apply_move(nb, mv, me)
                    val = alphabeta_tt_fixed(nb, depth_limit - 1, alpha, beta, opp)
                    if val > best_val:
                        best_val = val
                        current_best_move = mv
                    if val > alpha:
                        alpha = val
                
                best_move = current_best_move
                self.last_depth = depth_limit
        except TimeoutError:
            pass

        return best_move if best_move else moves[0]


class AlphaBetaBitboardTTStrategy(AlphaBetaBitboardStrategy):
    """Alpha-Beta strategy using bitboards AND Transposition Table.
    """
    name = "ab_bit_tt"

    def __init__(self, depth=100, max_time=3.0):
        super().__init__(depth=depth, max_time=max_time)
        self.tt = {}  # (me_bb, opp_bb) -> (depth, value, flag)

    def choose_move(self, board, me=1, opp=2):
        reversi.board_global = board
        legal_moves_list = reversi.get_legal_moves(board, me, opp)
        if not legal_moves_list:
            return None

        start_time = time.time()
        me_bb, opp_bb = self.to_bitboard(board, me, opp)
        best_move_idx = None
        
        # TT flags
        EXACT = 0
        LOWERBOUND = 1
        UPPERBOUND = 2

        def get_move_score(idx, my_b, op_b):
            mask = (1 << idx)
            s = 0
            if mask & self.CORNER_MASK: s += 10000
            elif mask & self.EDGE_MASK: s += 200
            flips = self.get_flips_bb(idx, my_b, op_b)
            s += bin(flips).count('1') * 10
            if mask & self.CORNER_ADJ_MASK:
                c_idx = self.ADJ_TO_CORNER.get(idx)
                if c_idx is not None:
                    if not ((my_b | op_b) & (1 << c_idx)):
                        s -= 800
            s -= idx * 0.001
            return s

        def alphabeta_bb_tt(my_b, op_b, depth, alpha, beta, maximizing):
            if self.max_time and self.max_time > 0:
                if time.time() - start_time > self.max_time:
                    raise TimeoutError()

            alpha_orig = alpha
            
            # TT Lookup
            tt_key = (my_b, op_b)
            tt_entry = self.tt.get(tt_key)
            if tt_entry:
                tt_depth, tt_value, tt_flag = tt_entry
                if tt_depth >= depth:
                    if tt_flag == EXACT:
                        return tt_value
                    elif tt_flag == LOWERBOUND:
                        alpha = max(alpha, tt_value)
                    elif tt_flag == UPPERBOUND:
                        beta = min(beta, tt_value)
                    if alpha >= beta:
                        return tt_value

            moves_mask = self.get_moves_bb(my_b, op_b)
            
            if depth == 0 or moves_mask == 0:
                if moves_mask == 0:
                    opp_moves_mask = self.get_moves_bb(op_b, my_b)
                    if opp_moves_mask == 0:
                        val = self.evaluate_bb(my_b if maximizing else op_b, op_b if maximizing else my_b)
                        self.tt[tt_key] = (1000, val, EXACT)
                        return val
                    val = alphabeta_bb_tt(op_b, my_b, depth, alpha, beta, not maximizing)
                    return val
                
                val = self.evaluate_bb(my_b if maximizing else op_b, op_b if maximizing else my_b)
                self.tt[tt_key] = (depth, val, EXACT)
                return val

            move_indices = []
            temp_mask = moves_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                idx = lsb.bit_length() - 1
                move_indices.append(idx)
                temp_mask ^= lsb

            move_indices.sort(key=lambda idx: get_move_score(idx, my_b, op_b), reverse=True)
            
            if maximizing:
                value = -10**9
                for idx in move_indices:
                    flips = self.get_flips_bb(idx, my_b, op_b)
                    new_my = my_b | (1 << idx) | flips
                    new_op = op_b & ~flips
                    
                    val = alphabeta_bb_tt(new_op, new_my, depth - 1, alpha, beta, False)
                    if val > value:
                        value = val
                    if value > alpha:
                        alpha = value
                    if alpha >= beta:
                        break
            else:
                value = 10**9
                for idx in move_indices:
                    flips = self.get_flips_bb(idx, my_b, op_b)
                    new_my = my_b | (1 << idx) | flips
                    new_op = op_b & ~flips
                    
                    val = alphabeta_bb_tt(new_op, new_my, depth - 1, alpha, beta, True)
                    if val < value:
                        value = val
                    if value < beta:
                        beta = value
                    if alpha >= beta:
                        break
            
            tt_flag = EXACT
            if value <= alpha_orig:
                tt_flag = UPPERBOUND
            elif value >= beta:
                tt_flag = LOWERBOUND
            
            self.tt[tt_key] = (depth, value, tt_flag)
            return value

        try:
            # Calculate max possible depth based on empty squares
            empty_squares = sum(row.count(0) for row in board)
            effective_max_depth = min(self.depth, empty_squares)

            for depth_limit in range(1, max(1, effective_max_depth) + 1):
                moves_mask = self.get_moves_bb(me_bb, opp_bb)
                if not moves_mask:
                    break
                    
                move_indices = []
                temp_mask = moves_mask
                while temp_mask:
                    lsb = temp_mask & -temp_mask
                    idx = lsb.bit_length() - 1
                    move_indices.append(idx)
                    temp_mask ^= lsb
                
                move_indices.sort(key=lambda idx: get_move_score(idx, me_bb, opp_bb), reverse=True)

                current_best = None
                best_val = -10**9
                alpha = -10**9
                beta = 10**9
                
                for idx in move_indices:
                    flips = self.get_flips_bb(idx, me_bb, opp_bb)
                    new_my = me_bb | (1 << idx) | flips
                    new_op = opp_bb & ~flips
                    
                    val = alphabeta_bb_tt(new_op, new_my, depth_limit - 1, alpha, beta, False)
                    
                    if val > best_val:
                        best_val = val
                        current_best = idx
                    if val > alpha:
                        alpha = val
                
                best_move_idx = current_best
                self.last_depth = depth_limit
                
        except TimeoutError:
            pass

        if best_move_idx is not None:
            r = best_move_idx // self.WIDTH
            c_aligned = best_move_idx % self.WIDTH
            c = c_aligned - self.ROW_OFFSETS[r]
            for m in legal_moves_list:
                if m[0] == r and m[1] == c:
                    return m
            
        return legal_moves_list[0]
