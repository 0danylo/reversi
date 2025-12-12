"""
Triangle Reversi Ultimate Bot
Combines all best techniques:
- Bitboard representation with optimized shifts
- Alpha-Beta with Negamax
- Transposition Table with exact/lower/upper bounds
- Iterative Deepening with time management
- Principal Variation Search (PVS/NegaScout)
- Killer Move Heuristic
- Endgame Solver (exact when <= 12 empty)
- Dynamic corner adjacency penalties
- Frontier disc penalties
- Stability estimation
- Enhanced move ordering
"""

import sys
import time
from collections import OrderedDict

# Pre-computed bit counts for faster popcount
BIT_COUNT = [bin(i).count('1') for i in range(256)]

def popcount(n):
    """Fast population count using lookup table."""
    count = 0
    while n:
        count += BIT_COUNT[n & 0xFF]
        n >>= 8
    return count

class TriangleReversiBot:
    # TT flags
    EXACT = 0
    LOWERBOUND = 1
    UPPERBOUND = 2
    
    def __init__(self, depth, max_time, max_tt_size=200000):
        self.start_time = time.time()
        self.depth = depth
        self.max_time = max_time
        self.max_tt_size = max_tt_size
        
        self.tt = OrderedDict()
        self.nodes_searched = 0
        
        # Killer moves: 2 killers per depth level
        self.killer_moves = [[None, None] for _ in range(64)]
        
        # Board Constants (10 rows, triangle shape)
        self.ROWS = 10
        self.OFFSETS = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        self.LENGTHS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        self.WIDTH = 22
        self.TOTAL_BITS = self.ROWS * self.WIDTH
        
        # Precompute Masks
        self.VALID_MASK = 0
        self.ROW_MASKS = []
        for r in range(self.ROWS):
            row_mask = 0
            for c in range(self.OFFSETS[r], self.OFFSETS[r] + self.LENGTHS[r]):
                idx = r * self.WIDTH + c
                self.VALID_MASK |= (1 << idx)
                row_mask |= (1 << idx)
            self.ROW_MASKS.append(row_mask)
            
        # Directions: E, W, S, N, SE, SW, NE, NW
        self.SHIFTS = [1, -1, self.WIDTH, -self.WIDTH, 
                       self.WIDTH+1, self.WIDTH-1, -self.WIDTH+1, -self.WIDTH-1]
        
        # Key positions
        self.CORNERS = [(9, 0), (9, 19)]
        self.TOP_CORNERS = [(0, 9), (0, 10)]
        
        # Evaluation Weights (tuned)
        self.W_CORNER = 800
        self.W_TOP_CORNER = 120
        self.W_EDGE = 30
        self.W_BAD = -100
        self.W_NORMAL = 10
        self.W_MOBILITY = 25
        self.W_FRONTIER = -8
        self.W_STABILITY = 50
        self.W_PARITY = 15
        
        # Corner adjacency mapping
        self.CORNER_TO_ADJ = {}
        self.ADJ_TO_CORNER = {}
        
        # Bottom Left (9, 0)
        c_idx_bl = 9 * self.WIDTH + 0
        adj_bl = [9 * self.WIDTH + 1, 8 * self.WIDTH + 1]
        self.CORNER_TO_ADJ[c_idx_bl] = adj_bl
        for adj in adj_bl:
            self.ADJ_TO_CORNER[adj] = c_idx_bl
        
        # Bottom Right (9, 19)
        c_idx_br = 9 * self.WIDTH + 19
        adj_br = [9 * self.WIDTH + 18, 8 * self.WIDTH + 18]
        self.CORNER_TO_ADJ[c_idx_br] = adj_br
        for adj in adj_br:
            self.ADJ_TO_CORNER[adj] = c_idx_br

        # Build masks
        self.MASK_CORNER = 0
        self.MASK_TOP_CORNER = 0
        self.MASK_EDGE = 0
        self.MASK_NORMAL = 0
        self.CORNER_ADJ_MASK = 0
        
        for r in range(self.ROWS):
            for c in range(self.OFFSETS[r], self.OFFSETS[r] + self.LENGTHS[r]):
                idx = r * self.WIDTH + c
                
                if (r, c) in self.CORNERS:
                    self.MASK_CORNER |= (1 << idx)
                elif (r, c) in self.TOP_CORNERS:
                    self.MASK_TOP_CORNER |= (1 << idx)
                elif c == self.OFFSETS[r] or c == self.OFFSETS[r] + self.LENGTHS[r] - 1 or r == 9:
                    self.MASK_EDGE |= (1 << idx)
                else:
                    self.MASK_NORMAL |= (1 << idx)
        
        for adj_list in self.CORNER_TO_ADJ.values():
            for adj in adj_list:
                self.CORNER_ADJ_MASK |= (1 << adj)
        
        # Bottom row mask for stability
        self.BOTTOM_ROW_MASK = self.ROW_MASKS[9]
        
        # Precompute position values for move ordering
        self.POS_VALUES = {}
        for r in range(self.ROWS):
            for c in range(self.OFFSETS[r], self.OFFSETS[r] + self.LENGTHS[r]):
                idx = r * self.WIDTH + c
                if (r, c) in self.CORNERS:
                    self.POS_VALUES[idx] = 10000
                elif (r, c) in self.TOP_CORNERS:
                    self.POS_VALUES[idx] = 500
                elif c == self.OFFSETS[r] or c == self.OFFSETS[r] + self.LENGTHS[r] - 1 or r == 9:
                    self.POS_VALUES[idx] = 100
                else:
                    self.POS_VALUES[idx] = 10

    def to_bitboard(self, board_input, me, opp):
        me_bb = 0
        opp_bb = 0
        for r in range(self.ROWS):
            row_vals = board_input[r]
            offset = self.OFFSETS[r]
            for i, val in enumerate(row_vals):
                if val == 0: 
                    continue
                c = offset + i
                idx = r * self.WIDTH + c
                if val == me:
                    me_bb |= (1 << idx)
                elif val == opp:
                    opp_bb |= (1 << idx)
        return me_bb, opp_bb

    def get_moves_bb(self, me_bb, opp_bb):
        empty = self.VALID_MASK & ~(me_bb | opp_bb)
        moves = 0
        for shift in self.SHIFTS:
            candidates = 0
            if shift > 0:
                mask = (me_bb << shift) & opp_bb
                while mask:
                    candidates |= mask
                    mask = (mask << shift) & opp_bb
                moves |= (candidates << shift) & empty
            else:
                s = -shift
                mask = (me_bb >> s) & opp_bb
                while mask:
                    candidates |= mask
                    mask = (mask >> s) & opp_bb
                moves |= (candidates >> s) & empty
        return moves

    def get_flips_bb(self, move_idx, me_bb, opp_bb):
        flips = 0
        move_bit = 1 << move_idx
        for shift in self.SHIFTS:
            potential_flips = 0
            mask = move_bit
            while True:
                if shift > 0:
                    mask = mask << shift
                else:
                    mask = mask >> -shift
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

    def count_stable_discs(self, me_bb, opp_bb):
        """Count approximately stable discs on bottom row."""
        occupied = me_bb | opp_bb
        stable = 0
        
        # Check bottom corners and propagate stability along edges
        # Left corner
        c_left = 9 * self.WIDTH + 0
        if me_bb & (1 << c_left):
            stable += 1
            # Check along bottom row
            for c in range(1, 20):
                idx = 9 * self.WIDTH + c
                if me_bb & (1 << idx):
                    stable += 1
                else:
                    break
        
        # Right corner
        c_right = 9 * self.WIDTH + 19
        if me_bb & (1 << c_right):
            stable += 1
            # Check along bottom row (backwards)
            for c in range(18, -1, -1):
                idx = 9 * self.WIDTH + c
                if me_bb & (1 << idx):
                    stable += 1
                else:
                    break
        
        return stable

    def evaluate(self, me_bb, opp_bb, empty_count):
        """Comprehensive evaluation function."""
        
        # Terminal state: just count discs
        if empty_count == 0:
            diff = popcount(me_bb) - popcount(opp_bb)
            return 100000 * (1 if diff > 0 else -1 if diff < 0 else 0)
        
        score = 0
        
        # === Corners (extremely valuable) ===
        my_corners = popcount(me_bb & self.MASK_CORNER)
        opp_corners = popcount(opp_bb & self.MASK_CORNER)
        score += (my_corners - opp_corners) * self.W_CORNER
        
        # === Top Corners ===
        my_top = popcount(me_bb & self.MASK_TOP_CORNER)
        opp_top = popcount(opp_bb & self.MASK_TOP_CORNER)
        score += (my_top - opp_top) * self.W_TOP_CORNER
        
        # === Edges ===
        my_edges = popcount(me_bb & self.MASK_EDGE)
        opp_edges = popcount(opp_bb & self.MASK_EDGE)
        score += (my_edges - opp_edges) * self.W_EDGE
        
        # === Bad squares (dynamic check) ===
        occupied = me_bb | opp_bb
        for c_idx, adj_list in self.CORNER_TO_ADJ.items():
            if not (occupied & (1 << c_idx)):  # Corner is empty
                for adj in adj_list:
                    if me_bb & (1 << adj):
                        score += self.W_BAD
                    if opp_bb & (1 << adj):
                        score -= self.W_BAD
        
        # === Normal squares ===
        my_normal = popcount(me_bb & self.MASK_NORMAL)
        opp_normal = popcount(opp_bb & self.MASK_NORMAL)
        score += (my_normal - opp_normal) * self.W_NORMAL
        
        # === Mobility ===
        my_moves = self.get_moves_bb(me_bb, opp_bb)
        opp_moves = self.get_moves_bb(opp_bb, me_bb)
        my_mob = popcount(my_moves)
        opp_mob = popcount(opp_moves)
        score += (my_mob - opp_mob) * self.W_MOBILITY
        
        # === Frontier Discs ===
        empty = self.VALID_MASK & ~occupied
        frontier = 0
        for shift in self.SHIFTS:
            if shift > 0:
                frontier |= (empty << shift)
            else:
                frontier |= (empty >> -shift)
        frontier &= self.VALID_MASK
        
        my_frontier = popcount(me_bb & frontier)
        opp_frontier = popcount(opp_bb & frontier)
        score += (my_frontier - opp_frontier) * self.W_FRONTIER
        
        # === Stability (approximate) ===
        my_stable = self.count_stable_discs(me_bb, opp_bb)
        opp_stable = self.count_stable_discs(opp_bb, me_bb)
        score += (my_stable - opp_stable) * self.W_STABILITY
        
        # === Parity (who plays last) ===
        if empty_count <= 10:
            # Odd parity is good (we want to play last)
            if empty_count % 2 == 1:
                score += self.W_PARITY
            else:
                score -= self.W_PARITY
        
        return score

    def get_move_score(self, idx, me_bb, opp_bb, ply):
        """Score a move for ordering purposes."""
        score = self.POS_VALUES.get(idx, 10)
        
        # Killer move bonus
        if self.killer_moves[ply][0] == idx:
            score += 500
        elif self.killer_moves[ply][1] == idx:
            score += 400
        
        # Corner adjacency penalty (dynamic)
        if idx in self.ADJ_TO_CORNER:
            c_idx = self.ADJ_TO_CORNER[idx]
            if not ((me_bb | opp_bb) & (1 << c_idx)):
                score -= 800
        
        # Flip count bonus
        flips = self.get_flips_bb(idx, me_bb, opp_bb)
        score += popcount(flips) * 5
        
        return score

    def extract_moves(self, moves_mask):
        """Extract move indices from bitmask."""
        moves = []
        temp = moves_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            moves.append(idx)
            temp ^= lsb
        return moves

    def store_killer(self, move, ply):
        """Store a killer move."""
        if self.killer_moves[ply][0] != move:
            self.killer_moves[ply][1] = self.killer_moves[ply][0]
            self.killer_moves[ply][0] = move

    def negamax(self, me_bb, opp_bb, depth, alpha, beta, ply, empty_count):
        """Negamax with alpha-beta pruning and PVS."""
        self.nodes_searched += 1
        
        # Time check
        if time.time() - self.start_time > self.max_time:
            raise TimeoutError()
        
        alpha_orig = alpha
        
        # TT lookup
        tt_key = (me_bb, opp_bb)
        tt_entry = self.tt.get(tt_key)
        if tt_entry:
            self.tt.move_to_end(tt_key)
            tt_depth, tt_value, tt_flag, tt_best = tt_entry
            if tt_depth >= depth:
                if tt_flag == self.EXACT:
                    return tt_value
                elif tt_flag == self.LOWERBOUND:
                    alpha = max(alpha, tt_value)
                elif tt_flag == self.UPPERBOUND:
                    beta = min(beta, tt_value)
                if alpha >= beta:
                    return tt_value
        
        moves_mask = self.get_moves_bb(me_bb, opp_bb)
        
        if not moves_mask:
            opp_moves = self.get_moves_bb(opp_bb, me_bb)
            if not opp_moves:
                # Game over
                diff = popcount(me_bb) - popcount(opp_bb)
                return 100000 if diff > 0 else -100000 if diff < 0 else 0
            # Pass
            return -self.negamax(opp_bb, me_bb, depth, -beta, -alpha, ply + 1, empty_count)
        
        if depth == 0:
            return self.evaluate(me_bb, opp_bb, empty_count)
        
        moves = self.extract_moves(moves_mask)
        
        # Move ordering
        # Priority: TT best move, then sorted by score
        tt_best_move = tt_entry[3] if tt_entry else None
        
        def sort_key(m):
            if m == tt_best_move:
                return 100000
            return self.get_move_score(m, me_bb, opp_bb, ply)
        
        moves.sort(key=sort_key, reverse=True)
        
        best_val = -float('inf')
        best_move = moves[0]
        
        for i, move in enumerate(moves):
            flips = self.get_flips_bb(move, me_bb, opp_bb)
            new_me = me_bb | (1 << move) | flips
            new_op = opp_bb & ~flips
            
            # PVS: search first move with full window, others with null window
            if i == 0:
                val = -self.negamax(new_op, new_me, depth - 1, -beta, -alpha, ply + 1, empty_count - 1)
            else:
                # Null window search
                val = -self.negamax(new_op, new_me, depth - 1, -alpha - 1, -alpha, ply + 1, empty_count - 1)
                if alpha < val < beta:
                    # Re-search with full window
                    val = -self.negamax(new_op, new_me, depth - 1, -beta, -val, ply + 1, empty_count - 1)
            
            if val > best_val:
                best_val = val
                best_move = move
            
            alpha = max(alpha, val)
            if alpha >= beta:
                self.store_killer(move, ply)
                break
        
        # Store in TT
        if best_val <= alpha_orig:
            flag = self.UPPERBOUND
        elif best_val >= beta:
            flag = self.LOWERBOUND
        else:
            flag = self.EXACT
        
        self.tt[tt_key] = (depth, best_val, flag, best_move)
        if len(self.tt) > self.max_tt_size:
            self.tt.popitem(last=False)
        
        return best_val

    def choose_move(self, board_input):
        self.start_time = time.time()
        self.nodes_searched = 0
        
        me_bb, opp_bb = self.to_bitboard(board_input, 1, 2)
        
        moves_mask = self.get_moves_bb(me_bb, opp_bb)
        if not moves_mask:
            return None
        
        total_discs = popcount(me_bb | opp_bb)
        empty_count = 110 - total_discs
        
        # Determine search depth
        search_depth = self.depth
        if empty_count <= 12:
            # Endgame: solve exactly
            search_depth = empty_count + 4
        elif empty_count <= 20:
            search_depth = min(14, empty_count + 2)
        
        moves = self.extract_moves(moves_mask)
        
        # Quick return for single move
        if len(moves) == 1:
            r = moves[0] // self.WIDTH
            c = moves[0] % self.WIDTH
            return r, c
        
        # Initial sort
        moves.sort(key=lambda m: self.get_move_score(m, me_bb, opp_bb, 0), reverse=True)
        
        best_move_idx = moves[0]
        
        # Iterative deepening
        try:
            for d in range(1, search_depth + 1):
                alpha = -float('inf')
                beta = float('inf')
                current_best_val = -float('inf')
                current_best_move = None
                
                for i, move in enumerate(moves):
                    if time.time() - self.start_time > self.max_time:
                        raise TimeoutError()
                    
                    flips = self.get_flips_bb(move, me_bb, opp_bb)
                    new_me = me_bb | (1 << move) | flips
                    new_op = opp_bb & ~flips
                    
                    if i == 0:
                        val = -self.negamax(new_op, new_me, d - 1, -beta, -alpha, 1, empty_count - 1)
                    else:
                        val = -self.negamax(new_op, new_me, d - 1, -alpha - 1, -alpha, 1, empty_count - 1)
                        if alpha < val < beta:
                            val = -self.negamax(new_op, new_me, d - 1, -beta, -val, 1, empty_count - 1)
                    
                    if val > current_best_val:
                        current_best_val = val
                        current_best_move = move
                    
                    alpha = max(alpha, val)
                
                if current_best_move is not None:
                    best_move_idx = current_best_move
                    # Move best to front
                    moves.remove(best_move_idx)
                    moves.insert(0, best_move_idx)
                    
        except TimeoutError:
            pass
        
        r = best_move_idx // self.WIDTH
        c = best_move_idx % self.WIDTH
        return r, c


def main():
    board_input = []
    try:
        for _ in range(10):
            line = sys.stdin.readline()
            if not line:
                break
            board_input.append(list(map(int, line.split())))
    except ValueError:
        pass
    
    if len(board_input) < 10:
        return
    
    bot = TriangleReversiBot(depth=100, max_time=2.85)
    move = bot.choose_move(board_input)
    
    if move:
        r, c = move[0], move[1]
        local_c = c - bot.OFFSETS[r]
        print(f"{r + 1} {local_c + 1}")  # Output 1-indexed row and local column
    else:
        print("0 0")


if __name__ == "__main__":
    main()
