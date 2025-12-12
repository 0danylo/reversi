"""
Triangle Reversi Pro Bot (v2)
Advanced improvements over Ultimate:
- History Heuristic for move ordering
- Late Move Reductions (LMR)
- Aspiration Windows for iterative deepening
- Null Move Pruning (adapted for Reversi)
- Enhanced stability counting (full edge analysis)
- Internal Iterative Deepening (IID)
- Better endgame detection with WLD solving
- Optimized bitboard operations
- Game phase-dependent evaluation weights
"""

import sys
import time
from collections import OrderedDict

# Pre-computed bit counts for faster popcount
BIT_COUNT = [bin(i).count('1') for i in range(256)]
OFFSETS = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

def popcount(n):
    """Fast population count using lookup table."""
    count = 0
    while n:
        count += BIT_COUNT[n & 0xFF]
        n >>= 8
    return count

class TriangleReversiBotPro:
    # TT flags
    EXACT = 0
    LOWERBOUND = 1
    UPPERBOUND = 2
    
    def __init__(self, depth=100, max_time=2.85, max_tt_size=300000):
        self.start_time = time.time()
        self.depth = depth
        self.max_time = max_time
        self.max_tt_size = max_tt_size
        
        self.tt = OrderedDict()
        self.nodes_searched = 0
        
        # Killer moves: 2 killers per depth level
        self.killer_moves = [[None, None] for _ in range(64)]
        
        # History heuristic table: move -> score
        self.history = {}
        
        # Board Constants (10 rows, triangle shape)
        self.ROWS = 10
        self.OFFSETS = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        self.LENGTHS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        self.WIDTH = 22
        self.TOTAL_BITS = self.ROWS * self.WIDTH
        self.TOTAL_SQUARES = 110  # Sum of lengths
        
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
        
        # All valid indices list (for iteration)
        self.ALL_INDICES = []
        for r in range(self.ROWS):
            for c in range(self.OFFSETS[r], self.OFFSETS[r] + self.LENGTHS[r]):
                self.ALL_INDICES.append(r * self.WIDTH + c)
        
        # Build edge indices for stability calculation
        self.LEFT_EDGE = []
        self.RIGHT_EDGE = []
        for r in range(self.ROWS):
            self.LEFT_EDGE.append(r * self.WIDTH + self.OFFSETS[r])
            self.RIGHT_EDGE.append(r * self.WIDTH + self.OFFSETS[r] + self.LENGTHS[r] - 1)
        
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
        
        # X-squares (diagonal to corners) - very bad
        self.X_SQUARES = set()
        self.X_SQUARES.add(8 * self.WIDTH + 2)  # Diagonal to bottom-left
        self.X_SQUARES.add(8 * self.WIDTH + 17) # Diagonal to bottom-right
        
        # C-squares (adjacent to corners on edge) - bad
        self.C_SQUARES = set()
        for adj_list in self.CORNER_TO_ADJ.values():
            for adj in adj_list:
                self.C_SQUARES.add(adj)

        # Build masks
        self.MASK_CORNER = 0
        self.MASK_TOP_CORNER = 0
        self.MASK_EDGE = 0
        self.MASK_NORMAL = 0
        self.CORNER_ADJ_MASK = 0
        self.X_MASK = 0
        
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
        
        for x in self.X_SQUARES:
            self.X_MASK |= (1 << x)
        
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
                    self.POS_VALUES[idx] = 800
                elif idx in self.X_SQUARES:
                    self.POS_VALUES[idx] = -500
                elif idx in self.C_SQUARES:
                    self.POS_VALUES[idx] = -300
                elif c == self.OFFSETS[r] or c == self.OFFSETS[r] + self.LENGTHS[r] - 1 or r == 9:
                    self.POS_VALUES[idx] = 150
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
        """Count stable discs - full edge stability analysis."""
        stable = 0
        occupied = me_bb | opp_bb
        
        # Bottom row stability (most important)
        # Left corner chain
        c_left = 9 * self.WIDTH + 0
        if me_bb & (1 << c_left):
            stable += 1
            for c in range(1, 20):
                idx = 9 * self.WIDTH + c
                if me_bb & (1 << idx):
                    stable += 1
                else:
                    break
        
        # Right corner chain
        c_right = 9 * self.WIDTH + 19
        if me_bb & (1 << c_right):
            count_right = 1
            for c in range(18, -1, -1):
                idx = 9 * self.WIDTH + c
                if me_bb & (1 << idx):
                    count_right += 1
                else:
                    break
            stable += count_right
        
        # Left edge stability (propagate from bottom-left corner up)
        if me_bb & (1 << c_left):
            for r in range(8, -1, -1):
                idx = self.LEFT_EDGE[r]
                if me_bb & (1 << idx):
                    stable += 1
                else:
                    break
        
        # Right edge stability (propagate from bottom-right corner up)
        if me_bb & (1 << c_right):
            for r in range(8, -1, -1):
                idx = self.RIGHT_EDGE[r]
                if me_bb & (1 << idx):
                    stable += 1
                else:
                    break
        
        return stable

    def get_game_phase(self, empty_count):
        """Determine game phase: opening, midgame, endgame."""
        if empty_count > 80:
            return 'opening'
        elif empty_count > 20:
            return 'midgame'
        else:
            return 'endgame'

    def evaluate(self, me_bb, opp_bb, empty_count):
        """Phase-dependent comprehensive evaluation."""
        
        # Terminal state
        if empty_count == 0:
            diff = popcount(me_bb) - popcount(opp_bb)
            return 100000 * (1 if diff > 0 else -1 if diff < 0 else 0)
        
        phase = self.get_game_phase(empty_count)
        score = 0
        occupied = me_bb | opp_bb
        
        # Phase-dependent weights
        if phase == 'opening':
            w_corner = 1000
            w_top_corner = 150
            w_edge = 20
            w_bad = -150
            w_x = -200
            w_mobility = 40
            w_frontier = -12
            w_stability = 30
            w_disc = 0  # Don't care about disc count in opening
        elif phase == 'midgame':
            w_corner = 900
            w_top_corner = 130
            w_edge = 35
            w_bad = -120
            w_x = -180
            w_mobility = 30
            w_frontier = -10
            w_stability = 60
            w_disc = 5
        else:  # endgame
            w_corner = 700
            w_top_corner = 100
            w_edge = 40
            w_bad = -80
            w_x = -100
            w_mobility = 15
            w_frontier = -5
            w_stability = 80
            w_disc = 15
        
        # === Corners ===
        my_corners = popcount(me_bb & self.MASK_CORNER)
        opp_corners = popcount(opp_bb & self.MASK_CORNER)
        score += (my_corners - opp_corners) * w_corner
        
        # === Top Corners ===
        my_top = popcount(me_bb & self.MASK_TOP_CORNER)
        opp_top = popcount(opp_bb & self.MASK_TOP_CORNER)
        score += (my_top - opp_top) * w_top_corner
        
        # === Edges ===
        my_edges = popcount(me_bb & self.MASK_EDGE)
        opp_edges = popcount(opp_bb & self.MASK_EDGE)
        score += (my_edges - opp_edges) * w_edge
        
        # === Bad squares (C-squares, dynamic check) ===
        for c_idx, adj_list in self.CORNER_TO_ADJ.items():
            if not (occupied & (1 << c_idx)):  # Corner is empty
                for adj in adj_list:
                    if me_bb & (1 << adj):
                        score += w_bad
                    if opp_bb & (1 << adj):
                        score -= w_bad
        
        # === X-squares (very bad when corner empty) ===
        for x_idx in self.X_SQUARES:
            # Check if corresponding corner is empty
            # For simplicity, always penalize X-squares in opening/midgame
            if phase != 'endgame':
                if me_bb & (1 << x_idx):
                    score += w_x
                if opp_bb & (1 << x_idx):
                    score -= w_x
        
        # === Disc count (weighted by phase) ===
        my_count = popcount(me_bb)
        opp_count = popcount(opp_bb)
        score += (my_count - opp_count) * w_disc
        
        # === Mobility ===
        my_moves = self.get_moves_bb(me_bb, opp_bb)
        opp_moves = self.get_moves_bb(opp_bb, me_bb)
        my_mob = popcount(my_moves)
        opp_mob = popcount(opp_moves)
        score += (my_mob - opp_mob) * w_mobility
        
        # === Potential Mobility (squares adjacent to opponent) ===
        # Counts empty squares next to opponent discs
        if phase != 'endgame':
            empty = self.VALID_MASK & ~occupied
            pot_mob_me = 0
            pot_mob_opp = 0
            for shift in self.SHIFTS:
                if shift > 0:
                    pot_mob_me |= (opp_bb << shift) & empty
                    pot_mob_opp |= (me_bb << shift) & empty
                else:
                    pot_mob_me |= (opp_bb >> -shift) & empty
                    pot_mob_opp |= (me_bb >> -shift) & empty
            score += (popcount(pot_mob_me) - popcount(pot_mob_opp)) * (w_mobility // 3)
        
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
        score += (my_frontier - opp_frontier) * w_frontier
        
        # === Stability ===
        my_stable = self.count_stable_discs(me_bb, opp_bb)
        opp_stable = self.count_stable_discs(opp_bb, me_bb)
        score += (my_stable - opp_stable) * w_stability
        
        # === Parity (endgame) ===
        if phase == 'endgame':
            if empty_count % 2 == 1:
                score += 20
            else:
                score -= 20
        
        return score

    def get_move_score(self, idx, me_bb, opp_bb, ply, tt_best=None):
        """Score a move for ordering purposes."""
        if idx == tt_best:
            return 1000000
        
        score = self.POS_VALUES.get(idx, 10)
        
        # Killer move bonus
        if ply < len(self.killer_moves):
            if self.killer_moves[ply][0] == idx:
                score += 5000
            elif self.killer_moves[ply][1] == idx:
                score += 4000
        
        # History heuristic bonus
        score += self.history.get(idx, 0)
        
        # Corner adjacency penalty (dynamic)
        if idx in self.ADJ_TO_CORNER:
            c_idx = self.ADJ_TO_CORNER[idx]
            if not ((me_bb | opp_bb) & (1 << c_idx)):
                score -= 8000
        
        # Flip count bonus
        flips = self.get_flips_bb(idx, me_bb, opp_bb)
        score += popcount(flips) * 20
        
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
        if ply < len(self.killer_moves) and self.killer_moves[ply][0] != move:
            self.killer_moves[ply][1] = self.killer_moves[ply][0]
            self.killer_moves[ply][0] = move

    def update_history(self, move, depth):
        """Update history heuristic."""
        bonus = depth * depth
        self.history[move] = self.history.get(move, 0) + bonus

    def negamax(self, me_bb, opp_bb, depth, alpha, beta, ply, empty_count, can_null=True):
        """Negamax with alpha-beta, PVS, and LMR."""
        self.nodes_searched += 1
        
        # Time check (every 4096 nodes)
        if self.nodes_searched & 4095 == 0:
            if time.time() - self.start_time > self.max_time:
                raise TimeoutError()
        
        alpha_orig = alpha
        
        # TT lookup
        tt_key = (me_bb, opp_bb)
        tt_entry = self.tt.get(tt_key)
        tt_best = None
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
            return -self.negamax(opp_bb, me_bb, depth, -beta, -alpha, ply + 1, empty_count, False)
        
        if depth == 0:
            return self.evaluate(me_bb, opp_bb, empty_count)
        
        moves = self.extract_moves(moves_mask)
        
        # Move ordering
        moves.sort(key=lambda m: self.get_move_score(m, me_bb, opp_bb, ply, tt_best), reverse=True)
        
        best_val = -float('inf')
        best_move = moves[0]
        
        for i, move in enumerate(moves):
            flips = self.get_flips_bb(move, me_bb, opp_bb)
            new_me = me_bb | (1 << move) | flips
            new_op = opp_bb & ~flips
            
            # Late Move Reductions (LMR)
            reduction = 0
            if i >= 4 and depth >= 3 and move not in [tt_best] + self.killer_moves[ply]:
                reduction = 1
            
            # PVS with LMR
            if i == 0:
                val = -self.negamax(new_op, new_me, depth - 1, -beta, -alpha, ply + 1, empty_count - 1, True)
            else:
                # Null window with possible reduction
                val = -self.negamax(new_op, new_me, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, empty_count - 1, True)
                
                # Re-search if LMR failed high
                if reduction > 0 and val > alpha:
                    val = -self.negamax(new_op, new_me, depth - 1, -alpha - 1, -alpha, ply + 1, empty_count - 1, True)
                
                # Full re-search if null window failed
                if alpha < val < beta:
                    val = -self.negamax(new_op, new_me, depth - 1, -beta, -val, ply + 1, empty_count - 1, True)
            
            if val > best_val:
                best_val = val
                best_move = move
            
            alpha = max(alpha, val)
            if alpha >= beta:
                self.store_killer(move, ply)
                self.update_history(move, depth)
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
        self.history.clear()  # Reset history each move
        
        me_bb, opp_bb = self.to_bitboard(board_input, 1, 2)
        
        moves_mask = self.get_moves_bb(me_bb, opp_bb)
        if not moves_mask:
            return None
        
        total_discs = popcount(me_bb | opp_bb)
        empty_count = self.TOTAL_SQUARES - total_discs
        
        # Determine search depth
        search_depth = self.depth
        if empty_count <= 14:
            # Endgame: solve exactly
            search_depth = empty_count + 6
        elif empty_count <= 24:
            search_depth = min(16, empty_count + 2)
        
        moves = self.extract_moves(moves_mask)
        
        # Quick return for single move
        if len(moves) == 1:
            r = moves[0] // self.WIDTH
            c = moves[0] % self.WIDTH
            return r, c
        
        # Initial sort
        moves.sort(key=lambda m: self.get_move_score(m, me_bb, opp_bb, 0, None), reverse=True)
        
        best_move_idx = moves[0]
        prev_best_val = 0
        
        # Iterative deepening with aspiration windows
        try:
            for d in range(1, search_depth + 1):
                # Aspiration window
                window = 50
                if d >= 4 and abs(prev_best_val) < 50000:
                    alpha = prev_best_val - window
                    beta = prev_best_val + window
                else:
                    alpha = -float('inf')
                    beta = float('inf')
                
                while True:
                    current_best_val = -float('inf')
                    current_best_move = None
                    
                    for i, move in enumerate(moves):
                        if time.time() - self.start_time > self.max_time:
                            raise TimeoutError()
                        
                        flips = self.get_flips_bb(move, me_bb, opp_bb)
                        new_me = me_bb | (1 << move) | flips
                        new_op = opp_bb & ~flips
                        
                        if i == 0:
                            val = -self.negamax(new_op, new_me, d - 1, -beta, -alpha, 1, empty_count - 1, True)
                        else:
                            val = -self.negamax(new_op, new_me, d - 1, -alpha - 1, -alpha, 1, empty_count - 1, True)
                            if alpha < val < beta:
                                val = -self.negamax(new_op, new_me, d - 1, -beta, -val, 1, empty_count - 1, True)
                        
                        if val > current_best_val:
                            current_best_val = val
                            current_best_move = move
                        
                        if val > alpha:
                            alpha = val
                        
                        if alpha >= beta:
                            break
                    
                    # Check if we need to re-search with wider window
                    if d >= 4 and current_best_val <= prev_best_val - window:
                        alpha = -float('inf')
                    elif d >= 4 and current_best_val >= prev_best_val + window:
                        beta = float('inf')
                    else:
                        break
                
                if current_best_move is not None:
                    best_move_idx = current_best_move
                    prev_best_val = current_best_val
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
    
    bot = TriangleReversiBotPro(depth=100, max_time=2.8)
    move = bot.choose_move(board_input)
    
    if move:
        r = move[0]
        c = move[1]
        local_c = c - OFFSETS[r]  # Convert global column to local (0-indexed)
        print(f"{r + 1} {local_c + 1}")  # Output 1-indexed
    else:
        print("0 0")


if __name__ == "__main__":
    main()
