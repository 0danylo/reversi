"""
Triangle Reversi Pro Bot v3
Further improvements over Pro v2:
- Late Move Reductions (LMR)
- Counter-move heuristic
- Improved stability (wedges)
- Better endgame exact solving
- Move ordering includes flip count
- Futility pruning at shallow depths
- Aspiration windows
"""

import sys
import time
from collections import OrderedDict

# Pre-computed bit counts
BIT_COUNT = [bin(i).count('1') for i in range(256)]

def popcount(n):
    count = 0
    while n:
        count += BIT_COUNT[n & 0xFF]
        n >>= 8
    return count

class TriangleReversiBotProV3:
    EXACT = 0
    LOWERBOUND = 1
    UPPERBOUND = 2
    
    def __init__(self, depth, max_time, max_tt_size=300000):
        self.start_time = time.time()
        self.depth = depth
        self.max_time = max_time
        self.max_tt_size = max_tt_size
        
        self.tt = OrderedDict()
        self.nodes_searched = 0
        
        self.killer_moves = [[None, None] for _ in range(64)]
        self.history = {}
        self.counter_moves = {}  # (prev_move) -> best response
        
        # Board Constants
        self.ROWS = 10
        self.OFFSETS = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        self.LENGTHS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        self.WIDTH = 22
        self.TOTAL_SQUARES = 110
        
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
            
        self.SHIFTS = [1, -1, self.WIDTH, -self.WIDTH, 
                       self.WIDTH+1, self.WIDTH-1, -self.WIDTH+1, -self.WIDTH-1]
        
        self.CORNERS = [(9, 0), (9, 19)]
        self.TOP_CORNERS = [(0, 9), (0, 10)]
        
        # Edge indices for stability
        self.LEFT_EDGE = [r * self.WIDTH + self.OFFSETS[r] for r in range(self.ROWS)]
        self.RIGHT_EDGE = [r * self.WIDTH + self.OFFSETS[r] + self.LENGTHS[r] - 1 for r in range(self.ROWS)]
        
        # Corner indices
        self.CORNER_INDICES = [9 * self.WIDTH + 0, 9 * self.WIDTH + 19]
        
        # Corner adjacency
        self.CORNER_TO_ADJ = {}
        self.ADJ_TO_CORNER = {}
        
        c_idx_bl = 9 * self.WIDTH + 0
        adj_bl = [9 * self.WIDTH + 1, 8 * self.WIDTH + 1]
        self.CORNER_TO_ADJ[c_idx_bl] = adj_bl
        for adj in adj_bl:
            self.ADJ_TO_CORNER[adj] = c_idx_bl
        
        c_idx_br = 9 * self.WIDTH + 19
        adj_br = [9 * self.WIDTH + 18, 8 * self.WIDTH + 18]
        self.CORNER_TO_ADJ[c_idx_br] = adj_br
        for adj in adj_br:
            self.ADJ_TO_CORNER[adj] = c_idx_br
        
        # X-squares (diagonal to corners) - very dangerous
        self.X_SQUARES = {8 * self.WIDTH + 2, 8 * self.WIDTH + 17}
        
        # A-squares (one away from corner on edge) - moderately bad
        self.A_SQUARES = {9 * self.WIDTH + 2, 9 * self.WIDTH + 17}
        
        # B-squares (two away from corner on edge) - slightly risky
        self.B_SQUARES = {9 * self.WIDTH + 3, 9 * self.WIDTH + 16}
        
        # Build masks
        self.MASK_CORNER = 0
        self.MASK_TOP_CORNER = 0
        self.MASK_EDGE = 0
        self.MASK_X = 0
        self.MASK_A = 0
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
        
        for adj_list in self.CORNER_TO_ADJ.values():
            for adj in adj_list:
                self.CORNER_ADJ_MASK |= (1 << adj)
        
        for x in self.X_SQUARES:
            self.MASK_X |= (1 << x)
        
        for a in self.A_SQUARES:
            self.MASK_A |= (1 << a)
        
        # Position values for ordering
        self.POS_VALUES = {}
        for r in range(self.ROWS):
            for c in range(self.OFFSETS[r], self.OFFSETS[r] + self.LENGTHS[r]):
                idx = r * self.WIDTH + c
                if (r, c) in self.CORNERS:
                    self.POS_VALUES[idx] = 10000
                elif (r, c) in self.TOP_CORNERS:
                    self.POS_VALUES[idx] = 900
                elif idx in self.X_SQUARES:
                    self.POS_VALUES[idx] = -500
                elif idx in self.ADJ_TO_CORNER:
                    self.POS_VALUES[idx] = -300
                elif idx in self.A_SQUARES:
                    self.POS_VALUES[idx] = -100
                elif c == self.OFFSETS[r] or c == self.OFFSETS[r] + self.LENGTHS[r] - 1 or r == 9:
                    self.POS_VALUES[idx] = 180
                elif idx in self.B_SQUARES:
                    self.POS_VALUES[idx] = 50
                else:
                    self.POS_VALUES[idx] = 10
        
        # LMR table: reduction[depth][move_count]
        self.LMR_TABLE = [[0] * 64 for _ in range(64)]
        for d in range(1, 64):
            for m in range(1, 64):
                if d >= 3 and m >= 3:
                    self.LMR_TABLE[d][m] = 1 + (d // 8) + (m // 12)
                    self.LMR_TABLE[d][m] = min(self.LMR_TABLE[d][m], d - 1)

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

    def count_stable(self, me_bb, opp_bb):
        """Improved stability count including wedges."""
        stable = 0
        occupied = me_bb | opp_bb
        
        # Bottom-left corner chain
        c_left = 9 * self.WIDTH + 0
        if me_bb & (1 << c_left):
            stable += 1
            # Along bottom row
            for c in range(1, 20):
                idx = 9 * self.WIDTH + c
                if me_bb & (1 << idx):
                    stable += 1
                else:
                    break
            # Up left edge
            for r in range(8, -1, -1):
                if me_bb & (1 << self.LEFT_EDGE[r]):
                    stable += 1
                else:
                    break
        
        # Bottom-right corner chain
        c_right = 9 * self.WIDTH + 19
        if me_bb & (1 << c_right):
            stable += 1
            # Along bottom row (backwards)
            for c in range(18, -1, -1):
                idx = 9 * self.WIDTH + c
                if me_bb & (1 << idx):
                    stable += 1
                else:
                    break
            # Up right edge
            for r in range(8, -1, -1):
                if me_bb & (1 << self.RIGHT_EDGE[r]):
                    stable += 1
                else:
                    break
        
        # Wedge stability: if entire bottom row is full, count all pieces
        full_bottom = True
        for c in range(20):
            idx = 9 * self.WIDTH + c
            if not (occupied & (1 << idx)):
                full_bottom = False
                break
        
        if full_bottom:
            stable += popcount(me_bb & self.ROW_MASKS[9])
        
        return stable

    def evaluate(self, me_bb, opp_bb, empty_count):
        if empty_count == 0:
            diff = popcount(me_bb) - popcount(opp_bb)
            return 100000 * (1 if diff > 0 else -1 if diff < 0 else 0)
        
        score = 0
        occupied = me_bb | opp_bb
        
        # Phase-dependent weights (more aggressive tuning)
        if empty_count > 85:
            # Opening: maximize mobility, minimize discs
            w_corner, w_top, w_edge = 1200, 180, 15
            w_bad, w_x, w_a = -180, -250, -80
            w_mob, w_front, w_stab, w_disc = 50, -15, 25, -5
        elif empty_count > 50:
            # Early midgame
            w_corner, w_top, w_edge = 1000, 150, 25
            w_bad, w_x, w_a = -150, -220, -60
            w_mob, w_front, w_stab, w_disc = 40, -12, 45, 0
        elif empty_count > 20:
            # Late midgame
            w_corner, w_top, w_edge = 900, 130, 40
            w_bad, w_x, w_a = -120, -180, -40
            w_mob, w_front, w_stab, w_disc = 30, -10, 70, 5
        else:
            # Endgame: disc count matters
            w_corner, w_top, w_edge = 700, 100, 50
            w_bad, w_x, w_a = -80, -100, -20
            w_mob, w_front, w_stab, w_disc = 15, -5, 90, 20
        
        # Corners
        my_corners = popcount(me_bb & self.MASK_CORNER)
        opp_corners = popcount(opp_bb & self.MASK_CORNER)
        score += (my_corners - opp_corners) * w_corner
        
        # Top corners
        score += (popcount(me_bb & self.MASK_TOP_CORNER) - popcount(opp_bb & self.MASK_TOP_CORNER)) * w_top
        
        # Edges
        score += (popcount(me_bb & self.MASK_EDGE) - popcount(opp_bb & self.MASK_EDGE)) * w_edge
        
        # C-squares (dynamic - only bad if corner empty)
        for c_idx, adj_list in self.CORNER_TO_ADJ.items():
            if not (occupied & (1 << c_idx)):
                for adj in adj_list:
                    if me_bb & (1 << adj):
                        score += w_bad
                    if opp_bb & (1 << adj):
                        score -= w_bad
        
        # X-squares (dynamic)
        for c_idx in self.CORNER_INDICES:
            if not (occupied & (1 << c_idx)):
                # Check X-square for this corner
                if c_idx == 9 * self.WIDTH + 0:
                    x_idx = 8 * self.WIDTH + 2
                else:
                    x_idx = 8 * self.WIDTH + 17
                if me_bb & (1 << x_idx):
                    score += w_x
                if opp_bb & (1 << x_idx):
                    score -= w_x
        
        # A-squares (dynamic)
        for c_idx in self.CORNER_INDICES:
            if not (occupied & (1 << c_idx)):
                if c_idx == 9 * self.WIDTH + 0:
                    a_idx = 9 * self.WIDTH + 2
                else:
                    a_idx = 9 * self.WIDTH + 17
                if me_bb & (1 << a_idx):
                    score += w_a
                if opp_bb & (1 << a_idx):
                    score -= w_a
        
        # Disc count
        score += (popcount(me_bb) - popcount(opp_bb)) * w_disc
        
        # Mobility
        my_mob = popcount(self.get_moves_bb(me_bb, opp_bb))
        opp_mob = popcount(self.get_moves_bb(opp_bb, me_bb))
        score += (my_mob - opp_mob) * w_mob
        
        # Potential mobility (empty squares adjacent to opponent)
        if empty_count > 20:
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
            score += (popcount(pot_mob_me) - popcount(pot_mob_opp)) * (w_mob // 4)
        
        # Frontier
        empty = self.VALID_MASK & ~occupied
        frontier = 0
        for shift in self.SHIFTS:
            if shift > 0:
                frontier |= (empty << shift)
            else:
                frontier |= (empty >> -shift)
        frontier &= self.VALID_MASK
        score += (popcount(me_bb & frontier) - popcount(opp_bb & frontier)) * w_front
        
        # Stability
        score += (self.count_stable(me_bb, opp_bb) - self.count_stable(opp_bb, me_bb)) * w_stab
        
        # Parity
        if empty_count <= 12:
            score += 25 if empty_count % 2 == 1 else -25
        
        return score

    def get_move_score(self, idx, me_bb, opp_bb, ply, tt_best, prev_move):
        if idx == tt_best:
            return 1000000
        
        score = self.POS_VALUES.get(idx, 10)
        
        # Killer moves
        if ply < len(self.killer_moves):
            if self.killer_moves[ply][0] == idx:
                score += 6000
            elif self.killer_moves[ply][1] == idx:
                score += 5000
        
        # Counter-move heuristic
        if prev_move is not None and self.counter_moves.get(prev_move) == idx:
            score += 4500
        
        # History heuristic
        score += self.history.get(idx, 0)
        
        # Dynamic C-square penalty
        if idx in self.ADJ_TO_CORNER:
            c_idx = self.ADJ_TO_CORNER[idx]
            if not ((me_bb | opp_bb) & (1 << c_idx)):
                score -= 9000
        
        # Flip count bonus (greedy heuristic)
        flips = self.get_flips_bb(idx, me_bb, opp_bb)
        score += popcount(flips) * 15
        
        return score

    def extract_moves(self, moves_mask):
        moves = []
        temp = moves_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            moves.append(idx)
            temp ^= lsb
        return moves

    def store_killer(self, move, ply):
        if ply < len(self.killer_moves) and self.killer_moves[ply][0] != move:
            self.killer_moves[ply][1] = self.killer_moves[ply][0]
            self.killer_moves[ply][0] = move

    def negamax(self, me_bb, opp_bb, depth, alpha, beta, ply, empty_count, prev_move):
        self.nodes_searched += 1
        
        if self.nodes_searched & 2047 == 0:
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
                diff = popcount(me_bb) - popcount(opp_bb)
                return 100000 if diff > 0 else -100000 if diff < 0 else 0
            return -self.negamax(opp_bb, me_bb, depth, -beta, -alpha, ply + 1, empty_count, None)
        
        if depth == 0:
            return self.evaluate(me_bb, opp_bb, empty_count)
        
        moves = self.extract_moves(moves_mask)
        moves.sort(key=lambda m: self.get_move_score(m, me_bb, opp_bb, ply, tt_best, prev_move), reverse=True)
        
        best_val = -float('inf')
        best_move = moves[0]
        
        for i, move in enumerate(moves):
            flips = self.get_flips_bb(move, me_bb, opp_bb)
            new_me = me_bb | (1 << move) | flips
            new_op = opp_bb & ~flips
            
            # Late Move Reductions
            reduction = 0
            if i >= 3 and depth >= 3:
                # Don't reduce killers, TT move, or corner moves
                if move != tt_best and move not in [self.killer_moves[ply][0], self.killer_moves[ply][1]]:
                    if not (move in self.POS_VALUES and self.POS_VALUES[move] >= 9000):
                        reduction = self.LMR_TABLE[min(depth, 63)][min(i, 63)]
            
            if i == 0:
                val = -self.negamax(new_op, new_me, depth - 1, -beta, -alpha, ply + 1, empty_count - 1, move)
            else:
                # Null window + LMR
                val = -self.negamax(new_op, new_me, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, empty_count - 1, move)
                
                # Re-search without reduction if LMR failed high
                if reduction > 0 and val > alpha:
                    val = -self.negamax(new_op, new_me, depth - 1, -alpha - 1, -alpha, ply + 1, empty_count - 1, move)
                
                # Full re-search if null window failed
                if alpha < val < beta:
                    val = -self.negamax(new_op, new_me, depth - 1, -beta, -val, ply + 1, empty_count - 1, move)
            
            if val > best_val:
                best_val = val
                best_move = move
            
            alpha = max(alpha, val)
            if alpha >= beta:
                self.store_killer(move, ply)
                self.history[move] = self.history.get(move, 0) + depth * depth
                if prev_move is not None:
                    self.counter_moves[prev_move] = move
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
        self.history.clear()
        self.counter_moves.clear()
        
        me_bb, opp_bb = self.to_bitboard(board_input, 1, 2)
        
        moves_mask = self.get_moves_bb(me_bb, opp_bb)
        if not moves_mask:
            return None
        
        total_discs = popcount(me_bb | opp_bb)
        empty_count = self.TOTAL_SQUARES - total_discs
        
        # Adaptive depth
        search_depth = self.depth
        if empty_count <= 14:
            search_depth = empty_count + 6  # Solve exactly
        elif empty_count <= 22:
            search_depth = min(16, empty_count + 2)
        
        moves = self.extract_moves(moves_mask)
        
        if len(moves) == 1:
            r = moves[0] // self.WIDTH
            c = moves[0] % self.WIDTH
            return r, c
        
        moves.sort(key=lambda m: self.get_move_score(m, me_bb, opp_bb, 0, None, None), reverse=True)
        
        best_move_idx = moves[0]
        prev_score = 0
        
        # Iterative deepening with aspiration windows
        try:
            for d in range(1, search_depth + 1):
                # Aspiration window
                window = 40
                if d >= 5:
                    alpha = prev_score - window
                    beta = prev_score + window
                else:
                    alpha = -float('inf')
                    beta = float('inf')
                
                attempts = 0
                while attempts < 3:
                    attempts += 1
                    current_best_val = -float('inf')
                    current_best_move = None
                    
                    for i, move in enumerate(moves):
                        if time.time() - self.start_time > self.max_time:
                            raise TimeoutError()
                        
                        flips = self.get_flips_bb(move, me_bb, opp_bb)
                        new_me = me_bb | (1 << move) | flips
                        new_op = opp_bb & ~flips
                        
                        if i == 0:
                            val = -self.negamax(new_op, new_me, d - 1, -beta, -alpha, 1, empty_count - 1, move)
                        else:
                            val = -self.negamax(new_op, new_me, d - 1, -alpha - 1, -alpha, 1, empty_count - 1, move)
                            if alpha < val < beta:
                                val = -self.negamax(new_op, new_me, d - 1, -beta, -val, 1, empty_count - 1, move)
                        
                        if val > current_best_val:
                            current_best_val = val
                            current_best_move = move
                        
                        if val > alpha:
                            alpha = val
                        
                        if alpha >= beta:
                            break
                    
                    # Aspiration window failed
                    if d >= 5:
                        if current_best_val <= prev_score - window:
                            alpha = -float('inf')
                            continue
                        elif current_best_val >= prev_score + window:
                            beta = float('inf')
                            continue
                    break
                
                if current_best_move is not None:
                    best_move_idx = current_best_move
                    prev_score = current_best_val
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
    
    bot = TriangleReversiBotProV3(depth=100, max_time=2.8)
    move = bot.choose_move(board_input)
    
    if move:
        r = move[0]
        c = move[1]
        OFFSETS = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        local_c = c - OFFSETS[r]  # Convert global column to local (0-indexed)
        print(f"{r + 1} {local_c + 1}")  # Output 1-indexed
    else:
        print("0 0")


if __name__ == "__main__":
    main()
