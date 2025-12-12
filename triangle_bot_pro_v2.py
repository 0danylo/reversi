"""
Triangle Reversi Pro Bot v2
Improvements over Ultimate with better performance:
- History Heuristic for move ordering
- Better game phase evaluation
- X-square and C-square penalties
- Optimized for speed
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

class TriangleReversiBotPro:
    EXACT = 0
    LOWERBOUND = 1
    UPPERBOUND = 2
    
    def __init__(self, depth=100, max_time=2.8, max_tt_size=250000):
        self.start_time = time.time()
        self.depth = depth
        self.max_time = max_time
        self.max_tt_size = max_tt_size
        
        self.tt = OrderedDict()
        self.nodes_searched = 0
        
        self.killer_moves = [[None, None] for _ in range(64)]
        self.history = {}
        
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
        
        # X-squares (diagonal to corners)
        self.X_SQUARES = {8 * self.WIDTH + 2, 8 * self.WIDTH + 17}
        
        # Build masks
        self.MASK_CORNER = 0
        self.MASK_TOP_CORNER = 0
        self.MASK_EDGE = 0
        self.MASK_X = 0
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
        
        # Position values for ordering
        self.POS_VALUES = {}
        for r in range(self.ROWS):
            for c in range(self.OFFSETS[r], self.OFFSETS[r] + self.LENGTHS[r]):
                idx = r * self.WIDTH + c
                if (r, c) in self.CORNERS:
                    self.POS_VALUES[idx] = 10000
                elif (r, c) in self.TOP_CORNERS:
                    self.POS_VALUES[idx] = 800
                elif idx in self.X_SQUARES:
                    self.POS_VALUES[idx] = -400
                elif idx in self.ADJ_TO_CORNER:
                    self.POS_VALUES[idx] = -200
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

    def count_stable(self, me_bb):
        """Fast approximate stability count."""
        stable = 0
        
        # Bottom-left corner chain
        c_left = 9 * self.WIDTH + 0
        if me_bb & (1 << c_left):
            stable += 1
            for c in range(1, 20):
                if me_bb & (1 << (9 * self.WIDTH + c)):
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
            for c in range(18, -1, -1):
                if me_bb & (1 << (9 * self.WIDTH + c)):
                    stable += 1
                else:
                    break
            # Up right edge
            for r in range(8, -1, -1):
                if me_bb & (1 << self.RIGHT_EDGE[r]):
                    stable += 1
                else:
                    break
        
        return stable

    def evaluate(self, me_bb, opp_bb, empty_count):
        if empty_count == 0:
            diff = popcount(me_bb) - popcount(opp_bb)
            return 100000 * (1 if diff > 0 else -1 if diff < 0 else 0)
        
        score = 0
        occupied = me_bb | opp_bb
        
        # Phase-dependent weights
        if empty_count > 80:
            w_corner, w_top, w_edge, w_bad, w_x, w_mob, w_front, w_stab, w_disc = 1000, 150, 20, -150, -200, 40, -12, 30, 0
        elif empty_count > 20:
            w_corner, w_top, w_edge, w_bad, w_x, w_mob, w_front, w_stab, w_disc = 900, 130, 35, -120, -180, 30, -10, 60, 5
        else:
            w_corner, w_top, w_edge, w_bad, w_x, w_mob, w_front, w_stab, w_disc = 700, 100, 40, -80, -100, 15, -5, 80, 15
        
        # Corners
        score += (popcount(me_bb & self.MASK_CORNER) - popcount(opp_bb & self.MASK_CORNER)) * w_corner
        
        # Top corners
        score += (popcount(me_bb & self.MASK_TOP_CORNER) - popcount(opp_bb & self.MASK_TOP_CORNER)) * w_top
        
        # Edges
        score += (popcount(me_bb & self.MASK_EDGE) - popcount(opp_bb & self.MASK_EDGE)) * w_edge
        
        # C-squares (dynamic)
        for c_idx, adj_list in self.CORNER_TO_ADJ.items():
            if not (occupied & (1 << c_idx)):
                for adj in adj_list:
                    if me_bb & (1 << adj):
                        score += w_bad
                    if opp_bb & (1 << adj):
                        score -= w_bad
        
        # X-squares
        if empty_count > 20:
            score += (popcount(me_bb & self.MASK_X) - popcount(opp_bb & self.MASK_X)) * w_x
        
        # Disc count
        score += (popcount(me_bb) - popcount(opp_bb)) * w_disc
        
        # Mobility
        my_mob = popcount(self.get_moves_bb(me_bb, opp_bb))
        opp_mob = popcount(self.get_moves_bb(opp_bb, me_bb))
        score += (my_mob - opp_mob) * w_mob
        
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
        score += (self.count_stable(me_bb) - self.count_stable(opp_bb)) * w_stab
        
        # Parity
        if empty_count <= 10:
            score += 20 if empty_count % 2 == 1 else -20
        
        return score

    def get_move_score(self, idx, me_bb, opp_bb, ply, tt_best):
        if idx == tt_best:
            return 1000000
        
        score = self.POS_VALUES.get(idx, 10)
        
        if ply < len(self.killer_moves):
            if self.killer_moves[ply][0] == idx:
                score += 5000
            elif self.killer_moves[ply][1] == idx:
                score += 4000
        
        score += self.history.get(idx, 0)
        
        # Dynamic C-square penalty
        if idx in self.ADJ_TO_CORNER:
            c_idx = self.ADJ_TO_CORNER[idx]
            if not ((me_bb | opp_bb) & (1 << c_idx)):
                score -= 8000
        
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

    def negamax(self, me_bb, opp_bb, depth, alpha, beta, ply, empty_count):
        self.nodes_searched += 1
        
        if self.nodes_searched & 2047 == 0:
            if time.time() - self.start_time > self.max_time:
                raise TimeoutError()
        
        alpha_orig = alpha
        
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
            return -self.negamax(opp_bb, me_bb, depth, -beta, -alpha, ply + 1, empty_count)
        
        if depth == 0:
            return self.evaluate(me_bb, opp_bb, empty_count)
        
        moves = self.extract_moves(moves_mask)
        moves.sort(key=lambda m: self.get_move_score(m, me_bb, opp_bb, ply, tt_best), reverse=True)
        
        best_val = -float('inf')
        best_move = moves[0]
        
        for i, move in enumerate(moves):
            flips = self.get_flips_bb(move, me_bb, opp_bb)
            new_me = me_bb | (1 << move) | flips
            new_op = opp_bb & ~flips
            
            if i == 0:
                val = -self.negamax(new_op, new_me, depth - 1, -beta, -alpha, ply + 1, empty_count - 1)
            else:
                val = -self.negamax(new_op, new_me, depth - 1, -alpha - 1, -alpha, ply + 1, empty_count - 1)
                if alpha < val < beta:
                    val = -self.negamax(new_op, new_me, depth - 1, -beta, -val, ply + 1, empty_count - 1)
            
            if val > best_val:
                best_val = val
                best_move = move
            
            alpha = max(alpha, val)
            if alpha >= beta:
                self.store_killer(move, ply)
                self.history[move] = self.history.get(move, 0) + depth * depth
                break
        
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
        
        me_bb, opp_bb = self.to_bitboard(board_input, 1, 2)
        
        moves_mask = self.get_moves_bb(me_bb, opp_bb)
        if not moves_mask:
            return None
        
        total_discs = popcount(me_bb | opp_bb)
        empty_count = self.TOTAL_SQUARES - total_discs
        
        search_depth = self.depth
        if empty_count <= 12:
            search_depth = empty_count + 4
        elif empty_count <= 20:
            search_depth = min(14, empty_count + 2)
        
        moves = self.extract_moves(moves_mask)
        
        if len(moves) == 1:
            r = moves[0] // self.WIDTH
            c = moves[0] % self.WIDTH
            return r, c
        
        moves.sort(key=lambda m: self.get_move_score(m, me_bb, opp_bb, 0, None), reverse=True)
        
        best_move_idx = moves[0]
        
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
        OFFSETS = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        local_c = c - OFFSETS[r]  # Convert global column to local (0-indexed)
        print(f"{r + 1} {local_c + 1}")  # Output 1-indexed
    else:
        print("0 0")


if __name__ == "__main__":
    main()
