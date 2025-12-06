import sys
import time
import pickle
import os
import random
from collections import OrderedDict

class TriangleReversiBot:
    def __init__(self, depth=6, max_time=2.75, max_tt_size=100000, tt_file="triangle_tt.pkl"):
        self.start_init_time = time.time()
        self.depth = depth
        self.max_time = max_time
        self.max_tt_size = max_tt_size
        self.tt_file = tt_file
        
        self.tt = OrderedDict()
        
        # Board Constants
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
            
        # Directions
        self.SHIFTS = [1, -1, self.WIDTH, -self.WIDTH, 
                       self.WIDTH+1, self.WIDTH-1, -self.WIDTH+1, -self.WIDTH-1]
        
        # Evaluation Weights (Fine-tuned v2)
        self.W_CORNER = 500
        self.W_TOP_CORNER = 97
        self.W_EDGE = 26
        self.W_BAD = -50
        self.W_NORMAL = 13
        self.W_MOBILITY = 25
        self.W_FRONTIER = -20
        
        self.CORNER_MASK = 0
        self.CORNERS = [(9, 0), (9, 19)]
        self.TOP_CORNERS = [(0, 9), (0, 10)]
        
        # Dynamic Corner Adjacency
        # Map Corner Index -> List of Adjacent Indices
        self.CORNER_TO_ADJ = {}
        
        # Bottom Left (9, 0) -> Neighbors (9, 1), (8, 1)
        c_idx_bl = 9 * self.WIDTH + 0
        adj_bl = [9 * self.WIDTH + 1, 8 * self.WIDTH + 1]
        self.CORNER_TO_ADJ[c_idx_bl] = adj_bl
        
        # Bottom Right (9, 19) -> Neighbors (9, 18), (8, 18)
        c_idx_br = 9 * self.WIDTH + 19
        adj_br = [9 * self.WIDTH + 18, 8 * self.WIDTH + 18]
        self.CORNER_TO_ADJ[c_idx_br] = adj_br

        # Masks for evaluation
        self.MASK_CORNER = 0
        self.MASK_TOP_CORNER = 0
        self.MASK_EDGE = 0
        self.MASK_NORMAL = 0
        
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

    def save_tt(self):
        # Simple persistence
        try:
            with open(self.tt_file, "wb") as f:
                pickle.dump(self.tt, f)
        except Exception:
            pass
            
    def load_tt(self):
        if os.path.exists(self.tt_file):
            try:
                with open(self.tt_file, "rb") as f:
                    self.tt = pickle.load(f)
            except Exception:
                pass

    def to_bitboard(self, board_input, me, opp):
        me_bb = 0
        opp_bb = 0
        for r in range(self.ROWS):
            row_vals = board_input[r]
            offset = self.OFFSETS[r]
            for i, val in enumerate(row_vals):
                if val == 0: continue
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
        for shift in self.SHIFTS:
            current_flips = 0
            mask = 1 << move_idx
            if shift > 0:
                mask = (mask << shift) & opp_bb
                while mask:
                    current_flips |= mask
                    mask = (mask << shift)
                    if mask & me_bb:
                        flips |= current_flips
                        break
                    if not (mask & opp_bb):
                        break
            else:
                s = -shift
                mask = (mask >> s) & opp_bb
                while mask:
                    current_flips |= mask
                    mask = (mask >> s)
                    if mask & me_bb:
                        flips |= current_flips
                        break
                    if not (mask & opp_bb):
                        break
        return flips

    def evaluate(self, me_bb, opp_bb):
        # Fast evaluation using bit counts
        
        # Mobility
        my_moves = self.get_moves_bb(me_bb, opp_bb)
        op_moves = self.get_moves_bb(opp_bb, me_bb)
        
        my_mob = bin(my_moves).count('1')
        op_mob = bin(op_moves).count('1')
        
        # Frontier Discs
        empty = self.VALID_MASK & ~(me_bb | opp_bb)
        frontier_mask = 0
        for shift in self.SHIFTS:
            if shift > 0:
                frontier_mask |= (empty << shift)
            else:
                frontier_mask |= (empty >> -shift)
        frontier_mask &= self.VALID_MASK
        
        my_frontier = bin(me_bb & frontier_mask).count('1')
        op_frontier = bin(opp_bb & frontier_mask).count('1')
        
        # Material weights
        my_score = 0
        op_score = 0
        
        my_score += bin(me_bb & self.MASK_CORNER).count('1') * self.W_CORNER
        op_score += bin(opp_bb & self.MASK_CORNER).count('1') * self.W_CORNER
        
        my_score += bin(me_bb & self.MASK_TOP_CORNER).count('1') * self.W_TOP_CORNER
        op_score += bin(opp_bb & self.MASK_TOP_CORNER).count('1') * self.W_TOP_CORNER
        
        my_score += bin(me_bb & self.MASK_EDGE).count('1') * self.W_EDGE
        op_score += bin(opp_bb & self.MASK_EDGE).count('1') * self.W_EDGE
        
        # Bad squares - Dynamic check
        for c_idx, adj_list in self.CORNER_TO_ADJ.items():
            # If corner is empty
            if not ((me_bb | opp_bb) & (1 << c_idx)):
                for adj in adj_list:
                    if me_bb & (1 << adj):
                        my_score += self.W_BAD
                    if opp_bb & (1 << adj):
                        op_score += self.W_BAD
        
        my_score += bin(me_bb & self.MASK_NORMAL).count('1') * self.W_NORMAL
        op_score += bin(opp_bb & self.MASK_NORMAL).count('1') * self.W_NORMAL
        
        # Frontier penalty
        my_score += my_frontier * self.W_FRONTIER
        op_score += op_frontier * self.W_FRONTIER
            
        score = (my_score - op_score) + self.W_MOBILITY * (my_mob - op_mob)
        return score

    def get_move_score(self, idx, me_bb, opp_bb):
        score = 0
        mask = 1 << idx
        
        if mask & self.MASK_CORNER: score += self.W_CORNER
        elif mask & self.MASK_TOP_CORNER: score += self.W_TOP_CORNER
        elif mask & self.MASK_EDGE: score += self.W_EDGE
        else: score += self.W_NORMAL
        
        # Dynamic Bad Square Penalty
        for c_idx, adj_list in self.CORNER_TO_ADJ.items():
            if idx in adj_list:
                # If corner is empty
                if not ((me_bb | opp_bb) & (1 << c_idx)):
                    score += self.W_BAD # Negative value
        
        # Flips (Mobility/Greedy)
        flips = self.get_flips_bb(idx, me_bb, opp_bb)
        score += bin(flips).count('1') * 10
        
        return score

    def alphabeta(self, me_bb, opp_bb, depth, alpha, beta, maximizing):
        # Check TT
        state_hash = (me_bb, opp_bb, maximizing)
        if state_hash in self.tt:
            entry = self.tt[state_hash]
            if entry['depth'] >= depth:
                if entry['flag'] == 'exact':
                    return entry['val']
                elif entry['flag'] == 'lower' and entry['val'] > alpha:
                    alpha = entry['val']
                elif entry['flag'] == 'upper' and entry['val'] < beta:
                    beta = entry['val']
                if alpha >= beta:
                    return entry['val']

        if depth == 0:
            val = self.evaluate(me_bb, opp_bb)
            return val

        moves_mask = self.get_moves_bb(me_bb, opp_bb)
        if not moves_mask:
            # Pass
            op_moves = self.get_moves_bb(opp_bb, me_bb)
            if not op_moves:
                # Game Over
                diff = bin(me_bb).count('1') - bin(opp_bb).count('1')
                return 100000 if diff > 0 else -100000 if diff < 0 else 0
            
            # Pass turn
            val = -self.alphabeta(opp_bb, me_bb, depth, -beta, -alpha, not maximizing)
            return val

        # Extract moves
        moves = []
        temp = moves_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            moves.append(idx)
            temp ^= lsb
            
        # Move ordering
        moves.sort(key=lambda m: self.get_move_score(m, me_bb, opp_bb), reverse=True)

        best_val = -float('inf')
        
        for move in moves:
            # Check time
            if time.time() - self.start_init_time > self.max_time:
                raise TimeoutError()
                
            flips = self.get_flips_bb(move, me_bb, opp_bb)
            new_me = me_bb | (1 << move) | flips
            new_op = opp_bb & ~flips
            
            val = -self.alphabeta(new_op, new_me, depth - 1, -beta, -alpha, not maximizing)
            
            if val > best_val:
                best_val = val
            
            alpha = max(alpha, val)
            if alpha >= beta:
                break
        
        # Store in TT
        flag = 'exact'
        if best_val <= alpha: flag = 'upper'
        
        self.tt[state_hash] = {'val': best_val, 'depth': depth, 'flag': flag}
        return best_val

    def choose_move(self, board_input):
        me_bb, opp_bb = self.to_bitboard(board_input, 1, 2)
        
        moves_mask = self.get_moves_bb(me_bb, opp_bb)
        if not moves_mask:
            return None
            
        # Endgame Solver Check
        total_discs = bin(me_bb | opp_bb).count('1')
        empty_count = 110 - total_discs
        
        current_depth = self.depth
        if empty_count <= 14:
            current_depth = empty_count + 2 # Ensure we search to the end
            # Increase time limit for endgame if needed, but usually it's fast
            self.max_time = 4.5 # Give more time for exact solve
            
        # Iterative Deepening
        best_move_idx = None
        
        # Extract moves for sorting
        moves = []
        temp = moves_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            moves.append(idx)
            temp ^= lsb
            
        # Initial sort
        moves.sort(key=lambda m: self.get_move_score(m, me_bb, opp_bb), reverse=True)
        
        try:
            for d in range(1, current_depth + 1):
                alpha = -float('inf')
                beta = float('inf')
                current_best_val = -float('inf')
                current_best_move = None
                
                for move in moves:
                    if time.time() - self.start_init_time > self.max_time:
                        raise TimeoutError()
                        
                    flips = self.get_flips_bb(move, me_bb, opp_bb)
                    new_me = me_bb | (1 << move) | flips
                    new_op = opp_bb & ~flips
                    
                    val = -self.alphabeta(new_op, new_me, d - 1, -beta, -alpha, False)
                    
                    if val > current_best_val:
                        current_best_val = val
                        current_best_move = move
                    
                    alpha = max(alpha, val)
                
                if current_best_move is not None:
                    best_move_idx = current_best_move
                    # Move best move to front for next iteration
                    moves.remove(best_move_idx)
                    moves.insert(0, best_move_idx)
                    
        except TimeoutError:
            pass
            
        if best_move_idx is None:
            # Fallback to first valid move
            temp = moves_mask
            lsb = temp & -temp
            best_move_idx = lsb.bit_length() - 1
            
        self.save_tt()
        
        # Convert back to (r, c)
        r = best_move_idx // self.WIDTH
        c = best_move_idx % self.WIDTH
        return r, c

def main():
    # Read input
    board_input = []
    try:
        for _ in range(10):
            line = sys.stdin.readline()
            if not line: break
            board_input.append(list(map(int, line.split())))
    except ValueError:
        pass
        
    if len(board_input) < 10:
        return

    bot = TriangleReversiBot(depth=10, max_time=2.75)
    move = bot.choose_move(board_input)
    
    if move:
        print(f"{move[0]} {move[1]}")
    else:
        print("0 0")

if __name__ == "__main__":
    main()
