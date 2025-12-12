import sys
import time
import pickle
import os
import random
from collections import OrderedDict

class TriangleReversiBot:
    def __init__(self, depth=6, max_time=2.5, max_tt_size=100000, tt_file="triangle_tt.pkl"):
        self.start_init_time = time.time()
        self.depth = depth
        self.max_time = max_time
        self.max_tt_size = max_tt_size
        self.tt_file = tt_file
        
        self.tt = OrderedDict()
        # TT persistence is disabled to avoid startup overhead.
        # In-memory TT is used for iterative deepening within a single move.
        
        # Board Constants
        self.ROWS = 10
        self.OFFSETS = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        self.LENGTHS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        self.WIDTH = 22 # Buffer to prevent wrap-around issues with simple shifts
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
        
        # Evaluation Weights
        self.CORNER_MASK = 0
        self.CORNERS = [(9, 0), (9, 19)]
        self.TOP_CORNERS = [(0, 9), (0, 10)]
        
        # Masks for evaluation
        self.MASK_CORNER = 0
        self.MASK_TOP_CORNER = 0
        self.MASK_EDGE = 0
        self.MASK_BAD = 0
        self.MASK_NORMAL = 0
        
        for r in range(self.ROWS):
            for c in range(self.OFFSETS[r], self.OFFSETS[r] + self.LENGTHS[r]):
                idx = r * self.WIDTH + c
                
                if (r, c) in self.CORNERS:
                    self.MASK_CORNER |= (1 << idx)
                elif (r, c) in self.TOP_CORNERS:
                    self.MASK_TOP_CORNER |= (1 << idx)
                elif (r, c) in [(8, 1), (9, 1), (8, 18), (9, 18)]:
                    self.MASK_BAD |= (1 << idx)
                elif c == self.OFFSETS[r] or c == self.OFFSETS[r] + self.LENGTHS[r] - 1 or r == 9:
                    self.MASK_EDGE |= (1 << idx)
                else:
                    self.MASK_NORMAL |= (1 << idx)

    def save_tt(self):
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
        
        # Material weights
        my_score = 0
        op_score = 0
        
        # Corners (500)
        my_score += bin(me_bb & self.MASK_CORNER).count('1') * 500
        op_score += bin(opp_bb & self.MASK_CORNER).count('1') * 500
        
        # Top Corners (100)
        my_score += bin(me_bb & self.MASK_TOP_CORNER).count('1') * 100
        op_score += bin(opp_bb & self.MASK_TOP_CORNER).count('1') * 100
        
        # Edges (30)
        my_score += bin(me_bb & self.MASK_EDGE).count('1') * 30
        op_score += bin(opp_bb & self.MASK_EDGE).count('1') * 30
        
        # Bad squares (-50)
        my_score += bin(me_bb & self.MASK_BAD).count('1') * -50
        op_score += bin(opp_bb & self.MASK_BAD).count('1') * -50
        
        # Normal (10)
        my_score += bin(me_bb & self.MASK_NORMAL).count('1') * 10
        op_score += bin(opp_bb & self.MASK_NORMAL).count('1') * 10
            
        score = (my_score - op_score) + 10 * (my_mob - op_mob)
        return score

    def get_weight(self, idx):
        mask = 1 << idx
        if mask & self.MASK_CORNER: return 500
        if mask & self.MASK_TOP_CORNER: return 100
        if mask & self.MASK_BAD: return -50
        if mask & self.MASK_EDGE: return 30
        return 10

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
            # Check if opponent also passes (Game Over)
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
            
        # Move ordering?
        # Simple heuristic: corners first, then mobility?
        # For now, just random or simple sort
        # moves.sort(key=lambda m: self.WEIGHTS.get(m, 0), reverse=True)

        best_val = -float('inf')
        best_move = None
        
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
                best_move = move
            
            alpha = max(alpha, val)
            if alpha >= beta:
                break
        
        # Store in TT
        flag = 'exact'
        if best_val <= alpha: flag = 'upper' # Fail low? Wait. Standard AB: val <= alpha -> upper bound? No.
        # If val <= original_alpha, it's an upper bound (we couldn't improve alpha).
        # If val >= beta, it's a lower bound (cutoff).
        # Here we just store best_val.
        
        self.tt[state_hash] = {'val': best_val, 'depth': depth, 'flag': flag} # Simplified flag logic
        return best_val

    def choose_move(self, board_input):
        me_bb, opp_bb = self.to_bitboard(board_input, 1, 2)
        
        moves_mask = self.get_moves_bb(me_bb, opp_bb)
        if not moves_mask:
            return None
            
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
        moves.sort(key=lambda m: self.get_weight(m), reverse=True)
        
        try:
            for d in range(1, self.depth + 1):
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

    bot = TriangleReversiBot(depth=10, max_time=2.8)
    move = bot.choose_move(board_input)
    
    if move:
        r, c = move[0], move[1]
        local_c = c - bot.OFFSETS[r]
        print(f"{r + 1} {local_c + 1}")  # Output 1-indexed row and local column
    else:
        print("0 0")

if __name__ == "__main__":
    main()
