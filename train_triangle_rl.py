import time
import random
import copy
import sys
from collections import OrderedDict

# --- Game Logic (Fast, In-Process) ---

ROWS = 10
OFFSETS = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
LENGTHS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
WIDTH = 22
TOTAL_BITS = ROWS * WIDTH

# Precompute Masks
VALID_MASK = 0
ROW_MASKS = []
for r in range(ROWS):
    row_mask = 0
    for c in range(OFFSETS[r], OFFSETS[r] + LENGTHS[r]):
        idx = r * WIDTH + c
        VALID_MASK |= (1 << idx)
        row_mask |= (1 << idx)
    ROW_MASKS.append(row_mask)

SHIFTS = [1, -1, WIDTH, -WIDTH, WIDTH+1, WIDTH-1, -WIDTH+1, -WIDTH-1]

def get_initial_board_bb():
    # 0: empty, 1: p1, 2: p2
    # Board is 10 rows.
    # Center is row 4 and 5.
    # Row 4 (len 10, offset 5): indices 5..14. Center 9, 10.
    # Row 5 (len 12, offset 4): indices 4..15. Center 9, 10.
    
    # (4, 9): 2, (4, 10): 1
    # (5, 9): 1, (5, 10): 2
    
    p1_bb = 0
    p2_bb = 0
    
    # P1: (4, 10), (5, 9)
    p1_bb |= (1 << (4 * WIDTH + 10))
    p1_bb |= (1 << (5 * WIDTH + 9))
    
    # P2: (4, 9), (5, 10)
    p2_bb |= (1 << (4 * WIDTH + 9))
    p2_bb |= (1 << (5 * WIDTH + 10))
    
    return p1_bb, p2_bb

def get_moves_bb(me_bb, opp_bb):
    empty = VALID_MASK & ~(me_bb | opp_bb)
    moves = 0
    for shift in SHIFTS:
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

def get_flips_bb(move_idx, me_bb, opp_bb):
    flips = 0
    for shift in SHIFTS:
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

# --- Bot Logic ---

class ParametricBot:
    def __init__(self, weights, depth=2):
        self.weights = weights # dict
        self.depth = depth
        self.tt = {}
        
        # Masks
        self.MASK_CORNER = 0
        self.MASK_TOP_CORNER = 0
        self.MASK_EDGE = 0
        self.MASK_NORMAL = 0
        
        CORNERS = [(9, 0), (9, 19)]
        TOP_CORNERS = [(0, 9), (0, 10)]
        
        # Dynamic Corner Adjacency
        self.CORNER_TO_ADJ = {}
        # Bottom Left (9, 0) -> Neighbors (9, 1), (8, 1)
        c_idx_bl = 9 * WIDTH + 0
        adj_bl = [9 * WIDTH + 1, 8 * WIDTH + 1]
        self.CORNER_TO_ADJ[c_idx_bl] = adj_bl
        
        # Bottom Right (9, 19) -> Neighbors (9, 18), (8, 18)
        c_idx_br = 9 * WIDTH + 19
        adj_br = [9 * WIDTH + 18, 8 * WIDTH + 18]
        self.CORNER_TO_ADJ[c_idx_br] = adj_br
        
        for r in range(ROWS):
            for c in range(OFFSETS[r], OFFSETS[r] + LENGTHS[r]):
                idx = r * WIDTH + c
                if (r, c) in CORNERS:
                    self.MASK_CORNER |= (1 << idx)
                elif (r, c) in TOP_CORNERS:
                    self.MASK_TOP_CORNER |= (1 << idx)
                elif c == OFFSETS[r] or c == OFFSETS[r] + LENGTHS[r] - 1 or r == 9:
                    self.MASK_EDGE |= (1 << idx)
                else:
                    self.MASK_NORMAL |= (1 << idx)

    def evaluate(self, me_bb, opp_bb):
        my_moves = get_moves_bb(me_bb, opp_bb)
        op_moves = get_moves_bb(opp_bb, me_bb)
        
        my_mob = bin(my_moves).count('1')
        op_mob = bin(op_moves).count('1')
        
        my_score = 0
        op_score = 0
        
        my_score += bin(me_bb & self.MASK_CORNER).count('1') * self.weights['corner']
        op_score += bin(opp_bb & self.MASK_CORNER).count('1') * self.weights['corner']
        
        my_score += bin(me_bb & self.MASK_TOP_CORNER).count('1') * self.weights['top_corner']
        op_score += bin(opp_bb & self.MASK_TOP_CORNER).count('1') * self.weights['top_corner']
        
        my_score += bin(me_bb & self.MASK_EDGE).count('1') * self.weights['edge']
        op_score += bin(opp_bb & self.MASK_EDGE).count('1') * self.weights['edge']
        
        # Dynamic Bad Square Penalty
        for c_idx, adj_list in self.CORNER_TO_ADJ.items():
            if not ((me_bb | opp_bb) & (1 << c_idx)):
                for adj in adj_list:
                    if me_bb & (1 << adj):
                        my_score += self.weights['bad']
                    if opp_bb & (1 << adj):
                        op_score += self.weights['bad']
        
        my_score += bin(me_bb & self.MASK_NORMAL).count('1') * self.weights['normal']
        op_score += bin(opp_bb & self.MASK_NORMAL).count('1') * self.weights['normal']
            
        score = (my_score - op_score) + self.weights['mobility'] * (my_mob - op_mob)
        return score

    def get_move_weight(self, idx, me_bb, opp_bb):
        mask = 1 << idx
        score = 0
        if mask & self.MASK_CORNER: score += self.weights['corner']
        elif mask & self.MASK_TOP_CORNER: score += self.weights['top_corner']
        elif mask & self.MASK_EDGE: score += self.weights['edge']
        else: score += self.weights['normal']
        
        # Dynamic Bad Square Penalty
        for c_idx, adj_list in self.CORNER_TO_ADJ.items():
            if idx in adj_list:
                if not ((me_bb | opp_bb) & (1 << c_idx)):
                    score += self.weights['bad']
                    
        return score

    def alphabeta(self, me_bb, opp_bb, depth, alpha, beta, maximizing):
        if depth == 0:
            return self.evaluate(me_bb, opp_bb)

        moves_mask = get_moves_bb(me_bb, opp_bb)
        if not moves_mask:
            op_moves = get_moves_bb(opp_bb, me_bb)
            if not op_moves:
                diff = bin(me_bb).count('1') - bin(opp_bb).count('1')
                return 100000 if diff > 0 else -100000 if diff < 0 else 0
            return -self.alphabeta(opp_bb, me_bb, depth, -beta, -alpha, not maximizing)

        moves = []
        temp = moves_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            moves.append(idx)
            temp ^= lsb
            
        moves.sort(key=lambda m: self.get_move_weight(m, me_bb, opp_bb), reverse=True)

        best_val = -float('inf')
        
        for move in moves:
            flips = get_flips_bb(move, me_bb, opp_bb)
            new_me = me_bb | (1 << move) | flips
            new_op = opp_bb & ~flips
            
            val = -self.alphabeta(new_op, new_me, depth - 1, -beta, -alpha, not maximizing)
            
            if val > best_val:
                best_val = val
            
            alpha = max(alpha, val)
            if alpha >= beta:
                break
        
        return best_val

    def choose_move(self, me_bb, opp_bb):
        moves_mask = get_moves_bb(me_bb, opp_bb)
        if not moves_mask:
            return None
            
        moves = []
        temp = moves_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            moves.append(idx)
            temp ^= lsb
            
        moves.sort(key=lambda m: self.get_move_weight(m, me_bb, opp_bb), reverse=True)
        
        best_move = moves[0]
        best_val = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        
        for move in moves:
            flips = get_flips_bb(move, me_bb, opp_bb)
            new_me = me_bb | (1 << move) | flips
            new_op = opp_bb & ~flips
            
            val = -self.alphabeta(new_op, new_me, self.depth - 1, -beta, -alpha, False)
            
            if val > best_val:
                best_val = val
                best_move = move
            
            alpha = max(alpha, val)
            
        return best_move

def play_game(bot1, bot2):
    p1_bb, p2_bb = get_initial_board_bb()
    current_player = 1
    skipped_last = False
    
    while True:
        if current_player == 1:
            me, opp = p1_bb, p2_bb
            bot = bot1
        else:
            me, opp = p2_bb, p1_bb
            bot = bot2
            
        move = bot.choose_move(me, opp)
        
        if move is None:
            if skipped_last:
                break
            skipped_last = True
        else:
            skipped_last = False
            flips = get_flips_bb(move, me, opp)
            me = me | (1 << move) | flips
            opp = opp & ~flips
            
            if current_player == 1:
                p1_bb, p2_bb = me, opp
            else:
                p2_bb, p1_bb = opp, me
                
        current_player = 3 - current_player
        
    score1 = bin(p1_bb).count('1')
    score2 = bin(p2_bb).count('1')
    return score1, score2

def train():
    # Initial weights
    base_weights = {
        'corner': 500,
        'top_corner': 97,
        'edge': 26,
        'bad': -50,
        'normal': 13,
        'mobility': 25
    }
    
    print("Starting training...")
    best_weights = base_weights
    
    # Simple Hill Climbing / Evolution
    # Generate N variants, play against best.
    
    generations = 8
    population_size = 8
    
    for gen in range(generations):
        print(f"Generation {gen+1}")
        population = []
        
        # Add current best
        population.append(best_weights)
        
        # Generate mutants
        for _ in range(population_size - 1):
            mutant = best_weights.copy()
            # Mutate one or two parameters
            num_mutations = random.randint(1, 2)
            keys = list(mutant.keys())
            for _ in range(num_mutations):
                k = random.choice(keys)
                # Add noise
                noise = random.randint(-15, 15)
                mutant[k] += noise
            population.append(mutant)
            
        # Tournament
        scores = [0] * len(population)
        
        # Play everyone against everyone (or just against best)
        # For speed, let's play everyone against the current best (index 0)
        # If they beat the best, they get points.
        
        # Actually, let's do a round robin if small population
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                bot1 = ParametricBot(population[i], depth=3) # Increased depth
                bot2 = ParametricBot(population[j], depth=3)
                
                # Play 1 vs 2
                s1, s2 = play_game(bot1, bot2)
                if s1 > s2: scores[i] += 1
                elif s2 > s1: scores[j] += 1
                
                # Play 2 vs 1 (swap sides)
                s1_swap, s2_swap = play_game(bot2, bot1)
                if s1_swap > s2_swap: scores[j] += 1
                elif s2_swap > s1_swap: scores[i] += 1
        
        # Find best
        best_idx = scores.index(max(scores))
        best_weights = population[best_idx]
        print(f"Best weights gen {gen+1}: {best_weights} (Score: {scores[best_idx]})")
        
    print("Final Best Weights:", best_weights)

if __name__ == "__main__":
    train()
