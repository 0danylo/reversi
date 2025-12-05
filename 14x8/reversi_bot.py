#!/usr/bin/env python3
import sys
import time
import copy
import argparse
import pickle
import os
from collections import OrderedDict

import strategies
import reversi
import engine

class ReversiBot:
    def __init__(self, depth=100, max_time=2.9, max_tt_size=50000, tt_file="tt.pkl"):
        self.start_init_time = time.time()
        self.depth = depth
        self.max_time = max_time
        self.max_tt_size = max_tt_size
        self.tt_file = tt_file
        
        if os.path.exists(self.tt_file):
            try:
                with open(self.tt_file, "rb") as f:
                    self.tt = pickle.load(f)
            except Exception:
                self.tt = OrderedDict()
        else:
            self.tt = OrderedDict()
        
        self.load_duration = time.time() - self.start_init_time
        # Bitboard constants
        self.ROW_OFFSETS = [3, 2, 1, 0, 0, 1, 2, 3]
        self.WIDTH = 15
        self.HEIGHT = 8
        self.TOTAL_BITS = self.WIDTH * self.HEIGHT
        
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

        self.SHIFTS = [1, -1, 15, -15, 16, 14, -14, -16]

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
        
        self.CORNER_ADJ_MASK = 0
        self.CORNER_TO_ADJ = {} 
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
        my_count = bin(me_bb).count('1')
        opp_count = bin(opp_bb).count('1')
        score = (my_count - opp_count) * 10

        my_corners = me_bb & self.CORNER_MASK
        opp_corners = opp_bb & self.CORNER_MASK
        score += bin(my_corners).count('1') * 1000
        score -= bin(opp_corners).count('1') * 1000

        my_edges = me_bb & self.EDGE_MASK
        opp_edges = opp_bb & self.EDGE_MASK
        score += bin(my_edges).count('1') * 50
        score -= bin(opp_edges).count('1') * 50

        empty_corners = self.CORNER_MASK & ~(me_bb | opp_bb)
        if empty_corners:
            for c_idx, adj_list in self.CORNER_TO_ADJ.items():
                if (empty_corners & (1 << c_idx)):
                    for adj in adj_list:
                        if (me_bb & (1 << adj)):
                            score -= 800
                        elif (opp_bb & (1 << adj)):
                            score += 800

        my_moves = self.get_moves_bb(me_bb, opp_bb)
        opp_moves = self.get_moves_bb(opp_bb, me_bb)
        score += (bin(my_moves).count('1') - bin(opp_moves).count('1')) * 20
        
        return score

    def save_tt(self):
        try:
            with open(self.tt_file, "wb") as f:
                pickle.dump(self.tt, f)
        except Exception:
            pass

    def choose_move(self, board, me=1, opp=2):
        # Adjust max_time to account for loading and saving overhead
        # Reserve time for saving (assume similar to load time + buffer)
        save_buffer = max(0.05, self.load_duration * 1.2)
        effective_max_time = self.max_time - self.load_duration - save_buffer
        
        if effective_max_time < 0.1:
            effective_max_time = 0.1 # Minimum search time
            
        start_time = time.time()
        me_bb, opp_bb = self.to_bitboard(board, me, opp)
        
        # Check if any moves exist
        moves_mask_initial = self.get_moves_bb(me_bb, opp_bb)
        if not moves_mask_initial:
            return None

        best_move_idx = None
        
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
            if effective_max_time > 0:
                if time.time() - start_time > effective_max_time:
                    raise TimeoutError()

            alpha_orig = alpha
            
            tt_key = (my_b, op_b)
            tt_entry = self.tt.get(tt_key)
            if tt_entry:
                self.tt.move_to_end(tt_key)
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
                        if len(self.tt) > self.max_tt_size:
                            self.tt.popitem(last=False)
                        return val
                val = self.evaluate_bb(my_b if maximizing else op_b, op_b if maximizing else my_b)
                self.tt[tt_key] = (depth, val, EXACT)
                if len(self.tt) > self.max_tt_size:
                    self.tt.popitem(last=False)
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
            if len(self.tt) > self.max_tt_size:
                self.tt.popitem(last=False)
            return value

        try:
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
                
        except TimeoutError:
            pass

        if best_move_idx is not None:
            r = best_move_idx // self.WIDTH
            c_aligned = best_move_idx % self.WIDTH
            c = c_aligned - self.ROW_OFFSETS[r]
            self.save_tt()
            return (r, c, [])
            
        # Fallback if something went wrong but we had moves initially
        # Just pick the first one
        temp_mask = moves_mask_initial
        lsb = temp_mask & -temp_mask
        idx = lsb.bit_length() - 1
        r = idx // self.WIDTH
        c_aligned = idx % self.WIDTH
        c = c_aligned - self.ROW_OFFSETS[r]
        self.save_tt()
        return (r, c, [])

def read_board():
    row_lengths = [8,10,12,14,14,12,10,8]
    board = []
    for i, ln in enumerate(row_lengths):
        try:
            line = input()
        except EOFError:
            line = ''
        parts = line.strip().split()
        vals = []
        for j, tok in enumerate(parts[:ln]):
            try:
                vals.append(int(tok))
            except ValueError:
                vals.append(0)
        board.append(vals)
    return board

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', default='ab_bit_tt', help='Strategy to use: ab2, ab_tt, ab_bit, ab_bit_tt')
    parser.add_argument('--depth', type=int, default=100, help='Search depth')
    parser.add_argument('--time', type=float, default=2.9, help='Time limit')
    args = parser.parse_args()

    board = read_board()
    if not board:
        return

    if args.strategy == 'ab_bit_tt':
        # Use the local implementation
        bot = ReversiBot(depth=args.depth, max_time=args.time)
        move = bot.choose_move(board)
    else:
        # Use strategies from strategies.py
        # Map names to classes
        strat_map = {
            'ab2': strategies.AlphaBetaImprovedStrategy,
            'ab_tt': strategies.AlphaBetaTTStrategy,
            'ab_bit': strategies.AlphaBetaBitboardStrategy,
            'ab_bit_tt_ref': strategies.AlphaBetaBitboardTTStrategy,
            'ab_bit_rl': strategies.AlphaBetaBitboardRLStrategy
        }
        
        if args.strategy not in strat_map:
            # Fallback or error
            # Try to find in strategies module by name
            pass
            
        cls = strat_map.get(args.strategy)
        if cls:
            bot = cls(depth=args.depth, max_time=args.time)
            # strategies.py classes expect board in choose_move
            # and return (r, c, flips) or None
            move = bot.choose_move(board)
        else:
            # Default to local if unknown? Or error?
            # Let's default to local for safety if it matches nothing, or print error
            sys.stderr.write(f"Unknown strategy: {args.strategy}\n")
            return

    if move is None:
        print("0 0")
    else:
        r, c, _ = move
        print(f"{r+1} {c+1}")

if __name__ == '__main__':
    main()
