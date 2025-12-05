import sys
import time
import random


DELTA = [(-1,-1),(-1, 0),(-1, 1),
        ( 0,-1),        ( 0, 1),
        ( 1,-1),( 1, 0),( 1, 1)]


def is_on_board(r, c, board_len):
   if not (0 <= r < board_len):
       return False
   row_lens = [8, 10, 12, 14, 14, 12, 10, 8]
   if 0 <= r < 8:
       return 0 <= c < row_lens[r]
   return False


def get_legal_moves(board, me=1, opp=2):
   moves = []
   row_offsets = [3, 2, 1, 0, 0, 1, 2, 3]
  
   for r in range(len(board)):
       for c in range(len(board[r])):
           if board[r][c] != 0:
               continue
          
           total_flips = []
           for dr, dc in DELTA:
               flips = []
               rr, cc = r + dr, c + dc
              
               if dr != 0:
                   if 0 <= rr < 8:
                       cc += (row_offsets[r] - row_offsets[rr])


               while is_on_board(rr, cc, 8) and board[rr][cc] == opp:
                   flips.append((rr, cc))
                   prev_rr = rr
                   rr += dr
                   cc += dc
                   if dr != 0 and 0 <= rr < 8:
                       cc += (row_offsets[prev_rr] - row_offsets[rr])
              
               if flips and is_on_board(rr, cc, 8) and board[rr][cc] == me:
                   total_flips.extend(flips)
          
           if total_flips:
               moves.append((r, c, total_flips))
   return moves


class Strategy:
   def choose_move(self, board, me=1, opp=2):
       raise NotImplementedError


class AlphaBetaStrategyV4(Strategy):
   def __init__(self, depth=5, max_time=2.8):
       self.depth = depth
       self.max_time = max_time
       self.safety_buffer = 0.
      
       self.ROW_OFFSETS = [3, 2, 1, 0, 0, 1, 2, 3]
       self.WIDTH = 14
       self.HEIGHT = 8
       self.TOTAL_BITS = self.WIDTH * self.HEIGHT
       self.VALID_MASK = 0
      
       for r in range(self.HEIGHT):
           row_len = [8, 10, 12, 14, 14, 12, 10, 8][r]
           offset = self.ROW_OFFSETS[r]
           for c in range(row_len):
               idx = r * self.WIDTH + (c + offset)
               self.VALID_MASK |= (1 << idx)


       self.SHIFTS = [1, -1, 14, -14, 15, 13, -13, -15]


       self.CORNER_MASK = 0
       self.EDGE_MASK = 0
       corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
      
       for r, c in corners:
           idx = r * self.WIDTH + (c + self.ROW_OFFSETS[r])
           self.CORNER_MASK |= (1 << idx)


       for r in range(self.HEIGHT):
           row_len = [8, 10, 12, 14, 14, 12, 10, 8][r]
           offset = self.ROW_OFFSETS[r]
           for c in range(row_len):
               if r == 0 or r == 7 or c == 0 or c == row_len - 1:
                   idx = r * self.WIDTH + (c + offset)
                   self.EDGE_MASK |= (1 << idx)
      
       self.CORNER_ADJ_MASK = 0
       self.CORNER_TO_ADJ = {}
      
       for r, c in corners:
           c_idx = r * self.WIDTH + (c + self.ROW_OFFSETS[r])
           adj_indices = []
           for shift in self.SHIFTS:
               adj = c_idx + shift
               if 0 <= adj < self.TOTAL_BITS and (self.VALID_MASK & (1 << adj)):
                   self.CORNER_ADJ_MASK |= (1 << adj)
                   adj_indices.append(adj)
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
               if shift > 0: mask = (mask << shift)
               else: mask = (mask >> -shift)
              
               if not (mask & self.VALID_MASK): break
               if mask & opp_bb:
                   potential_flips |= mask
               elif mask & me_bb:
                   flips |= potential_flips
                   break
               else: break
       return flips


   def evaluate_bb(self, me_bb, opp_bb):
       my_count = bin(me_bb).count('1')
       opp_count = bin(opp_bb).count('1')
       score = (my_count - opp_count) * 10


       my_moves_mask = self.get_moves_bb(me_bb, opp_bb)
       opp_moves_mask = self.get_moves_bb(opp_bb, me_bb)
       my_mob = bin(my_moves_mask).count('1')
       opp_mob = bin(opp_moves_mask).count('1')
       score += (my_mob - opp_mob) * 20


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
       return score


   def choose_move(self, board, me=1, opp=2):
       legal_moves_list = get_legal_moves(board, me, opp)
       if not legal_moves_list:
           return None


       start_time = time.time()
       me_bb, opp_bb = self.to_bitboard(board, me, opp)
      
       empty_squares = sum(row.count(0) for row in board)
      
       if empty_squares <= 14:
           max_depth = 20
       else:
           max_depth = self.depth


       best_move_idx = None
      
       def get_move_score(idx, my_b, op_b):
           mask = (1 << idx)
           s = 0
           if mask & self.CORNER_MASK: s += 10000
           if mask & self.EDGE_MASK: s += 200
           s -= idx * 0.01
           return s


       def alphabeta_bb(my_b, op_b, depth, alpha, beta, maximizing):
           if (time.time() - start_time > self.max_time - self.safety_buffer):
               raise TimeoutError()


           moves_mask = self.get_moves_bb(my_b, op_b)
          
           if depth == 0 or moves_mask == 0:
               if moves_mask == 0:
                   opp_moves_mask = self.get_moves_bb(op_b, my_b)
                   if opp_moves_mask == 0:
                       my_c = bin(my_b).count('1')
                       op_c = bin(op_b).count('1')
                       return (my_c - op_c) * 10000
                   return alphabeta_bb(op_b, my_b, depth, alpha, beta, not maximizing)
               return self.evaluate_bb(my_b if maximizing else op_b, op_b if maximizing else my_b)


           move_indices = []
           temp_mask = moves_mask
           while temp_mask:
               lsb = temp_mask & -temp_mask
               idx = lsb.bit_length() - 1
               move_indices.append(idx)
               temp_mask ^= lsb
          
           move_indices.sort(key=lambda idx: get_move_score(idx, my_b, op_b), reverse=True)
          
           if maximizing:
               value = -float('inf')
               for idx in move_indices:
                   flips = self.get_flips_bb(idx, my_b, op_b)
                   new_my = my_b | (1 << idx) | flips
                   new_op = op_b & ~flips
                   val = alphabeta_bb(new_op, new_my, depth - 1, alpha, beta, False)
                   if val > value: value = val
                   if value > alpha: alpha = value
                   if alpha >= beta: break
               return value
           else:
               value = float('inf')
               for idx in move_indices:
                   flips = self.get_flips_bb(idx, my_b, op_b)
                   new_my = my_b | (1 << idx) | flips
                   new_op = op_b & ~flips
                   val = alphabeta_bb(new_op, new_my, depth - 1, alpha, beta, True)
                   if val < value: value = val
                   if value < beta: beta = value
                   if alpha >= beta: break
               return value


       try:
           for depth_limit in range(1, max_depth + 1):
               moves_mask = self.get_moves_bb(me_bb, opp_bb)
               if not moves_mask: break


               move_indices = []
               temp_mask = moves_mask
               while temp_mask:
                   lsb = temp_mask & -temp_mask
                   idx = lsb.bit_length() - 1
                   move_indices.append(idx)
                   temp_mask ^= lsb
              
               move_indices.sort(key=lambda idx: get_move_score(idx, me_bb, opp_bb), reverse=True)


               current_best = None
               best_val = -float('inf')
               alpha = -float('inf')
               beta = float('inf')


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


def read_board():
   row_lengths = [8,10,12,14,14,12,10,8]
   board = []
   for i, ln in enumerate(row_lengths):
       try:
           line = sys.stdin.readline()
       except EOFError:
           break
       parts = line.strip().split()
       vals = []
       for j, tok in enumerate(parts[:ln]):
           try:
               vals.append(int(tok))
           except ValueError:
               vals.append(0)
       if len(vals) < ln:
           vals += [0] * (ln - len(vals))
       board.append(vals)
   return board


def main():
   board = read_board()
   if not board:
       return
   bot = AlphaBetaStrategyV4(depth=100, max_time=2.9)
   move = bot.choose_move(board, me=1, opp=2)
   if move:
       r, c, _ = move
       print(f"{r+1} {c+1}")


if __name__ == "__main__":
   main()

