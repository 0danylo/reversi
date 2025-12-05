
import random
import reversi
import strategies
import engine
from copy import deepcopy

def test_bitboard_diamond():
    # Create a bitboard strategy instance
    bb_strat = strategies.AlphaBetaBitboardStrategy()
    
    # Run 100 random tests
    for i in range(100):
        # Create a diamond board (jagged)
        row_lengths = [8, 10, 12, 14, 14, 12, 10, 8]
        board = []
        for ln in row_lengths:
            board.append([0] * ln)
            
        # Place some random pieces
        for r in range(8):
            for c in range(len(board[r])):
                if random.random() < 0.3:
                    board[r][c] = random.choice([1, 2])
        
        # Ensure center is standard-ish
        # Row 3 (len 14): center is 6,7
        # Row 4 (len 14): center is 6,7
        board[3][6], board[4][7] = 2, 2
        board[3][7], board[4][6] = 1, 1
        
        reversi.board_global = board
        
        # 1. Test Move Generation
        legal_moves = reversi.get_legal_moves(board, 1, 2)
        legal_moves_set = set((m[0], m[1]) for m in legal_moves)
        
        me_bb, opp_bb = bb_strat.to_bitboard(board, 1, 2)
        moves_mask = bb_strat.get_moves_bb(me_bb, opp_bb)
        
        bb_moves_set = set()
        temp_mask = moves_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            idx = lsb.bit_length() - 1
            r = idx // bb_strat.WIDTH
            c = (idx % bb_strat.WIDTH) - bb_strat.ROW_OFFSETS[r]
            bb_moves_set.add((r, c))
            temp_mask ^= lsb
            
        if legal_moves_set != bb_moves_set:
            print(f"Mismatch in moves at iteration {i}")
            print("Legal:", legal_moves_set)
            print("Bitboard:", bb_moves_set)
            # Print board
            for r in range(8):
                print(board[r])
            return

        # 2. Test Flips and Application
        for r, c in legal_moves_set:
            # Apply using engine
            b_copy = deepcopy(board)
            move_tuple = next(m for m in legal_moves if m[0]==r and m[1]==c)
            engine.apply_move(b_copy, move_tuple, 1)
            
            # Apply using bitboard
            idx = r * bb_strat.WIDTH + (c + bb_strat.ROW_OFFSETS[r])
            flips = bb_strat.get_flips_bb(idx, me_bb, opp_bb)
            new_me = me_bb | (1 << idx) | flips
            new_opp = opp_bb & ~flips
            
            # Convert back to board to compare
            b_copy_me, b_copy_opp = bb_strat.to_bitboard(b_copy, 1, 2)
            
            if new_me != b_copy_me or new_opp != b_copy_opp:
                print(f"Mismatch in application at {r},{c}")
                print(f"Expected Me: {bin(b_copy_me)}")
                print(f"Got Me:      {bin(new_me)}")
                return

    print("All 100 diamond tests passed!")

if __name__ == "__main__":
    test_bitboard_diamond()
