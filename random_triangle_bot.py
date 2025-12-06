import sys
import random

# Board configuration
ROWS = 10
OFFSETS = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
LENGTHS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

def is_valid_coord(r, c):
    if 0 <= r < ROWS:
        offset = OFFSETS[r]
        length = LENGTHS[r]
        if offset <= c < offset + length:
            return True
    return False

def get_directions():
    return [(-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1), (1, 0), (1, 1)]

def get_flips(board, player, r, c):
    if not is_valid_coord(r, c):
        return []
    
    local_c = c - OFFSETS[r]
    if board[r][local_c] != 0:
        return []
    
    opponent = 3 - player
    flips = []
    
    for dr, dc in get_directions():
        current_flips = []
        cr, cc = r + dr, c + dc
        while is_valid_coord(cr, cc):
            local_cc = cc - OFFSETS[cr]
            val = board[cr][local_cc]
            if val == opponent:
                current_flips.append((cr, cc))
            elif val == player:
                flips.extend(current_flips)
                break
            else: # empty
                break
            cr += dr
            cc += dc
            
    return flips

def main():
    # Read board from stdin
    board = []
    try:
        for r in range(ROWS):
            line = sys.stdin.readline()
            if not line:
                break
            parts = line.strip().split()
            row_vals = [int(x) for x in parts]
            board.append(row_vals)
    except Exception:
        pass

    if len(board) != ROWS:
        return

    # Find valid moves
    # My color is always 1 because the server swaps it for me
    my_color = 1
    valid_moves = []
    
    for r in range(ROWS):
        offset = OFFSETS[r]
        length = LENGTHS[r]
        for local_c in range(length):
            c = offset + local_c
            if board[r][local_c] == 0:
                if get_flips(board, my_color, r, c):
                    valid_moves.append((r, c))
    
    if valid_moves:
        r, c = random.choice(valid_moves)
        print(f"{r} {c}")
    else:
        # Should not happen if server checks for moves, but just in case
        print("0 0")

if __name__ == "__main__":
    main()
