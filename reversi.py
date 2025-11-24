import sys

def read_board():
    # Read exactly 8 lines
    row_lengths = [8,10,12,14,14,12,10,8]
    board = []

    for i, ln in enumerate(row_lengths):
        try:
            line = input()
        except EOFError:
            line = ''
        parts = line.strip().split()

        # parse up to expected length for this row
        vals = []
        for j, tok in enumerate(parts[:ln]):
            try:
                vals.append(int(tok))
            except ValueError:
                vals.append(0)

        board.append(vals)

    # print([print(b) for b in board], len(board), [len(b) for b in board])
    return board


DELTA = [(-1,-1),(-1, 0),(-1, 1),\
         ( 0,-1),        ( 0, 1),\
         ( 1,-1),( 1, 0),( 1, 1)]

# module-level reference used by is_on_board to handle jagged rows
board_global = None

def is_on_board(r,c):
    if board_global is None:
        return False

    if not (0 <= r < len(board_global)):
        return False

    return 0 <= c < len(board_global[r])

def get_legal_moves(board, me=1, opp=2):
    global board_global
    board_global = board  # Set global so is_on_board works
    
    moves = []  # (r,c,flipped_positions_list)
    
    # Calculate row offsets for diamond centering
    # Compute max row length from the provided board so both diamond-shaped
    # and rectangular boards are supported. For diamond layouts this will
    # be 14, while for a regular 8x8 board it will be 8 (and offsets zero).
    max_len = max(len(row) for row in board) if board else 0
    row_offsets = [(max_len - len(board[r])) // 2 for r in range(len(board))]
    
    # Iterate through all rows
    for r in range(len(board)):
        # Iterate through all columns in this row
        for c in range(len(board[r])):
            # Skip if cell is occupied
            if board[r][c] != 0:
                continue
            
            total_flips = []
            
            # Check all 8 directions
            for dr, dc in DELTA:
                flips = []
                rr = r + dr
                cc = c + dc
                
                # Adjust column for offset when moving to a different row
                if dr != 0 and is_on_board(rr, 0):
                    cc += (row_offsets[r] - row_offsets[rr])
                
                # Walk in this direction, collecting opponent pieces
                while is_on_board(rr, cc) and board[rr][cc] == opp:
                    flips.append((rr, cc))
                    
                    prev_rr = rr
                    rr = rr + dr
                    cc = cc + dc
                    
                    # check if next row is on board
                    if dr != 0 and is_on_board(rr, 0):
                        cc += (row_offsets[prev_rr] - row_offsets[rr])
                
                # If we found opponent pieces AND ended on our own piece, these flips are valid
                if flips and is_on_board(rr, cc) and board[rr][cc] == me:
                    total_flips.extend(flips)
            
            # If we can flip at least one piece, this is a legal move
            if total_flips:
                moves.append((r, c, total_flips))
    
    return moves

def score_move(board, move, me=1, opp=2):
    r,c,flips = move
    score = 0
    
    # prioritize corners heavily (corners are ends of first and last rows)
    corners = {(0, 0), (0, len(board[0]) - 1), (len(board) - 1, 0), (len(board) - 1, len(board[-1]) - 1)}
    if (r,c) in corners:
        score += 10000

    # edges: any cell on outermost row or at start/end of its row
    if r == 0 or r == len(board) - 1 or c == 0 or c == len(board[r]) - 1:
        score += 200
    
    # number of disks flipped
    score += len(flips) * 10

    # strongly avoid moves that give opponent corner next turn: positions adjacent to corners
    corner_adjacent = set()
    for cr,cc in corners:
        for dr,dc in DELTA:
            nr,nc = cr+dr, cc+dc
            if is_on_board(nr,nc):
                corner_adjacent.add((nr,nc))
    
    # If our move is adjacent to any corner and that corner is empty, penalize
    if (r,c) in corner_adjacent:
        for cr,cc in corners:
            if is_on_board(cr,cc) and board[cr][cc] == 0:
                # penalize more strongly if that corner is currently empty
                score -= 800

    # small tie-breaker to prefer top-left earlier (deterministic)
    score -= (r*100 + c) * 0.001
    return score

def choose_move(board):
    global board_global
    board_global = board
    moves = get_legal_moves(board, me=1, opp=2)
    if not moves:
        return None

    best = None
    best_score = -10**9
    for m in moves:
        s = score_move(board, m, me=1, opp=2)
        if s > best_score:
            best_score = s
            best = m

    return best

def main():
    board = read_board()
    if board is None:
        return
    
    move = choose_move(board)
    if move is None:
        print("0 0")
    else:
        r, c, _ = move
        # output 1-based row and column
        print(f"{r+1} {c+1}")

if __name__ == '__main__':
    main()
