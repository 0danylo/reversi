def test_visualize_moves():
    import reversi
    
    # Sample jagged board from earlier (rows lengths 8,10,12,14,14,12,10,8)
    board = [
              [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,2,1,0,0,0,0,0],
        [0,0,0,0,0,0,2,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,2,1,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0],
    ]
    
    # ensure reversi helper uses this board for on-board checks
    reversi.board_global = board
    
    moves = reversi.get_legal_moves(board, me=1, opp=2)
    
    # Create a copy to mark moves
    display_board = [row[:] for row in board]
    
    # Mark legal moves with 'X' (we'll use -1 to represent it)
    move_positions = set()
    for r, c, flips in moves:
        move_positions.add((r, c)) 
        display_board[r][c] = -1  # -1 represents a legal move
    
    # Print the board
    max_len = 14
    print("\nBoard visualization (0=empty, 1=player, 2=opponent, X=legal move):\n")
    
    for r, row in enumerate(display_board):
        row_len = len(row)
        offset = (max_len - row_len) // 2
        
        # Print leading spaces for centering
        print(" " * offset * 2, end="")
        
        # Print the row
        for c, val in enumerate(row):
            if val == 0:
                print(". ", end="")
            elif val == 1:
                print("1 ", end="")
            elif val == 2:
                print("2 ", end="")
            elif val == -1:
                print("X ", end="")
        print(f"  (row {r}, len={row_len})")
    
    print(f"\nFound {len(moves)} legal moves:")
    for r, c, flips in moves:
        print(f"  Row {r+1}, Col {c+1} - flips {[(r+1, c+1) for (r,c) in flips]}") # add one because input for tournament is 1-indexed

if __name__ == '__main__':
    test_visualize_moves()