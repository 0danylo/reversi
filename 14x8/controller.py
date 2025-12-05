#!/usr/bin/env python3
import subprocess
import sys
import time
import reversi
import engine

def get_initial_board(filename="start_board.txt"):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Could not find board file {filename}")
        sys.exit(1)
    
    row_lengths = [8,10,12,14,14,12,10,8]
    board = []
    for i, ln in enumerate(row_lengths):
        if i < len(lines):
            parts = lines[i].strip().split()
        else:
            parts = []
        vals = []
        for x in parts[:ln]:
            try:
                vals.append(int(x))
            except ValueError:
                vals.append(0)
        
        if len(vals) < ln:
            vals += [0] * (ln - len(vals))
        board.append(vals)
    return board

def board_to_string(board):
    lines = []
    for row in board:
        lines.append(" ".join(map(str, row)))
    return "\n".join(lines)

def swap_board(board):
    new_board = []
    for row in board:
        new_row = []
        for val in row:
            if val == 1:
                new_row.append(2)
            elif val == 2:
                new_row.append(1)
            else:
                new_row.append(0)
        new_board.append(new_row)
    return new_board

def play_match(bot1_cmd, bot2_cmd, board_file="start_board.txt"):
    board = get_initial_board(board_file)
    current_player = 1 # 1 is Black (bot1), 2 is White (bot2)
    
    # Map player to command
    bots = {1: bot1_cmd, 2: bot2_cmd}
    names = {1: "Black (Bot 1)", 2: "White (Bot 2)"}
    
    print(f"Starting match: {names[1]} vs {names[2]}")
    
    turn = 0
    while True:
        turn += 1
        reversi.board_global = board # Update global for is_on_board
        
        # Check for game over or pass
        moves = reversi.get_legal_moves(board, current_player, 3-current_player)
        if not moves:
            # Check if opponent has moves
            opp_moves = reversi.get_legal_moves(board, 3-current_player, current_player)
            if not opp_moves:
                break # Game Over
            
            print(f"Turn {turn}: {names[current_player]} has no moves. Passing.")
            current_player = 3 - current_player
            continue

        # Prepare input for bot
        # Bots always play as Player 1. If it's Player 2's turn, we swap the board.
        if current_player == 2:
            bot_board = swap_board(board)
        else:
            bot_board = board
            
        board_str = board_to_string(bot_board)
        cmd = bots[current_player]
        
        # print(f"Turn {turn}: {names[current_player]} thinking...")
        start_t = time.time()
        
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate(input=board_str)
            duration = time.time() - start_t
            
            if process.returncode != 0:
                print(f"Error running bot {names[current_player]}: {stderr}")
                break
                
            # Parse output "r c"
            output_lines = stdout.strip().split('\n')
            if not output_lines:
                print(f"No output from bot {names[current_player]}")
                break
                
            line = output_lines[-1] # Take last line
            parts = line.split()
            if len(parts) >= 2:
                try:
                    r, c = int(parts[0]), int(parts[1])
                except ValueError:
                    print(f"Invalid output format from bot: {line}")
                    break
            else:
                print(f"Invalid output from bot: {line}")
                break
            
            if r == 0 and c == 0:
                if moves:
                    print(f"Bot returned 0 0 (pass) but moves exist! Illegal move.")
                    break
                else:
                    pass
            else:
                # Adjust to 0-based
                r_idx, c_idx = r - 1, c - 1
                
                # Validate move
                valid = False
                chosen_move = None
                for m in moves:
                    if m[0] == r_idx and m[1] == c_idx:
                        valid = True
                        chosen_move = m
                        break
                
                if not valid:
                    print(f"Illegal move by {names[current_player]}: {r} {c}")
                    break
                
                engine.apply_move(board, chosen_move, current_player)
                print(f"Turn {turn}: {names[current_player]} played {r} {c} ({duration:.2f}s)")
                
        except Exception as e:
            print(f"Exception running bot: {e}")
            break
            
        current_player = 3 - current_player

    # Game Over
    counts = engine.count_disks(board)
    print("\nGame Over!")
    print(f"Black: {counts[1]}")
    print(f"White: {counts[2]}")
    if counts[1] > counts[2]:
        print("Winner: Black")
    elif counts[2] > counts[1]:
        print("Winner: White")
    else:
        print("Draw")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 controller.py <bot1_cmd> <bot2_cmd> [start_board_file]")
        sys.exit(1)
    
    bot1 = sys.argv[1]
    bot2 = sys.argv[2]
    board_file = sys.argv[3] if len(sys.argv) > 3 else "start_board.txt"
    
    play_match(bot1, bot2, board_file)
