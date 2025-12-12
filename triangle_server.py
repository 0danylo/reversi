import sys
import subprocess
import time
from tqdm import tqdm

# Board configuration
ROWS = 10
OFFSETS = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
LENGTHS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

def get_initial_board():
    # 0: empty, 1: p1, 2: p2
    board = []
    for r in range(ROWS):
        board.append([0] * LENGTHS[r])
    
    # Standard starting position for Reversi is usually center 4.
    # Center of the board is around row 4-5.
    # Row 4 (len 10, offset 5): indices 5..14. Center 9, 10.
    # Row 5 (len 12, offset 4): indices 4..15. Center 9, 10.
    
    # Colors:
    # (4, 9): 2, (4, 10): 1
    # (5, 9): 1, (5, 10): 2
    
    # Local indices:
    # Row 4: global 9 -> local 9-5 = 4. global 10 -> local 10-5 = 5.
    # Row 5: global 9 -> local 9-4 = 5. global 10 -> local 10-4 = 6.
    
    board[4][4] = 2
    board[4][5] = 1
    board[5][5] = 1
    board[5][6] = 2
    
    return board

def print_board(board):
    for r in range(ROWS):
        # Print with offset for visualization
        print("  " * OFFSETS[r] + " ".join(f"{x}" for x in board[r]))

def get_board_string(board, player):
    # Convert board to string for player.
    # If player is 2, swap 1 and 2.
    lines = []
    for r in range(ROWS):
        row_vals = []
        for val in board[r]:
            if val == 0:
                row_vals.append('0')
            elif val == player:
                row_vals.append('1')
            else:
                row_vals.append('2')
        lines.append(" ".join(row_vals))
    return "\n".join(lines)

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
    # Check if move (r, c) is valid and return flipped pieces.
    # (r, c) are global coordinates.
    
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

def has_valid_move(board, player):
    for r in range(ROWS):
        offset = OFFSETS[r]
        length = LENGTHS[r]
        for local_c in range(length):
            c = offset + local_c
            if board[r][local_c] == 0:
                if get_flips(board, player, r, c):
                    return True
    return False

def apply_move(board, player, r, c, flips):
    local_c = c - OFFSETS[r]
    board[r][local_c] = player
    for fr, fc in flips:
        board[fr][fc - OFFSETS[fr]] = player

def run_player(command, board_str):
    try:
        start_time = time.time()
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True
        )
        stdout, stderr = process.communicate(input=board_str, timeout=3)
        duration = time.time() - start_time
        if process.returncode != 0:
            # print(f"Error running {command}: {stderr}")
            return None
        return stdout.strip()
    except subprocess.TimeoutExpired:
        if 'process' in locals():
            process.kill()
        print(f"Timeout running {command}")
        return None
    except Exception as e:
        print(f"Exception running {command}: {e}")
        return None

def play_game(p1_cmd, p2_cmd, verbose=False, disable_progress=False):
    board = get_initial_board()
    current_player = 1
    skipped_last = False
    
    # Disable progress bar if requested OR if verbose logging is on (to prevent output mixing)
    pbar = tqdm(total=106, disable=(disable_progress or verbose), desc="Game Progress", unit="move")
    
    while True:
        # Check if current player has moves
        if not has_valid_move(board, current_player):
            if verbose: print(f"Player {current_player} has no moves.")
            if skipped_last:
                if verbose: print("Both players skipped. Game over.")
                break
            skipped_last = True
            current_player = 3 - current_player
            continue
        
        skipped_last = False
        
        cmd = p1_cmd if current_player == 1 else p2_cmd
        board_str = get_board_string(board, current_player)
        
        output = run_player(cmd, board_str)
        
        if output is None:
            if verbose: print(f"Player {current_player} crashed or timed out. Player {3-current_player} wins.")
            # Assign remaining squares to winner? Or just end.
            # For simplicity, just return current score but penalize crasher?
            # Let's just return current board state.
            break
            
        try:
            parts = output.split()
            if len(parts) < 2:
                raise ValueError("Not enough parts")
            # Bot outputs 1-indexed row and 1-indexed local column
            r_1idx, local_c_1idx = int(parts[0]), int(parts[1])
            r = r_1idx - 1  # Convert to 0-indexed row
            c = OFFSETS[r] + (local_c_1idx - 1)  # Convert local column to global
        except (ValueError, IndexError) as e:
            if verbose: print(f"Invalid output from Player {current_player}: {output} ({e})")
            break
            
        flips = get_flips(board, current_player, r, c)
        if not flips:
            if verbose: print(f"Invalid move from Player {current_player}: {r} {c}")
            break
            
        apply_move(board, current_player, r, c, flips)
        if verbose: print(f"Player {current_player} played {r} {c}")
        pbar.update(1)
        
        current_player = 3 - current_player

    pbar.close()
    # Game over, count score
    p1_score = sum(row.count(1) for row in board)
    p2_score = sum(row.count(2) for row in board)
    
    # [print(f'{OFFSETS[i]}{b}') for i, b in enumerate(board)]
    return p1_score, p2_score

def main():
    if len(sys.argv) < 3:
        print("Usage: python triangle_server.py <p1_cmd> <p2_cmd>")
        sys.exit(1)
        
    cmd1 = sys.argv[1]
    cmd2 = sys.argv[2]
    
    print(f"Game 1: P1={cmd1} vs P2={cmd2}")
    s1, s2 = play_game(cmd1, cmd2, verbose=False)
    print(f"Result: {s1} - {s2}")
    if s1 > s2: print(f"Winner: {cmd1}")
    elif s2 > s1: print(f"Winner: {cmd2}")
    else: print("Draw")
    
    print("-" * 20)
    
    print(f"Game 2: P1={cmd2} vs P2={cmd1}")
    s1_swap, s2_swap = play_game(cmd2, cmd1, verbose=False)
    print(f"Result: {s1_swap} - {s2_swap}")
    if s1_swap > s2_swap: print(f"Winner: {cmd2}")
    elif s2_swap > s1_swap: print(f"Winner: {cmd1}")
    else: print("Draw")
    
    print("=" * 20)
    total_cmd1 = s1 + s2_swap
    total_cmd2 = s2 + s1_swap
    print(f"Total Score: {cmd1}={total_cmd1}, {cmd2}={total_cmd2}")
    if total_cmd1 > total_cmd2:
        print(f"Overall Winner: {cmd1}")
    elif total_cmd2 > total_cmd1:
        print(f"Overall Winner: {cmd2}")
    else:
        print("Overall Draw")

if __name__ == "__main__":
    main()
