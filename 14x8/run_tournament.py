#!/usr/bin/env python3
import subprocess
import sys
import re

strategies = ['ab2', 'ab_tt', 'ab_bit', 'ab_bit_tt']
results = {s: 0 for s in strategies}
games_per_pair = 2 # One each color

def run_match(black_strat, white_strat):
    cmd_black = f"python3 reversi_bot.py --strategy {black_strat}"
    cmd_white = f"python3 reversi_bot.py --strategy {white_strat}"
    
    print(f"Match: {black_strat} (Black) vs {white_strat} (White)")
    
    # Run controller
    cmd = ["python3", "controller.py", cmd_black, cmd_white, "start_board.txt"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout
        # Parse winner
        if "Winner: Black" in output:
            return 1 # Black wins
        elif "Winner: White" in output:
            return 2 # White wins
        elif "Draw" in output:
            return 0 # Draw
        else:
            print("Error: Could not determine winner")
            print(output)
            return -1
    except subprocess.TimeoutExpired:
        print("Match timed out")
        return -1
    except Exception as e:
        print(f"Exception: {e}")
        return -1

print(f"Running tournament between: {strategies}")

for i in range(len(strategies)):
    for j in range(i + 1, len(strategies)):
        s1 = strategies[i]
        s2 = strategies[j]
        
        # Game 1: s1 is Black
        winner = run_match(s1, s2)
        if winner == 1:
            results[s1] += 1
            print(f"Winner: {s1}")
        elif winner == 2:
            results[s2] += 1
            print(f"Winner: {s2}")
        elif winner == 0:
            results[s1] += 0.5
            results[s2] += 0.5
            print("Draw")
            
        # Game 2: s2 is Black
        winner = run_match(s2, s1)
        if winner == 1:
            results[s2] += 1
            print(f"Winner: {s2}")
        elif winner == 2:
            results[s1] += 1
            print(f"Winner: {s1}")
        elif winner == 0:
            results[s1] += 0.5
            results[s2] += 0.5
            print("Draw")

print("\nTournament Results:")
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for s, score in sorted_results:
    print(f"{s}: {score}")
