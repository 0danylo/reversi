#!/usr/bin/env python3
import subprocess
import sys

def run_match(black_cmd, white_cmd, match_name):
    print(f"Starting Match: {match_name}")
    print(f"Black: {black_cmd}")
    print(f"White: {white_cmd}")
    
    cmd = ["python3", "controller.py", black_cmd, white_cmd, "start_board.txt"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout
        
        if "Winner: Black" in output:
            print(f"Winner: Black ({black_cmd})")
            return 1
        elif "Winner: White" in output:
            print(f"Winner: White ({white_cmd})")
            return 2
        elif "Draw" in output:
            print("Result: Draw")
            return 0
        else:
            print("Error: Could not determine winner")
            # print(output) # Print output for debugging if needed
            return -1
    except subprocess.TimeoutExpired:
        print("Match timed out")
        return -1
    except Exception as e:
        print(f"Exception: {e}")
        return -1

def main():
    bot1 = "python3 reversi_bot.py"
    bot2 = "python3 revbot2.py"
    
    wins_bot1 = 0
    wins_bot2 = 0
    draws = 0
    
    # Game 1: Bot 1 is Black
    res = run_match(bot1, bot2, "Game 1")
    if res == 1: wins_bot1 += 1
    elif res == 2: wins_bot2 += 1
    elif res == 0: draws += 1
    
    print("-" * 20)
    
    # Game 2: Bot 2 is Black
    res = run_match(bot2, bot1, "Game 2")
    if res == 1: wins_bot2 += 1
    elif res == 2: wins_bot1 += 1
    elif res == 0: draws += 1
    
    print("=" * 20)
    print("Tournament Results:")
    print(f"reversi_bot.py: {wins_bot1} wins")
    print(f"revbot2.py:     {wins_bot2} wins")
    print(f"Draws:          {draws}")

if __name__ == "__main__":
    main()
