#!/usr/bin/env python3
import subprocess
import concurrent.futures
import sys

bot1_cmd = "python3 g3mini.py"
bot2_cmd = "python3 revbot2.py"
bot1_name = "reversi_bot"
bot2_name = "revbot2"

results = {bot1_name: 0, bot2_name: 0, 'draws': 0}
games_per_side = 10  # Total 20 games

def run_single_match(black_cmd, white_cmd, match_id):
    cmd = ["python3", "controller.py", black_cmd, white_cmd, "start_board.txt"]
    try:
        # Run controller
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout
        
        winner = None
        if "Winner: Black" in output:
            winner = 'black'
        elif "Winner: White" in output:
            winner = 'white'
        elif "Draw" in output:
            winner = 'draw'
            
        return match_id, winner, output
    except subprocess.TimeoutExpired:
        return match_id, 'timeout', "Timeout"
    except Exception as e:
        return match_id, 'error', str(e)

def main():
    print(f"Running parallel tournament: {bot1_name} VS {bot2_name}")
    print(f"Total games: {games_per_side * 2}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        
        for i in range(games_per_side):
            # Game set 1: Bot 1 is Black
            mid1 = f"game_{i}_b1_black"
            futures.append(executor.submit(run_single_match, bot1_cmd, bot2_cmd, mid1))
            
            # Game set 2: Bot 2 is Black
            mid2 = f"game_{i}_b2_black"
            futures.append(executor.submit(run_single_match, bot2_cmd, bot1_cmd, mid2))
            
        for future in concurrent.futures.as_completed(futures):
            mid, winner, details = future.result()
            
            print(f"Match {mid} finished. Winner: {winner}")
            
            if winner == 'timeout' or winner == 'error':
                print(f"Match failed: {details}")
                continue
            
            # Determine who won based on match ID
            if "b1_black" in mid:
                # Bot 1 was Black, Bot 2 was White
                if winner == 'black':
                    results[bot1_name] += 1
                elif winner == 'white':
                    results[bot2_name] += 1
                else:
                    results['draws'] += 1
            else:
                # Bot 2 was Black, Bot 1 was White
                if winner == 'black':
                    results[bot2_name] += 1
                elif winner == 'white':
                    results[bot1_name] += 1
                else:
                    results['draws'] += 1

    print("\nTournament Results:")
    print(f"{bot1_name}: {results[bot1_name]} wins")
    print(f"{bot2_name}: {results[bot2_name]} wins")
    print(f"Draws: {results['draws']}")

if __name__ == "__main__":
    main()
