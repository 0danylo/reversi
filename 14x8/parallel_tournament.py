#!/usr/bin/env python3
import subprocess
import concurrent.futures
import sys

target_strategy = 'ab_bit_tt'
opponents = ['ab2', 'ab_tt', 'ab_bit']
results = {opp: {'wins': 0, 'losses': 0, 'draws': 0} for opp in opponents}

def run_single_match(black_strat, white_strat, match_id):
    cmd_black = f"python3 reversi_bot.py --strategy {black_strat}"
    cmd_white = f"python3 reversi_bot.py --strategy {white_strat}"
    
    cmd = ["python3", "controller.py", cmd_black, cmd_white, "start_board.txt"]
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
    print(f"Running tournament: {target_strategy} VS {opponents}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = []
        
        for opp in opponents:
            # Game 1: Target is Black
            mid1 = f"{target_strategy}_vs_{opp}"
            futures.append(executor.submit(run_single_match, target_strategy, opp, mid1))
            
            # Game 2: Target is White
            mid2 = f"{opp}_vs_{target_strategy}"
            futures.append(executor.submit(run_single_match, opp, target_strategy, mid2))
            
        for future in concurrent.futures.as_completed(futures):
            mid, winner, details = future.result()
            
            # Parse who won relative to target
            # mid format: "black_strat_vs_white_strat"
            parts = mid.split('_vs_')
            # Handle potential underscores in strategy names carefully
            # We know target is ab_bit_tt
            if parts[0] == target_strategy:
                black = target_strategy
                white = mid[len(target_strategy)+4:] # remove "ab_bit_tt_vs_"
            else:
                white = target_strategy
                black = mid[:-len(target_strategy)-4] # remove "_vs_ab_bit_tt"
            
            opponent = white if black == target_strategy else black
            
            print(f"Match {mid} finished. Winner: {winner}")
            
            if winner == 'timeout' or winner == 'error':
                print(f"Match failed: {details}")
                continue
                
            if black == target_strategy:
                if winner == 'black':
                    results[opponent]['wins'] += 1
                elif winner == 'white':
                    results[opponent]['losses'] += 1
                else:
                    results[opponent]['draws'] += 1
            else: # target is white
                if winner == 'white':
                    results[opponent]['wins'] += 1
                elif winner == 'black':
                    results[opponent]['losses'] += 1
                else:
                    results[opponent]['draws'] += 1

    print("\nResults for ab_bit_tt:")
    for opp, res in results.items():
        print(f"VS {opp}: {res['wins']} Wins, {res['losses']} Losses, {res['draws']} Draws")

if __name__ == "__main__":
    main()
