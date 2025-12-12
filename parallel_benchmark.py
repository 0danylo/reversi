import sys
import os
import glob
import multiprocessing
from triangle_server import play_game

def run_match(args):
    p1_cmd, p2_cmd = args
    try:
        # Run silent game
        s1, s2 = play_game(p1_cmd, p2_cmd, verbose=False, disable_progress=True)
        return (p1_cmd, p2_cmd, s1, s2)
    except Exception as e:
        return (p1_cmd, p2_cmd, -1, -1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 parallel_benchmark.py <hero_bot.py>")
        sys.exit(1)

    hero_file = sys.argv[1]
    if not os.path.exists(hero_file):
        print(f"Error: {hero_file} not found.")
        sys.exit(1)

    hero_cmd = f"python3 {hero_file}"
    
    # Find opponents: all triangle_bot*.py except hero
    files = glob.glob("triangle_bot*.py")
    opponents = []
    abs_hero = os.path.abspath(hero_file)
    
    for f in files:
        if os.path.abspath(f) != abs_hero:
            opponents.append(f) 
            
    if not opponents:
        print("No opponents found.")
        sys.exit(0)
        
    print(f"Hero: {hero_file}")
    print(f"Opponents: {opponents}")
    
    tasks = []
    for opp in opponents:
        opp_cmd = f"python3 {opp}"
        # Game 1: Hero vs Opponent
        tasks.append((hero_cmd, opp_cmd))
        # Game 2: Opponent vs Hero
        tasks.append((opp_cmd, hero_cmd))
        
    print(f"Scheduled {len(tasks)} games.")
    
    cpu_count = multiprocessing.cpu_count()
    print(f"Running on {cpu_count} cores...")
    
    results = []
    with multiprocessing.Pool(processes=cpu_count) as pool:
        # Use tqdm for overall progress
        try:
            from tqdm import tqdm
            results = list(tqdm(pool.imap(run_match, tasks), total=len(tasks), unit="game"))
        except ImportError:
            results = pool.map(run_match, tasks)

    # Aggregation
    wins = 0
    losses = 0
    draws = 0
    total_score_for = 0
    total_score_against = 0
    
    print("\nDetailed Results:")
    print(f"{'P1 (Black)':<35} | {'P2 (White)':<35} | {'Score':<10} | {'Winner'}")
    print("-" * 100)
    
    for p1, p2, s1, s2 in results:
        if s1 == -1:
            print(f"{p1:<35} | {p2:<35} | ERROR      | ERROR")
            continue
            
        winner = "Draw"
        if s1 > s2: winner = p1
        elif s2 > s1: winner = p2
        
        # Shorten names for display
        p1_disp = p1.replace("python3 ", "")
        p2_disp = p2.replace("python3 ", "")
        winner_disp = winner.replace("python3 ", "")
        
        print(f"{p1_disp:<35} | {p2_disp:<35} | {s1}-{s2:<5} | {winner_disp}")
        
        # Stats
        if p1 == hero_cmd:
            my_score = s1
            op_score = s2
        else:
            my_score = s2
            op_score = s1
            
        total_score_for += my_score
        total_score_against += op_score
        
        if my_score > op_score: wins += 1
        elif op_score > my_score: losses += 1
        else: draws += 1

    print("\n" + "="*40)
    print(f"Summary for {hero_file}")
    print("="*40)
    print(f"Opponents faced: {len(opponents)}")
    print(f"Total Games: {len(tasks)}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Draws: {draws}")
    if len(tasks) > 0:
        print(f"Win Rate: {wins/len(tasks)*100:.1f}%")
        print(f"Total Score: {total_score_for} - {total_score_against}")
        print(f"Average Score: {total_score_for/len(tasks):.1f} - {total_score_against/len(tasks):.1f}")
    else:
        print("No games played.")

if __name__ == "__main__":
    main()
