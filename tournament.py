#!/usr/bin/env python3
import concurrent.futures
import argparse
import time
from play import parse_board_lines
import engine
import strategies
from copy import deepcopy
import os
import reversi


def _game_worker(args):
    # Worker runs a single game; args is tuple:
    # (board, black_name, white_name, depth, max_time, verbose, seed, random_moves)
    board, black_name, white_name, depth, max_time, verbose, seed, random_moves = args
    # make deterministic-ish by seeding random in the worker
    try:
        import random as _random
        _random.seed(seed)
    except Exception:
        pass

    # Apply random moves if requested
    if random_moves > 0:
        curr_board = deepcopy(board)
        curr = 1
        for _ in range(random_moves):
            reversi.board_global = curr_board
            moves = reversi.get_legal_moves(curr_board, curr, 3-curr)
            if moves:
                m = _random.choice(moves)
                engine.apply_move(curr_board, m, curr)
            else:
                # If no moves, try other player? Or just stop?
                # If pass, we should swap current but not apply move.
                # Check if other has moves
                reversi.board_global = curr_board
                opp_moves = reversi.get_legal_moves(curr_board, 3-curr, curr)
                if not opp_moves:
                    break # Game over
            curr = 3 - curr
        board = curr_board

    black = make_strategy(black_name, depth, max_time)
    white = make_strategy(white_name, depth, max_time)
    final_board, counts, winner = engine.play_game(deepcopy(board), black, white, verbose=verbose)
    return winner


def make_strategy(name, depth=None, max_time=None):
    n = name.lower()
    if n == "random":
        return strategies.RandomStrategy()
    if n == "greedy":
        return strategies.GreedyStrategy()
    if n in ("corner", "corner_first", "cornerfirst"):
        return strategies.CornerFirstStrategy()
    if n in ("ab", "alphabeta"):
        return strategies.AlphaBetaStrategy(depth=depth if depth is not None else 3, max_time=max_time)
    if n in ("ab2", "ab_improved", "ab+"):
        return strategies.AlphaBetaImprovedStrategy(depth=depth if depth is not None else 3, max_time=max_time)
    if n in ("ab_opt", "ab3"):
        return strategies.AlphaBetaOptimizedStrategy(depth=depth if depth is not None else 3, max_time=max_time)
    if n in ("ab_tt", "ab_transposition"):
        return strategies.AlphaBetaTTStrategy(depth=depth if depth is not None else 3, max_time=max_time)
    if n in ("ab_bit", "ab_bitboard"):
        return strategies.AlphaBetaBitboardStrategy(depth=depth if depth is not None else 3, max_time=max_time)
    if n in ("ab_bit_tt", "ab_bitboard_tt"):
        return strategies.AlphaBetaBitboardTTStrategy(depth=depth if depth is not None else 3, max_time=max_time)
    raise SystemExit(f"Unknown strategy: {name}")


def compare_strategy(board, target_name, opponents, games_per_opponent=10, depth=None, verbose=False, random_moves=0):
    # run games in parallel
    results = {}
    
    # Prepare all jobs
    all_jobs = []
    job_map = {} # index -> (opp_name, is_target_black)

    for opp_name in opponents:
        if opp_name == target_name:
            continue
        results[opp_name] = {"target_wins": 0, "opp_wins": 0, "ties": 0}

        for i in range(games_per_opponent):
            if i % 2 == 0:
                black_name = target_name
                white_name = opp_name
                is_target_black = True
            else:
                black_name = opp_name
                white_name = target_name
                is_target_black = False
            
            job = (board, black_name, white_name, depth, compare_strategy.max_time if hasattr(compare_strategy, 'max_time') else None, verbose, i, random_moves)
            all_jobs.append(job)
            job_map[len(all_jobs)-1] = (opp_name, is_target_black)

    print(f"Running {len(all_jobs)} games in parallel...")
    
    start_time = time.time()
    completed_count = 0
    total_jobs = len(all_jobs)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all jobs
        future_to_job_idx = {executor.submit(_game_worker, job): i for i, job in enumerate(all_jobs)}
        
        for future in concurrent.futures.as_completed(future_to_job_idx):
            job_idx = future_to_job_idx[future]
            opp_name, is_target_black = job_map[job_idx]
            
            try:
                winner = future.result()
                completed_count += 1
                elapsed = time.time() - start_time
                print(f"[{completed_count}/{total_jobs}] Match finished ({elapsed:.1f}s). Winner: {winner}")

                if winner == 0:
                    results[opp_name]["ties"] += 1
                else:
                    if (winner == 1 and is_target_black) or (winner == 2 and not is_target_black):
                        results[opp_name]["target_wins"] += 1
                    else:
                        results[opp_name]["opp_wins"] += 1
            except Exception as exc:
                print(f'Game {job_idx} generated an exception: {exc}')

    total_time = time.time() - start_time
    print(f"Tournament completed in {total_time:.2f} seconds")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare one strategy against all others")
    parser.add_argument("input", help="input file with 8 board lines (diamond layout)")
    parser.add_argument("--rect", action="store_true", help="interpret input as regular rectangular 8x8 board")
    parser.add_argument("--strategy", default="greedy", help="strategy to test: random, greedy, corner, ab, ab2, ab_tt, ab_bit, ab_bit_tt")
    parser.add_argument("--opponent", help="specific opponent to test against (overrides default list)")
    parser.add_argument("--games", type=int, default=10, help="games per opponent (will alternate colors)")
    parser.add_argument("--random-moves", type=int, default=0, help="number of random moves to play at start")
    parser.add_argument("--depth", type=int, default=3, help="alpha-beta search depth for AB strategy")
    parser.add_argument("--time", type=float, default=None, help="per-move time budget (seconds) for AB strategies")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        lines = [l.rstrip("\n") for l in f.readlines()]

    if args.rect:
        # use rectangular 8x8 parser
        from play import parse_board_lines_rect
        board = parse_board_lines_rect(lines)
    else:
        board = parse_board_lines(lines)

    # attach max_time to compare_strategy so worker jobs can see it
    compare_strategy.max_time = args.time

    if args.opponent:
        all_ops = [args.opponent]
    else:
        all_ops = ["random", "greedy", "corner", "ab"]

    results = compare_strategy(board, args.strategy, all_ops, games_per_opponent=args.games, depth=args.depth, verbose=args.verbose, random_moves=args.random_moves)

    print(f"Results for strategy '{args.strategy}' vs others (games per opponent={args.games}):")
    for opp, res in results.items():
        total = res["target_wins"] + res["opp_wins"] + res["ties"]
        win_rate = res["target_wins"] / total if total else 0
        print(f"  vs {opp}: target_wins={res['target_wins']} opp_wins={res['opp_wins']} ties={res['ties']} win_rate={win_rate:.2%}")


if __name__ == '__main__':
    main()
