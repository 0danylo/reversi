#!/usr/bin/env python3
import argparse
from play import parse_board_lines
import engine
import strategies
import multiprocessing
from copy import deepcopy
import os


def _game_worker(args):
    # Worker runs a single game; args is tuple:
    # (board, black_name, white_name, depth, max_time, verbose, seed)
    board, black_name, white_name, depth, max_time, verbose, seed = args
    # make deterministic-ish by seeding random in the worker
    try:
        import random as _random
        _random.seed(seed + (os.getpid() if hasattr(os, 'getpid') else 0))
    except Exception:
        pass

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
    raise SystemExit(f"Unknown strategy: {name}")


def compare_strategy(board, target_name, opponents, games_per_opponent=10, depth=None, verbose=False):
    # parallelize per-opponent games using multiprocessing
    results = {}

    # use top-level worker to ensure picklability with multiprocessing

    cpu_count = max(1, multiprocessing.cpu_count() - 1)

    for opp_name in opponents:
        if opp_name == target_name:
            continue
        results[opp_name] = {"target_wins": 0, "opp_wins": 0, "ties": 0}

        # build jobs list for this opponent
        jobs = []
        for i in range(games_per_opponent):
            if i % 2 == 0:
                black_name = target_name
                white_name = opp_name
            else:
                black_name = opp_name
                white_name = target_name
            # include max_time in job args (None unless user provided)
            jobs.append((board, black_name, white_name, depth, compare_strategy.max_time if hasattr(compare_strategy, 'max_time') else None, verbose, i))

        # run games in parallel using non-daemonic worker processes so that
        # strategies that spawn child processes (for strict timeouts) work.
        def process_worker(in_q, out_q):
            while True:
                job = in_q.get()
                if job is None:
                    break
                try:
                    res = _game_worker(job)
                except Exception as e:
                    res = ('error', str(e))
                out_q.put(res)

        in_q = multiprocessing.Queue()
        out_q = multiprocessing.Queue()
        workers = []
        for _ in range(cpu_count):
            p = multiprocessing.Process(target=process_worker, args=(in_q, out_q))
            p.start()
            workers.append(p)

        for job in jobs:
            in_q.put(job)
        for _ in workers:
            in_q.put(None)

        winners = []
        for _ in range(len(jobs)):
            res = out_q.get()
            if isinstance(res, tuple) and res and res[0] == 'error':
                raise RuntimeError(f"Worker error: {res[1]}")
            winners.append(res)

        for p in workers:
            p.join()

        # aggregate winners
        for i, winner in enumerate(winners):
            if winner == 0:
                results[opp_name]["ties"] += 1
            else:
                target_is_black = (i % 2 == 0)
                if (winner == 1 and target_is_black) or (winner == 2 and not target_is_black):
                    results[opp_name]["target_wins"] += 1
                else:
                    results[opp_name]["opp_wins"] += 1

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare one strategy against all others")
    parser.add_argument("input", help="input file with 8 board lines (diamond layout)")
    parser.add_argument("--strategy", default="greedy", help="strategy to test: random, greedy, corner, ab")
    parser.add_argument("--games", type=int, default=10, help="games per opponent (will alternate colors)")
    parser.add_argument("--depth", type=int, default=3, help="alpha-beta search depth for AB strategy")
    parser.add_argument("--time", type=float, default=None, help="per-move time budget (seconds) for AB strategies")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        lines = [l.rstrip("\n") for l in f.readlines()]

    board = parse_board_lines(lines)

    # attach max_time to compare_strategy so worker jobs can see it
    compare_strategy.max_time = args.time

    all_ops = ["random", "greedy", "corner", "ab"]

    results = compare_strategy(board, args.strategy, all_ops, games_per_opponent=args.games, depth=args.depth, verbose=args.verbose)

    print(f"Results for strategy '{args.strategy}' vs others (games per opponent={args.games}):")
    for opp, res in results.items():
        total = res["target_wins"] + res["opp_wins"] + res["ties"]
        win_rate = res["target_wins"] / total if total else 0
        print(f"  vs {opp}: target_wins={res['target_wins']} opp_wins={res['opp_wins']} ties={res['ties']} win_rate={win_rate:.2%}")


if __name__ == '__main__':
    main()
