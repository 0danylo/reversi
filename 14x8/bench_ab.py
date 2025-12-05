#!/usr/bin/env python3
import time
from play import parse_board_lines
import strategies, engine
from copy import deepcopy
import argparse
import os


class TimerStrategy:
    """Wrap a strategy to measure total choose_move time and counts."""
    def __init__(self, strat):
        self.strat = strat
        self.total_time = 0.0
        self.move_count = 0

    def choose_move(self, board, me=1, opp=2):
        t0 = time.time()
        mv = self.strat.choose_move(board, me, opp)
        dt = time.time() - t0
        if mv is not None:
            self.total_time += dt
            self.move_count += 1
        return mv


def _bench_game_worker(job):
    """Worker run in single process.

    job: (board, time_limit, seed, game_index)
    returns: (time_limit, winner, ab2_time, ab2_moves, ab_time, ab_moves, ab2_is_black)
    """
    board, time_limit, seed, game_index = job
    try:
        import random as _random
        _random.seed(seed)
    except Exception:
        pass

    if game_index % 2 == 0:
        black = TimerStrategy(strategies.AlphaBetaImprovedStrategy(depth=8, max_time=time_limit))
        white = TimerStrategy(strategies.AlphaBetaStrategy(depth=4, max_time=None))
        ab2_is_black = True
    else:
        black = TimerStrategy(strategies.AlphaBetaStrategy(depth=4, max_time=None))
        white = TimerStrategy(strategies.AlphaBetaImprovedStrategy(depth=8, max_time=time_limit))
        ab2_is_black = False

    final_board, counts, winner = engine.play_game(deepcopy(board), black, white, verbose=False)

    if ab2_is_black:
        return (time_limit, winner, black.total_time, black.move_count, white.total_time, white.move_count, True)
    else:
        return (time_limit, winner, white.total_time, white.move_count, black.total_time, black.move_count, False)


def run_benchmark(board_file='start_board.txt', games_per_setting=10):
    with open(board_file) as f:
        lines = [l.rstrip('\n') for l in f.readlines()]
    board = parse_board_lines(lines)

    time_limits = [0.25, 0.5, 1.0, 2.0, 3.0]
    stats_map = {t: {'time_limit': t, 'ab2_wins': 0, 'ab_wins': 0, 'ties': 0, 'ab2_time': 0.0, 'ab_time': 0.0, 'ab2_moves': 0, 'ab_moves': 0} for t in time_limits}

    # build job list
    jobs_list = []
    for t in time_limits:
        for g in range(games_per_setting):
            # use g as seed and game_index
            jobs_list.append((board, t, g, g))

    total = len(jobs_list)
    processed = 0
    print(f"Running benchmark with 1 worker, {total} games total...")

    # run sequentially
    for job in jobs_list:
        try:
            res = _bench_game_worker(job)
        except Exception as e:
            print(f"Worker error: {e}")
            continue

        processed += 1
        t, winner, ab2_time, ab2_moves, ab_time, ab_moves, ab2_is_black = res
        s = stats_map[t]

        if winner == 0:
            s['ties'] += 1
        else:
            if ab2_is_black:
                if winner == 1:
                    s['ab2_wins'] += 1
                else:
                    s['ab_wins'] += 1
            else:
                if winner == 2:
                    s['ab2_wins'] += 1
                else:
                    s['ab_wins'] += 1

        # aggregate timings
        s['ab2_time'] += ab2_time
        s['ab2_moves'] += ab2_moves
        s['ab_time'] += ab_time
        s['ab_moves'] += ab_moves

        # progress report
        if processed % max(1, total // 20) == 0 or processed == total:
            print(f"Progress: {processed}/{total} games processed")

    # print summary
    print('\nBenchmark results (AB2 iterative deepening vs AB fixed-depth)')
    for t in time_limits:
        s = stats_map[t]
        avg_ab2 = s['ab2_time'] / s['ab2_moves'] if s['ab2_moves'] else 0
        avg_ab = s['ab_time'] / s['ab_moves'] if s['ab_moves'] else 0
        print(f"time_limit={t:>4.2f}s: ab2_wins={s['ab2_wins']:2d} ab_wins={s['ab_wins']:2d} ties={s['ties']:2d} avg_ab2_move={avg_ab2:.3f}s avg_ab_move={avg_ab:.3f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark AB variants sequentially')
    parser.add_argument('--board', default='start_board.txt')
    parser.add_argument('--games', type=int, default=10, help='games per time setting')
    args = parser.parse_args()
    run_benchmark(board_file=args.board, games_per_setting=args.games)
