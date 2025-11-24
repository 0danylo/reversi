#!/usr/bin/env python3
import time
from play import parse_board_lines
import strategies, engine
from copy import deepcopy
import multiprocessing
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
    """Worker run in child process.

    job: (board, time_limit, seed, game_index)
    returns: (time_limit, winner, ab2_time, ab2_moves, ab_time, ab_moves, ab2_is_black)
    """
    board, time_limit, seed, game_index = job
    try:
        import random as _random
        _random.seed(seed + (os.getpid() if hasattr(os, 'getpid') else 0))
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


def run_benchmark(board_file='start_board.txt', games_per_setting=10, jobs=4):
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
    print(f"Running benchmark with {jobs} workers, {total} games total...")

    cpu_count = max(1, min(jobs, multiprocessing.cpu_count()))

    # Use a custom worker pool that uses non-daemonic Processes so that
    # per-game workers may spawn child Processes (used by the strict
    # per-move enforcement in strategies).
    def process_worker(in_q, out_q):
        while True:
            job = in_q.get()
            if job is None:
                break
            try:
                res = _bench_game_worker(job)
            except Exception as e:
                # propagate exception info
                res = ('error', str(e))
            out_q.put(res)

    in_q = multiprocessing.Queue()
    out_q = multiprocessing.Queue()

    workers = []
    for _ in range(cpu_count):
        p = multiprocessing.Process(target=process_worker, args=(in_q, out_q))
        p.start()
        workers.append(p)

    # enqueue jobs
    for job in jobs_list:
        in_q.put(job)
    # send sentinel to stop workers
    for _ in workers:
        in_q.put(None)

    # collect results
    while processed < total:
        res = out_q.get()
        processed += 1
        if isinstance(res, tuple) and res and res[0] == 'error':
            print(f"Worker error: {res[1]}")
            continue
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

    # join workers
    for p in workers:
        p.join()

    # print summary
    print('\nBenchmark results (AB2 iterative deepening vs AB fixed-depth)')
    for t in time_limits:
        s = stats_map[t]
        avg_ab2 = s['ab2_time'] / s['ab2_moves'] if s['ab2_moves'] else 0
        avg_ab = s['ab_time'] / s['ab_moves'] if s['ab_moves'] else 0
        print(f"time_limit={t:>4.2f}s: ab2_wins={s['ab2_wins']:2d} ab_wins={s['ab_wins']:2d} ties={s['ties']:2d} avg_ab2_move={avg_ab2:.3f}s avg_ab_move={avg_ab:.3f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark AB variants in parallel')
    parser.add_argument('--board', default='start_board.txt')
    parser.add_argument('--games', type=int, default=10, help='games per time setting')
    parser.add_argument('--jobs', type=int, default=4, help='parallel workers')
    args = parser.parse_args()
    run_benchmark(board_file=args.board, games_per_setting=args.games, jobs=args.jobs)
#!/usr/bin/env python3
import time
from play import parse_board_lines
import strategies, engine
from copy import deepcopy


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


def run_benchmark(board_file='start_board.txt', games_per_setting=10):
    with open(board_file) as f:
        lines = [l.rstrip('\n') for l in f.readlines()]
    board = parse_board_lines(lines)

    time_limits = [0.25, 0.5, 1.0, 2.0, 3.0]
    results = []

    for t in time_limits:
        stats = {'time_limit': t, 'ab2_wins': 0, 'ab_wins': 0, 'ties': 0, 'ab2_time': 0.0, 'ab_time': 0.0, 'ab2_moves': 0, 'ab_moves': 0}

        for g in range(games_per_setting):
            # alternate colors: even g -> ab2 black, odd -> ab2 white
            if g % 2 == 0:
                black = TimerStrategy(strategies.AlphaBetaImprovedStrategy(depth=8, max_time=t))
                white = TimerStrategy(strategies.AlphaBetaStrategy(depth=4, max_time=None))
                ab2_is_black = True
            else:
                black = TimerStrategy(strategies.AlphaBetaStrategy(depth=4, max_time=None))
                white = TimerStrategy(strategies.AlphaBetaImprovedStrategy(depth=8, max_time=t))
                ab2_is_black = False

            final_board, counts, winner = engine.play_game(deepcopy(board), black, white, verbose=False)

            if winner == 0:
                stats['ties'] += 1
            else:
                if (winner == 1 and ab2_is_black) or (winner == 2 and not ab2_is_black):
                    stats['ab2_wins'] += 1
                else:
                    stats['ab_wins'] += 1

            # accumulate timing
            if ab2_is_black:
                stats['ab2_time'] += black.total_time
                stats['ab2_moves'] += black.move_count
                stats['ab_time'] += white.total_time
                stats['ab_moves'] += white.move_count
            else:
                stats['ab_time'] += black.total_time
                stats['ab_moves'] += black.move_count
                stats['ab2_time'] += white.total_time
                stats['ab2_moves'] += white.move_count

        results.append(stats)

    # print summary
    print('Benchmark results (AB2 iterative deepening vs AB fixed-depth)')
    for s in results:
        avg_ab2 = s['ab2_time'] / s['ab2_moves'] if s['ab2_moves'] else 0
        avg_ab = s['ab_time'] / s['ab_moves'] if s['ab_moves'] else 0
        print(f"time_limit={s['time_limit']:>4.2f}s: ab2_wins={s['ab2_wins']:2d} ab_wins={s['ab_wins']:2d} ties={s['ties']:2d} avg_ab2_move={avg_ab2:.3f}s avg_ab_move={avg_ab:.3f}s")


if __name__ == '__main__':
    run_benchmark()
