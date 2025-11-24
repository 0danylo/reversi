#!/usr/bin/env python3
import argparse
import random
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, *args, **kwargs): pass
        def update(self, n=1): pass
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

from strategies import RandomStrategy, GreedyStrategy, CornerFirstStrategy, AlphaBetaStrategy, AlphaBetaImprovedStrategy, AlphaBetaOptimizedStrategy, AlphaBetaTTStrategy, AlphaBetaBitboardStrategy, AlphaBetaBitboardTTStrategy
import engine
from engine import play_game

def parse_board_lines(lines):
    # same logic as reversi.read_board but from provided lines
    row_lengths = [8, 10, 12, 14, 14, 12, 10, 8]
    board = []
    for i, ln in enumerate(row_lengths):
        if i < len(lines):
            parts = lines[i].strip().split()
        else:
            parts = []

        vals = []
        for j, tok in enumerate(parts[:ln]):
            try:
                vals.append(int(tok))
            except ValueError:
                vals.append(0)

        if len(vals) < ln:
            vals += [0] * (ln - len(vals))

        board.append(vals)

    return board


def parse_board_lines_rect(lines):
    """Parse a regular rectangular 8x8 board from lines.

    Each of the first 8 lines should contain 8 integers (0,1,2) separated by
    spaces. Missing values are treated as 0. Returns a list of 8 lists.
    """
    board = []
    for i in range(8):
        if i < len(lines):
            parts = lines[i].strip().split()
        else:
            parts = []
        vals = []
        for j, tok in enumerate(parts[:8]):
            try:
                vals.append(int(tok))
            except Exception:
                vals.append(0)
        if len(vals) < 8:
            vals += [0] * (8 - len(vals))
        board.append(vals)
    return board


def strategy_from_name(name):
    name = name.lower()
    if name == 'ab':
        return AlphaBetaStrategy()
    if name in ('ab2', 'ab_improved', 'ab+'): 
        return AlphaBetaImprovedStrategy()
    if name in ('ab_opt', 'ab3'):
        return AlphaBetaOptimizedStrategy()
    if name in ('ab_tt', 'ab_transposition'):
        return AlphaBetaTTStrategy()
    if name in ('ab_bit', 'ab_bitboard'):
        return AlphaBetaBitboardStrategy()
    if name in ('ab_bit_tt', 'ab_bitboard_tt'):
        return AlphaBetaBitboardTTStrategy()
    if name == "random":
        return RandomStrategy()
    if name == "greedy":
        return GreedyStrategy()
    if name in ("corner", "corner_first", "cornerfirst"):
        return CornerFirstStrategy()
    raise SystemExit(f"Unknown strategy: {name}")


def print_board(board):
    max_len = 14
    print("  " + "-" * (max_len * 2))
    for row in board:
        # Map values to symbols
        symbols = []
        for cell in row:
            if cell == 0: symbols.append('.')
            elif cell == 1: symbols.append('B')
            elif cell == 2: symbols.append('W')
            else: symbols.append('?')
        
        line = " ".join(symbols)
        padding = (max_len - len(row)) 
        print(" " * padding + line)
    print("  " + "-" * (max_len * 2))


def main():
    parser = argparse.ArgumentParser(description="Play a single Reversi match between two strategies")
    # parser.add_argument("input", help="input file with 8 board lines (diamond layout)")
    parser.add_argument("--black", default="greedy", help="strategy for black (player 1). options: random, greedy, corner")
    parser.add_argument("--white", default="random", help="strategy for white (player 2).")
    parser.add_argument("--depth", type=int, default=3, help="global alpha-beta search depth (used when per-player depth not provided)")
    parser.add_argument("--black-depth", "-B", type=int, default=None, help="alpha-beta depth for black when using 'ab' (overrides --depth)")
    parser.add_argument("--white-depth", "-W", type=int, default=None, help="alpha-beta depth for white when using 'ab' (overrides --depth)")
    parser.add_argument("--time", type=float, default=3.0, help="global per-move time budget in seconds for AB strategies (default 3s)")
    parser.add_argument("--black-time", type=float, default=None, help="per-move time budget for black AB strategy (overrides --time)")
    parser.add_argument("--white-time", type=float, default=None, help="per-move time budget for white AB strategy (overrides --time)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)

    args.input = 'start_board.txt'
    with open(args.input, "r") as f:
        lines = [l.rstrip("\n") for l in f.readlines()]

    board = parse_board_lines(lines)

    # pass depth into AB strategies when requested; per-player depth overrides global
    def make(name, is_black=False):
        nl = name.lower()
        if nl == 'ab':
            # choose per-player depth if provided, otherwise fall back to global
            if is_black:
                d = args.black_depth if args.black_depth is not None else args.depth
                t = args.black_time if args.black_time is not None else args.time
            else:
                d = args.white_depth if args.white_depth is not None else args.depth
                t = args.white_time if args.white_time is not None else args.time
            return AlphaBetaStrategy(depth=d, max_time=t)
        if nl in ('ab2', 'ab_improved', 'ab+'):
            if is_black:
                d = args.black_depth if args.black_depth is not None else args.depth
                t = args.black_time if args.black_time is not None else args.time
            else:
                d = args.white_depth if args.white_depth is not None else args.depth
                t = args.white_time if args.white_time is not None else args.time
            return AlphaBetaImprovedStrategy(depth=d, max_time=t)
        if nl in ('ab3', 'ab_opt'):
            if is_black:
                d = args.black_depth if args.black_depth is not None else args.depth
                t = args.black_time if args.black_time is not None else args.time
            else:
                d = args.white_depth if args.white_depth is not None else args.depth
                t = args.white_time if args.white_time is not None else args.time
            return AlphaBetaOptimizedStrategy(depth=d, max_time=t)
        if nl in ('ab_tt', 'ab_transposition'):
            if is_black:
                d = args.black_depth if args.black_depth is not None else args.depth
                t = args.black_time if args.black_time is not None else args.time
            else:
                d = args.white_depth if args.white_depth is not None else args.depth
                t = args.white_time if args.white_time is not None else args.time
            return AlphaBetaTTStrategy(depth=d, max_time=t)
        if nl in ('ab_bit', 'ab_bitboard'):
            if is_black:
                d = args.black_depth if args.black_depth is not None else args.depth
                t = args.black_time if args.black_time is not None else args.time
            else:
                d = args.white_depth if args.white_depth is not None else args.depth
                t = args.white_time if args.white_time is not None else args.time
            return AlphaBetaBitboardStrategy(depth=d, max_time=t)
        if nl in ('ab_bit_tt', 'ab_bitboard_tt'):
            if is_black:
                d = args.black_depth if args.black_depth is not None else args.depth
                t = args.black_time if args.black_time is not None else args.time
            else:
                d = args.white_depth if args.white_depth is not None else args.depth
                t = args.white_time if args.white_time is not None else args.time
            return AlphaBetaBitboardTTStrategy(depth=d, max_time=t)
        return strategy_from_name(name)

    black = make(args.black, is_black=True)
    white = make(args.white, is_black=False)

    print(f"Black (player 1) = {args.black}, White (player 2) = {args.white}")
    
    total_empty = sum(row.count(0) for row in board)
    pbar = tqdm(total=total_empty, desc="Game Progress", unit="move")
    
    stats = {
        1: {'depths': [], 'max_adv': -float('inf')},
        2: {'depths': [], 'max_adv': -float('inf')}
    }

    def progress(board, me):
        pbar.update(1)
        strat = black if me == 1 else white
        depth = getattr(strat, 'last_depth', None)
        if depth is not None:
            stats[me]['depths'].append(depth)
        
        counts = engine.count_disks(board)
        diff = counts[1] - counts[2]
        
        if diff > stats[1]['max_adv']:
            stats[1]['max_adv'] = diff
        if -diff > stats[2]['max_adv']:
            stats[2]['max_adv'] = -diff

    final_board, counts, winner = play_game(board, black, white, verbose=args.verbose, progress_callback=progress)
    pbar.close()

    avg_depth_p1 = sum(stats[1]['depths']) / len(stats[1]['depths']) if stats[1]['depths'] else 0
    avg_depth_p2 = sum(stats[2]['depths']) / len(stats[2]['depths']) if stats[2]['depths'] else 0
    
    min_depth_p1 = min(stats[1]['depths']) if stats[1]['depths'] else 0
    max_depth_p1 = max(stats[1]['depths']) if stats[1]['depths'] else 0
    min_depth_p2 = min(stats[2]['depths']) if stats[2]['depths'] else 0
    max_depth_p2 = max(stats[2]['depths']) if stats[2]['depths'] else 0

    print("Final board:")
    print_board(final_board)
    
    print("\nFinal Statistics:")
    print("-" * 60)
    print(f"{'Metric':<20} | {'Black (1)':<18} | {'White (2)':<18}")
    print("-" * 60)
    print(f"{'Strategy':<20} | {args.black:<18} | {args.white:<18}")
    print(f"{'Score':<20} | {counts[1]:<18} | {counts[2]:<18}")
    print(f"{'Average Depth':<20} | {avg_depth_p1:<18.2f} | {avg_depth_p2:<18.2f}")
    print(f"{'Min/Max Depth':<20} | {f'{min_depth_p1}/{max_depth_p1}':<18} | {f'{min_depth_p2}/{max_depth_p2}':<18}")
    print(f"{'Max Advantage':<20} | {stats[1]['max_adv']:<18} | {stats[2]['max_adv']:<18}")
    print("-" * 60)

    if winner == 0:
        print("Result: Tie")
    elif winner == 1:
        print(f"Winner: Black (player 1) — {args.black}")
    else:
        print(f"Winner: White (player 2) — {args.white}")


if __name__ == '__main__':
    main()
