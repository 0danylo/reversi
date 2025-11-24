#!/usr/bin/env python3
"""Profile AB2: run several games and record the depth reached by AB2 per move.

This script runs games between `ab2` and an opponent (default `greedy`) and
records the `last_depth` attribute set by `AlphaBetaImprovedStrategy` when it
receives iterative-deepening updates from the worker.
"""
import argparse
from copy import deepcopy
from play import parse_board_lines, parse_board_lines_rect
import strategies
import engine


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


def run_profile(board, ab2_is_black, opponent_name, games=6, depth=3, time_budget=1.0, rect=False, verbose=False):
    depths_record = []  # list of depths reached by AB2 per move
    for i in range(games):
        # alternate colors each game
        if i % 2 == 0:
            black_name = 'ab2' if ab2_is_black else opponent_name
            white_name = opponent_name if ab2_is_black else 'ab2'
        else:
            black_name = opponent_name if ab2_is_black else 'ab2'
            white_name = 'ab2' if ab2_is_black else opponent_name

        black = make_strategy(black_name, depth, time_budget)
        white = make_strategy(white_name, depth, time_budget)

        b = deepcopy(board)
        current = 0
        players = [(1, black), (2, white)]
        consecutive_passes = 0

        while True:
            me, strat = players[current]
            opp = 1 if me == 2 else 2
            reversi = __import__('reversi')
            reversi.board_global = b
            moves = reversi.get_legal_moves(b, me, opp)
            if moves:
                move = strat.choose_move(b, me, opp) if strat else None
                if move is None:
                    move = moves[0]
                # If this move was produced by AB2, check strat.last_depth
                if isinstance(strat, strategies.AlphaBetaImprovedStrategy):
                    d = getattr(strat, 'last_depth', None)
                    depths_record.append(d)
                engine.apply_move(b, move, me)
                consecutive_passes = 0
            else:
                consecutive_passes += 1
                if consecutive_passes >= 2:
                    break
            current = 1 - current

    return depths_record


def summarize(depths):
    from statistics import mean
    cleaned = [d for d in depths if isinstance(d, int)]
    print(f"Samples: {len(depths)} AB2 moves recorded ({len(cleaned)} with numeric depth)")
    if cleaned:
        print(f"Average depth reached: {mean(cleaned):.2f}")
        # simple histogram
        from collections import Counter
        cnt = Counter(cleaned)
        for depth in sorted(cnt):
            print(f"  depth {depth}: {cnt[depth]} moves")
    else:
        print("No numeric depth data was collected; ensure AB2 worker is emitting depth tuples.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board', default='8x8/start_board.txt')
    parser.add_argument('--rect', action='store_true')
    parser.add_argument('--games', type=int, default=6)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--time', type=float, default=1.0)
    parser.add_argument('--opponent', default='greedy')
    parser.add_argument('--ab2-black', action='store_true', help='Make ab2 play black for all games (otherwise alternate)')
    args = parser.parse_args()

    with open(args.board) as f:
        lines = [l.rstrip('\n') for l in f.readlines()]
    if args.rect:
        board = parse_board_lines_rect(lines)
    else:
        board = parse_board_lines(lines)

    depths = run_profile(board, args.ab2_black, args.opponent, games=args.games, depth=args.depth, time_budget=args.time, rect=args.rect)
    summarize(depths)


if __name__ == '__main__':
    main()
