#!/usr/bin/env python3
import argparse
import time
from copy import deepcopy
from play import parse_board_lines, parse_board_lines_rect, strategy_from_name
import reversi
import engine
import strategies
import human_play


def load_board(path, rect=False):
    with open(path) as f:
        lines = [l.rstrip('\n') for l in f.readlines()]
    if rect:
        return parse_board_lines_rect(lines)
    return parse_board_lines(lines)


def make_strategy(name, depth, max_time):
    name = name.lower()
    if name in ('ab', 'alphabeta'):
        return strategies.AlphaBetaStrategy(depth=depth, max_time=max_time)
    if name in ('ab2', 'ab_improved', 'ab+'):
        return strategies.AlphaBetaImprovedStrategy(depth=depth, max_time=max_time)
    return strategy_from_name(name)


def print_move_info(me, move, elapsed, strat):
    sname = getattr(strat, 'name', strat.__class__.__name__)
    if move is None:
        print(f"Player {me} ({sname}) passes (time={elapsed:.3f}s)")
        return
    r, c, flips = move
    depth = getattr(strat, 'last_depth', None)
    print(f"Player {me} ({sname}) plays {(r+1, c+1)} flipping {len(flips)} (time={elapsed:.3f}s, depth={depth})")


def run_one(board, black_strategy, white_strategy, show_board=True):
    b = deepcopy(board)
    players = [(1, black_strategy), (2, white_strategy)]
    print(f"Starting: Player 1 (Black) = {getattr(black_strategy,'name',black_strategy.__class__.__name__)}, Player 2 (White) = {getattr(white_strategy,'name',white_strategy.__class__.__name__)}")
    current = 0
    consecutive_passes = 0

    while True:
        me, strat = players[current]
        opp = 1 if me == 2 else 2
        reversi.board_global = b
        moves = reversi.get_legal_moves(b, me, opp)
        if moves:
            t0 = time.time()
            move = strat.choose_move(b, me, opp) if strat else None
            t1 = time.time()
            if move is None:
                move = moves[0]
            print_move_info(me, move, t1 - t0, strat)
            engine.apply_move(b, move, me)
            consecutive_passes = 0
            if show_board:
                human_play.print_board(b)
                print()
        else:
            consecutive_passes += 1
            print(f"Player {me} passes")
            if consecutive_passes >= 2:
                break
        current = 1 - current

    counts = engine.count_disks(b)
    print('Final board:')
    human_play.print_board(b)
    print(f"Counts: black= {counts[1]} white= {counts[2]}")
    if counts[1] > counts[2]:
        print('Winner: Black (player 1)')
    elif counts[2] > counts[1]:
        print('Winner: White (player 2)')
    else:
        print('Result: tie')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--black', default='ab2', help='Black strategy')
    parser.add_argument('--white', default='ab', help='White strategy')
    parser.add_argument('--depth', type=int, default=4, help='Depth for AB strategies')
    parser.add_argument('--time', type=float, default=0.5, help='Time for AB strategies')
    parser.add_argument('--rect', action='store_true', help='Use rectangular board parser')
    parser.add_argument('board', nargs='?', default='start_board.txt')
    args = parser.parse_args()

    board = load_board(args.board, rect=args.rect)
    black = make_strategy(args.black, args.depth, args.time)
    white = make_strategy(args.white, args.depth, args.time)
    run_one(board, black, white, show_board=True)
