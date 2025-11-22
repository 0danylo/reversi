#!/usr/bin/env python3
import argparse
from strategies import RandomStrategy, GreedyStrategy, CornerFirstStrategy, AlphaBetaStrategy
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


def strategy_from_name(name):
    name = name.lower()
    if name == 'ab':
        return AlphaBetaStrategy()
    if name == "random":
        return RandomStrategy()
    if name == "greedy":
        return GreedyStrategy()
    if name in ("corner", "corner_first", "cornerfirst"):
        return CornerFirstStrategy()
    raise SystemExit(f"Unknown strategy: {name}")


def print_board(board):
    for r in board:
        print(" ".join(str(x) for x in r))


def main():
    parser = argparse.ArgumentParser(description="Play a single Reversi match between two strategies")
    parser.add_argument("input", help="input file with 8 board lines (diamond layout)")
    parser.add_argument("--black", default="greedy", help="strategy for black (player 1). options: random, greedy, corner")
    parser.add_argument("--white", default="random", help="strategy for white (player 2).")
    parser.add_argument("--depth", type=int, default=3, help="alpha-beta search depth (when using 'ab' strategy)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        lines = [l.rstrip("\n") for l in f.readlines()]

    board = parse_board_lines(lines)

    # pass depth into AB strategies when requested
    def make(name):
        if name.lower() == 'ab':
            return AlphaBetaStrategy(depth=args.depth)
        return strategy_from_name(name)

    black = make(args.black)
    white = make(args.white)

    final_board, counts, winner = play_game(board, black, white, verbose=args.verbose)

    print("Final board:")
    print_board(final_board)
    print(f"Counts: black= {counts[1]} white= {counts[2]}")
    if winner == 0:
        print("Result: tie")
    else:
        print(f"Winner: player {winner}")


if __name__ == '__main__':
    main()
