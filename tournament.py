#!/usr/bin/env python3
import argparse
from play import parse_board_lines
import engine
import strategies


def make_strategy(name, depth=None):
    n = name.lower()
    if n == "random":
        return strategies.RandomStrategy()
    if n == "greedy":
        return strategies.GreedyStrategy()
    if n in ("corner", "corner_first", "cornerfirst"):
        return strategies.CornerFirstStrategy()
    if n in ("ab", "alphabeta"):
        return strategies.AlphaBetaStrategy(depth=depth if depth is not None else 3)
    if n in ("ab2", "ab_improved", "ab+"):
        return strategies.AlphaBetaImprovedStrategy(depth=depth if depth is not None else 3)
    raise SystemExit(f"Unknown strategy: {name}")


def compare_strategy(board, target_name, opponents, games_per_opponent=10, depth=None, verbose=False):
    results = {}
    for opp_name in opponents:
        if opp_name == target_name:
            continue
        results[opp_name] = {"target_wins": 0, "opp_wins": 0, "ties": 0}

        for i in range(games_per_opponent):
            # alternate colors: even i -> target is black, odd i -> target is white
            if i % 2 == 0:
                black = make_strategy(target_name, depth)
                white = make_strategy(opp_name, depth)
                target_is_black = True
            else:
                black = make_strategy(opp_name, depth)
                white = make_strategy(target_name, depth)
                target_is_black = False

            final_board, counts, winner = engine.play_game(board, black, white, verbose=verbose)

            if winner == 0:
                results[opp_name]["ties"] += 1
            else:
                # determine if the winner was the target strategy
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
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        lines = [l.rstrip("\n") for l in f.readlines()]

    board = parse_board_lines(lines)

    all_ops = ["random", "greedy", "corner", "ab"]

    results = compare_strategy(board, args.strategy, all_ops, games_per_opponent=args.games, depth=args.depth, verbose=args.verbose)

    print(f"Results for strategy '{args.strategy}' vs others (games per opponent={args.games}):")
    for opp, res in results.items():
        total = res["target_wins"] + res["opp_wins"] + res["ties"]
        win_rate = res["target_wins"] / total if total else 0
        print(f"  vs {opp}: target_wins={res['target_wins']} opp_wins={res['opp_wins']} ties={res['ties']} win_rate={win_rate:.2%}")


if __name__ == '__main__':
    main()
