#!/usr/bin/env python3
import argparse
import reversi
from play import parse_board_lines, strategy_from_name
import engine
from copy import deepcopy


class HumanStrategy:
    name = "human"

    def __init__(self, prompt_prefix=None):
        self.prompt_prefix = prompt_prefix or "Your"
        # last board state seen by this human strategy (used to highlight
        # opponent's most-recent move when the human is shown the board)
        self.last_board = None

    def choose_move(self, board, me=1, opp=2):
        # list legal moves
        reversi.board_global = board
        moves = reversi.get_legal_moves(board, me, opp)
        if not moves:
            # no moves -> return None so engine treats it as pass
            return None

        # determine highlight positions: cells changed to opponent since last seen
        highlight = set()
        if self.last_board is not None:
            for r in range(len(board)):
                for c in range(len(board[r])):
                    prev = None
                    try:
                        prev = self.last_board[r][c]
                    except Exception:
                        prev = None
                    cur = board[r][c]
                    if prev != cur and cur == opp:
                        highlight.add((r+1, c+1))

        # print board and moves (overlay move indices on the board)
        moves_map = {}
        for i, mv in enumerate(moves, start=1):
            r, c, _ = mv
            moves_map[(r+1, c+1)] = str(i)
        print_board(board, moves_map=moves_map, highlight_positions=highlight)
        print()
        print(f"{self.prompt_prefix} turn (player {me}). Legal moves:")
        # show 1-based coords
        indexed = {}
        for i, mv in enumerate(moves, start=1):
            r, c, flips = mv
            print(f"  {i:2d}: row={r+1}, col={c+1} flips={len(flips)}")
            indexed[(r+1, c+1)] = mv

        # prompt until valid input
        while True:
            s = input("Enter move as 'row col' (or index number): ").strip()
            if not s:
                continue
            if s.lower() in ("q", "quit", "exit"):
                print("Resigning / quitting.")
                return None
            # try index
            try:
                idx = int(s)
                if 1 <= idx <= len(moves):
                    chosen = moves[idx-1]
                    # simulate applying our chosen move so last_board reflects
                    # the board after our move (the engine will also apply it)
                    try:
                        nb = deepcopy(board)
                        engine.apply_move(nb, chosen, me)
                        self.last_board = nb
                    except Exception:
                        pass
                    return chosen
            except Exception:
                pass
            parts = s.split()
            if len(parts) >= 2:
                try:
                    rr = int(parts[0])
                    cc = int(parts[1])
                    if (rr, cc) in indexed:
                        chosen = indexed[(rr, cc)]
                        try:
                            nb = deepcopy(board)
                            engine.apply_move(nb, chosen, me)
                            self.last_board = nb
                        except Exception:
                            pass
                        return chosen
                except Exception:
                    pass
            print("Invalid move. Try again (enter index or 'row col').")


def print_board(board, moves_map=None, highlight_positions=None):
    # Improved board printing with row and column coordinates.
    # Prints each row with its row number on the left and a per-row
    # column header above the cells. Uses a diamond-centred layout so
    # columns align with offsets.
    max_len = max(len(r) for r in board)
    row_offsets = [(max_len - len(board[r])) // 2 for r in range(len(board))]

    def cell_symbol(v):
        if v == 1:
            return '●'
        if v == 2:
            return '○'
        return '.'

    # print a single global column header on top
    header = '   ' + ' '.join(f"{c:>2}" for c in range(1, max_len + 1))
    print(header)

    # For each row print the row number and the centered cells. If moves_map
    # provides an index for a (row,col) position, show that index instead
    # of the symbol to make available moves visible.
    for i, row in enumerate(board, start=1):
        offset = row_offsets[i-1]
        cell_strs = []
        for j, v in enumerate(row, start=1):
            # highlight (opponent's most recent move) takes precedence
            if highlight_positions and (i, j) in highlight_positions:
                cell_strs.append(f">{cell_symbol(v)}")
            elif moves_map and (i, j) in moves_map:
                marker = moves_map[(i, j)]
                cell_strs.append(f"{marker:>2}")
            else:
                cell_strs.append(f"{cell_symbol(v):>2}")
        cells = '   ' * offset + ' '.join(cell_strs)
        print(f"{i:>2} {cells}")


def make_strategy(name, is_black=False, depth=3, time_budget=3.0):
    name = name.lower()
    if name == 'human':
        return HumanStrategy(prompt_prefix=('Black' if is_black else 'White'))
    return strategy_from_name(name)


def main():
    parser = argparse.ArgumentParser(description='Play human vs strategy')
    parser.add_argument('--board', default='start_board.txt')
    parser.add_argument('--side', choices=['black', 'white'], default='black', help='Which side you play')
    parser.add_argument('--opponent', default='ab2', help='Opponent strategy name (random, greedy, corner, ab, ab2)')
    parser.add_argument('--depth', type=int, default=4, help='Depth for opponent AB strategies')
    parser.add_argument('--time', type=float, default=3.0, help='Time budget for opponent AB (seconds)')
    args = parser.parse_args()

    with open(args.board) as f:
        lines = [l.rstrip('\n') for l in f.readlines()]
    board = parse_board_lines(lines)

    human_is_black = (args.side == 'black')

    if human_is_black:
        black = HumanStrategy(prompt_prefix='Black')
        white = make_strategy(args.opponent, is_black=False)
    else:
        white = HumanStrategy(prompt_prefix='White')
        black = make_strategy(args.opponent, is_black=True)

    # if opponent is AB-like, pass depth/time through play.make-like behavior
    # best-effort: if the returned strategy has 'depth' attribute leave as-is

    print(f"Starting game: Human plays {'Black' if human_is_black else 'White'}, opponent={args.opponent}")
    final_board, counts, winner = engine.play_game(board, black, white, verbose=True)

    print('\nFinal board:')
    print_board(final_board)
    print(f"Counts: black= {counts[1]} white= {counts[2]}")
    if winner == 0:
        print("Result: tie")
    else:
        print(f"Winner: player {winner}")


if __name__ == '__main__':
    main()
