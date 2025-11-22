from copy import deepcopy
import reversi


def apply_move(board, move, me):
    """Apply a move (r,c,flips) to the board in-place."""
    r, c, flips = move
    board[r][c] = me
    for fr, fc in flips:
        board[fr][fc] = me


def count_disks(board):
    cnt1 = 0
    cnt2 = 0
    for row in board:
        for v in row:
            if v == 1:
                cnt1 += 1
            elif v == 2:
                cnt2 += 1
    return {1: cnt1, 2: cnt2}


def play_game(board, black_strategy, white_strategy, verbose=False):
    """Play a single game between two strategy objects.

    black_strategy: strategy for player 1
    white_strategy: strategy for player 2
    Returns: (final_board, counts, winner) where winner is 1,2 or 0 for tie.
    """
    board = deepcopy(board)

    players = [(1, black_strategy), (2, white_strategy)]
    current = 0
    consecutive_passes = 0

    while True:
        me, strat = players[current]
        opp = 1 if me == 2 else 2

        # ensure helper functions that rely on board_global work
        reversi.board_global = board

        moves = reversi.get_legal_moves(board, me, opp)
        if moves:
            # ask strategy for a move
            move = strat.choose_move(board, me, opp) if strat else None
            if move is None:
                # fallback to first legal move
                move = moves[0]

            apply_move(board, move, me)
            consecutive_passes = 0
            if verbose:
                print(f"Player {me} plays {(move[0]+1, move[1]+1)} flipping {len(move[2])}")
        else:
            consecutive_passes += 1
            if verbose:
                print(f"Player {me} passes")
            if consecutive_passes >= 2:
                break

        current = 1 - current

    counts = count_disks(board)
    if counts[1] > counts[2]:
        winner = 1
    elif counts[2] > counts[1]:
        winner = 2
    else:
        winner = 0

    return board, counts, winner
