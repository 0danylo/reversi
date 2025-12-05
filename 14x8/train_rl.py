
import argparse
import random
import json
import os
import time
from copy import deepcopy
import engine
import reversi
import strategies
from concurrent.futures import ProcessPoolExecutor

def train_worker(args):
    weights, games, seed = args
    # Initialize strategy with current weights
    # We use a small depth for training to be fast, or maybe larger?
    # Let's use depth 2 for speed in self-play
    rl_strat = strategies.AlphaBetaBitboardRLStrategy(depth=2, max_time=0.5, weights=weights)
    
    # Opponent can be itself (self-play) or a fixed strategy
    # Let's do self-play
    
    wins = 0
    losses = 0
    ties = 0
    
    # Gradient accumulation
    grad_accum = [0.0] * len(weights)
    
    # Learning rate
    alpha = 0.01
    
    random.seed(seed)

    for _ in range(games):
        # Randomize start
        board = reversi.read_board_lines([
            "0 0 0 0 0 0 0 0",
            "0 0 0 0 0 0 0 0 0 0",
            "0 0 0 0 0 0 0 0 0 0 0 0",
            "0 0 0 0 0 0 2 1 0 0 0 0 0 0",
            "0 0 0 0 0 0 1 2 0 0 0 0 0 0",
            "0 0 0 0 0 0 0 0 0 0 0 0",
            "0 0 0 0 0 0 0 0 0 0",
            "0 0 0 0 0 0 0 0"
        ])
        
        # Random moves to diversify
        curr = 1
        for _ in range(4):
            reversi.board_global = board
            moves = reversi.get_legal_moves(board, curr, 3-curr)
            if moves:
                m = random.choice(moves)
                engine.apply_move(board, m, curr)
            curr = 3 - curr
            
        # Play game
        # We need to record features encountered to update weights?
        # Simple approach: Policy Gradient or just update based on final outcome?
        # Since we are learning evaluation function V(s), we can use TD(lambda) or Monte Carlo.
        # Let's use Monte Carlo: V(s) should predict final outcome.
        # Outcome: +1 for win, -1 for loss.
        # We want V(s) ~ Outcome.
        # Loss = (V(s) - Outcome)^2
        # dLoss/dw = 2 * (V(s) - Outcome) * dV/dw
        # dV/dw = features(s)
        # Update: w <- w - lr * (V(s) - Outcome) * features(s)
        
        # We need to track trajectory
        trajectory = [] # (features, player)
        
        b_copy = deepcopy(board)
        players = {1: rl_strat, 2: rl_strat}
        current = 1
        
        while True:
            reversi.board_global = b_copy
            moves = reversi.get_legal_moves(b_copy, current, 3-current)
            if not moves:
                reversi.board_global = b_copy
                if not reversi.get_legal_moves(b_copy, 3-current, current):
                    break
                current = 3 - current
                continue
                
            # Choose move
            # We want to explore a bit? Epsilon-greedy?
            if random.random() < 0.1:
                move = random.choice(moves)
            else:
                move = players[current].choose_move(b_copy, current, 3-current)
                if move is None: move = random.choice(moves) # Should not happen if moves exists
            
            # Record features of the position AFTER move? 
            # Or before? The evaluation is called on leaf nodes.
            # The strategy evaluates positions.
            # Let's record the features of the board state *before* the move, 
            # but from the perspective of the player whose turn it is?
            # Actually, V(s) is usually state value.
            # Our evaluate_bb(me, opp) returns score for 'me'.
            
            me_bb, opp_bb = rl_strat.to_bitboard(b_copy, current, 3-current)
            feats = rl_strat.get_features(me_bb, opp_bb)
            trajectory.append((feats, current))
            
            engine.apply_move(b_copy, move, current)
            current = 3 - current
            
        # Game over
        counts = engine.count_disks(b_copy)
        if counts[1] > counts[2]:
            winner = 1
        elif counts[2] > counts[1]:
            winner = 2
        else:
            winner = 0
            
        # Update weights
        # For each state in trajectory:
        # If winner == current, target = 1000 (arbitrary large score)
        # If winner != current, target = -1000
        # If tie, target = 0
        
        for feats, p in trajectory:
            if winner == 0:
                target = 0
            elif winner == p:
                target = 10000 # Max score roughly
            else:
                target = -10000
            
            # Prediction
            pred = sum(w * f for w, f in zip(weights, feats))
            
            # Error
            error = pred - target
            
            # Accumulate gradient: error * features
            # We want to minimize error^2, so descent direction is -error * features
            # w_new = w - lr * error * features
            # But we accumulate for batch update
            for i in range(len(weights)):
                grad_accum[i] += error * feats[i]
                
    return grad_accum

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games-per-iter", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default="rl_weights.json")
    args = parser.parse_args()
    
    # Initial weights (heuristic)
    weights = [10.0, 1000.0, 50.0, -800.0, 20.0]
    
    print(f"Starting training with weights: {weights}")
    
    for it in range(args.iterations):
        print(f"Iteration {it+1}/{args.iterations}...")
        
        # Run games in parallel
        jobs = []
        for i in range(args.workers):
            jobs.append((weights, args.games_per_iter // args.workers, i + it*100))
            
        total_grad = [0.0] * len(weights)
        
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(train_worker, jobs))
            
        for grad in results:
            for i in range(len(weights)):
                total_grad[i] += grad[i]
                
        # Update weights
        # Normalize gradient by total games
        total_games = args.games_per_iter
        lr = 0.0001 # Small learning rate
        
        # Apply update
        # w = w - lr * (average_grad)
        for i in range(len(weights)):
            avg_grad = total_grad[i] / total_games
            weights[i] -= lr * avg_grad
            
        print(f"  Updated weights: {[f'{w:.2f}' for w in weights]}")
        
    # Save weights
    with open(args.output, 'w') as f:
        json.dump(weights, f)
    print(f"Saved weights to {args.output}")

if __name__ == "__main__":
    main()
