"""
texel_generate.py — Generate a labeled dataset of FEN positions with Stockfish evals.

Produces a CSV with columns [fen, cp] where cp is the centipawn score from
white's perspective, clamped to [-3000, 3000]. Mate scores are excluded.

Usage:
    python texel_generate.py
    python texel_generate.py --positions 100000 --eval-depth 12 --stockfish stockfish
    python texel_generate.py --stockfish "C:/stockfish/stockfish.exe"

Install dependencies first:
    pip install chess

Stockfish binary: https://stockfishchess.org/download/
"""

import chess
import chess.engine
import csv
import random
import argparse
import sys
from pathlib import Path


# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_POSITIONS  = 50_000   # dataset size — 50k is a good start; 200k+ is better
DEFAULT_GEN_DEPTH  = 6        # depth for self-play move generation (fast, just for variety)
DEFAULT_EVAL_DEPTH = 10       # depth for Stockfish evals (higher = better labels)
DEFAULT_OUTPUT     = "texel_data.csv"
DEFAULT_STOCKFISH  = "stockfish"   # change to full path if not on PATH


# ── Position generation ──────────────────────────────────────────────────────

def generate_positions(n: int, engine_path: str, gen_depth: int) -> list[str]:
    """
    Generate FEN positions by playing randomised games with Stockfish.

    Strategy:
      1. Make 8–16 random moves for opening variety
      2. Then use Stockfish to play the rest of the game
      3. Collect quiet (non-check) positions from each game
    """
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    positions = []

    print(f"  Generating positions (target: {n:,})...")
    games_played = 0

    try:
        while len(positions) < n:
            board = chess.Board()

            # Randomised opening — ensures dataset variety
            random_moves = random.randint(8, 16)
            for _ in range(random_moves):
                if board.is_game_over():
                    break
                legal = list(board.legal_moves)
                board.push(random.choice(legal))

            # Collect positions as the game plays out
            game_fens = []
            while not board.is_game_over():
                # Skip positions in check — they're noisy for eval tuning
                if not board.is_check():
                    game_fens.append(board.fen())

                result = engine.play(board, chess.engine.Limit(depth=gen_depth))
                board.push(result.move)

            positions.extend(game_fens)
            games_played += 1

            if games_played % 10 == 0:
                pct = min(100, 100 * len(positions) // n)
                print(f"    {len(positions):>8,} / {n:,}  ({pct}%)  [{games_played} games]",
                      end="\r", flush=True)
    finally:
        engine.quit()

    print(f"\n  Generated {len(positions):,} raw positions from {games_played} games.")
    return positions[:n]


# ── Stockfish evaluation ─────────────────────────────────────────────────────

def evaluate_positions(fens: list[str], engine_path: str, eval_depth: int) -> list[tuple]:
    """
    Evaluate each FEN with Stockfish. Returns list of (fen, cp) pairs.
    Mate scores are dropped; all others are clamped to [-3000, 3000].
    """
    engine  = chess.engine.SimpleEngine.popen_uci(engine_path)
    results = []
    skipped = 0

    print(f"  Evaluating {len(fens):,} positions at depth {eval_depth}...")

    try:
        for i, fen in enumerate(fens):
            board = chess.Board(fen)
            info  = engine.analyse(board, chess.engine.Limit(depth=eval_depth))
            score = info["score"].white()

            if score.is_mate():
                # Mate scores are extreme outliers — skip them
                skipped += 1
                continue

            cp = score.score()
            cp = max(-3000, min(3000, cp))   # clamp
            results.append((fen, cp))

            if (i + 1) % 1000 == 0:
                pct = 100 * (i + 1) // len(fens)
                print(f"    {i+1:>8,} / {len(fens):,}  ({pct}%)", end="\r", flush=True)
    finally:
        engine.quit()

    print(f"\n  Evaluated {len(results):,} positions  ({skipped} mate positions skipped).")
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate Texel tuning dataset using Stockfish."
    )
    parser.add_argument("--positions",  type=int, default=DEFAULT_POSITIONS,
                        help=f"Target number of labeled positions (default: {DEFAULT_POSITIONS:,})")
    parser.add_argument("--gen-depth",  type=int, default=DEFAULT_GEN_DEPTH,
                        help=f"Stockfish depth for game generation (default: {DEFAULT_GEN_DEPTH})")
    parser.add_argument("--eval-depth", type=int, default=DEFAULT_EVAL_DEPTH,
                        help=f"Stockfish depth for position evaluation (default: {DEFAULT_EVAL_DEPTH})")
    parser.add_argument("--stockfish",  default=DEFAULT_STOCKFISH,
                        help="Path to Stockfish executable")
    parser.add_argument("--output",     default=DEFAULT_OUTPUT,
                        help=f"Output CSV file (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    # Verify Stockfish is accessible
    try:
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
        engine.quit()
        print(f"✓ Stockfish found: {args.stockfish}")
    except Exception as e:
        print(f"✗ Could not open Stockfish at '{args.stockfish}'")
        print(f"  Error: {e}")
        print("  Download from https://stockfishchess.org/download/ and use --stockfish <path>")
        sys.exit(1)

    print(f"\n[1/2] Generating {args.positions:,} positions...")
    fens = generate_positions(args.positions, args.stockfish, args.gen_depth)

    print(f"\n[2/2] Evaluating positions at depth {args.eval_depth}...")
    labeled = evaluate_positions(fens, args.stockfish, args.eval_depth)

    out = Path(args.output)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fen", "cp"])
        writer.writerows(labeled)

    print(f"\n✓ Saved {len(labeled):,} labeled positions to: {out.resolve()}")
    print(f"\nNext step:")
    print(f"  python texel_train.py --data {args.output}")


if __name__ == "__main__":
    main()
