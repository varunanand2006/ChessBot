"""
texel_generate_lichess.py — Generate a Texel tuning dataset from a Lichess PGN dump.

Streams through a .pgn.zst file from https://database.lichess.org,
extracts quiet positions from real games, evaluates them with Stockfish,
and writes the same texel_data.csv format that texel_train.py expects.

Usage:
    python texel_generate_lichess.py --pgn lichess_db_2013-01.pgn.zst
    python texel_generate_lichess.py --pgn lichess_db_2013-01.pgn.zst --positions 200000 --min-elo 2000
    python texel_generate_lichess.py --pgn lichess_db_2013-01.pgn.zst --stockfish "C:/stockfish/stockfish.exe"

Install dependencies:
    pip install chess zstandard

Download PGN dumps from:
    https://database.lichess.org  (standard games section)
    Recommended starting file: any month — even 2013-01 has ~500k games
"""

import chess
import chess.pgn
import chess.engine
import zstandard
import csv
import io
import random
import argparse
import sys
from pathlib import Path


# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_POSITIONS  = 200_000
DEFAULT_EVAL_DEPTH = 10
DEFAULT_MIN_ELO    = 2000     # skip games below this average rating
DEFAULT_OUTPUT     = "texel_data.csv"
DEFAULT_STOCKFISH  = "stockfish"
DEFAULT_SKIP_MOVES = 8        # ignore the first N moves of each game (opening theory)
DEFAULT_SAMPLE     = 4        # sample 1 position every N moves (avoids correlated positions)


# ── Streaming PGN reader ─────────────────────────────────────────────────────

def stream_pgn(path: str):
    """
    Yield chess.pgn.Game objects from a .pgn or .pgn.zst file.
    Handles both compressed and uncompressed files automatically.
    Never loads the whole file into memory.
    """
    p = Path(path)
    if not p.exists():
        print(f"✗ File not found: {path}")
        sys.exit(1)

    if p.suffix == ".zst":
        fh   = open(path, "rb")
        dctx = zstandard.ZstdDecompressor(max_window_size=2**31)
        raw  = dctx.stream_reader(fh, read_size=65536)
        text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
    else:
        # Plain .pgn file
        text = open(path, encoding="utf-8", errors="replace")

    try:
        while True:
            game = chess.pgn.read_game(text)
            if game is None:
                break
            yield game
    finally:
        text.close()
        if p.suffix == ".zst":
            fh.close()


# ── Rating filter ─────────────────────────────────────────────────────────────

def get_average_elo(game: chess.pgn.Game) -> int:
    """
    Return the average ELO of both players, or 0 if not available.
    Lichess headers use WhiteElo / BlackElo.
    """
    try:
        w = int(game.headers.get("WhiteElo", "0"))
        b = int(game.headers.get("BlackElo", "0"))
        if w > 0 and b > 0:
            return (w + b) // 2
        return 0
    except (ValueError, TypeError):
        return 0


# ── Position extraction ───────────────────────────────────────────────────────

def extract_positions_from_game(
    game: chess.pgn.Game,
    skip_moves: int,
    sample_every: int,
) -> list[str]:
    """
    Walk through a game and collect FEN strings from quiet positions.

    Filters applied:
      - Skip the first `skip_moves` half-moves (opening theory)
      - Skip positions where the side to move is in check (volatile eval)
      - Sample 1 position every `sample_every` moves (reduces correlation)
      - Stop collecting after move 120 (very long games are mostly noise)
    """
    positions = []
    board     = game.board()

    for i, move in enumerate(game.mainline_moves()):
        board.push(move)

        if i < skip_moves:
            continue
        if i > 120:
            break
        if (i - skip_moves) % sample_every != 0:
            continue
        if board.is_check():
            continue

        positions.append(board.fen())

    return positions


# ── Stockfish evaluation ─────────────────────────────────────────────────────

def evaluate_positions(
    fens: list[str],
    engine_path: str,
    eval_depth: int,
) -> list[tuple[str, int]]:
    """
    Evaluate each FEN with Stockfish at the given depth.
    Returns (fen, cp) pairs where cp is white-relative centipawns,
    clamped to [-3000, 3000]. Mate scores are dropped.
    """
    engine  = chess.engine.SimpleEngine.popen_uci(engine_path)
    results = []
    skipped = 0
    n       = len(fens)

    print(f"  Evaluating {n:,} positions at depth {eval_depth}...")

    try:
        for i, fen in enumerate(fens):
            board = chess.Board(fen)
            info  = engine.analyse(board, chess.engine.Limit(depth=eval_depth))
            score = info["score"].white()

            if score.is_mate():
                skipped += 1
                continue

            cp = score.score()
            cp = max(-3000, min(3000, cp))
            results.append((fen, cp))

            if (i + 1) % 1000 == 0:
                pct = 100 * (i + 1) // n
                print(f"    {i+1:>8,} / {n:,}  ({pct}%)", end="\r", flush=True)

    finally:
        engine.quit()

    print(f"\n  Evaluated {len(results):,} positions  "
          f"({skipped} mate positions skipped).")
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate Texel tuning dataset from a Lichess PGN dump."
    )
    parser.add_argument("--pgn",        required=True,
                        help="Path to .pgn or .pgn.zst file from database.lichess.org")
    parser.add_argument("--positions",  type=int,   default=DEFAULT_POSITIONS,
                        help=f"Target positions to collect (default: {DEFAULT_POSITIONS:,})")
    parser.add_argument("--eval-depth", type=int,   default=DEFAULT_EVAL_DEPTH,
                        help=f"Stockfish eval depth (default: {DEFAULT_EVAL_DEPTH})")
    parser.add_argument("--min-elo",    type=int,   default=DEFAULT_MIN_ELO,
                        help=f"Minimum average game ELO (default: {DEFAULT_MIN_ELO})")
    parser.add_argument("--skip-moves", type=int,   default=DEFAULT_SKIP_MOVES,
                        help=f"Opening moves to skip per game (default: {DEFAULT_SKIP_MOVES})")
    parser.add_argument("--sample",     type=int,   default=DEFAULT_SAMPLE,
                        help=f"Sample 1 position every N moves (default: {DEFAULT_SAMPLE})")
    parser.add_argument("--stockfish",  default=DEFAULT_STOCKFISH,
                        help="Path to Stockfish executable")
    parser.add_argument("--output",     default=DEFAULT_OUTPUT,
                        help=f"Output CSV file (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    # ── Verify Stockfish ──────────────────────────────────────────
    try:
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
        engine.quit()
        print(f"✓ Stockfish found: {args.stockfish}")
    except Exception as e:
        print(f"✗ Could not open Stockfish at '{args.stockfish}'")
        print(f"  Error: {e}")
        sys.exit(1)

    # ── Verify PGN file ───────────────────────────────────────────
    pgn_path = Path(args.pgn)
    if not pgn_path.exists():
        print(f"✗ PGN file not found: {pgn_path}")
        print(f"  Download from https://database.lichess.org")
        sys.exit(1)

    size_mb = pgn_path.stat().st_size / 1_000_000
    print(f"✓ PGN file: {pgn_path.name}  ({size_mb:.0f} MB compressed)")

    # ── Stream games and collect positions ───────────────────────
    print(f"\n[1/2] Collecting {args.positions:,} positions "
          f"(min ELO: {args.min_elo}, skip first {args.skip_moves} moves, "
          f"sample every {args.sample})...")

    all_fens      = []
    games_read    = 0
    games_used    = 0
    games_skipped = 0

    for game in stream_pgn(args.pgn):
        games_read += 1

        # Filter by rating
        avg_elo = get_average_elo(game)
        if avg_elo > 0 and avg_elo < args.min_elo:
            games_skipped += 1
            continue

        # Skip games with no moves (corrupt entries)
        if game.mainline_moves() is None:
            games_skipped += 1
            continue

        fens = extract_positions_from_game(game, args.skip_moves, args.sample)
        if fens:
            all_fens.extend(fens)
            games_used += 1

        # Progress update every 500 games
        if games_read % 500 == 0:
            pct = min(100, 100 * len(all_fens) // args.positions)
            print(f"  {len(all_fens):>8,} / {args.positions:,}  ({pct}%)  "
                  f"[{games_read:,} games read, {games_skipped:,} skipped]",
                  end="\r", flush=True)

        if len(all_fens) >= args.positions:
            break

    print(f"\n  Collected {len(all_fens):,} raw positions from "
          f"{games_used:,} games  ({games_skipped:,} skipped below ELO threshold).")

    if len(all_fens) < args.positions:
        print(f"\n  Warning: only found {len(all_fens):,} positions "
              f"(target was {args.positions:,}).")
        print(f"  Try lowering --min-elo or using a larger PGN file.")

    # Shuffle so evaluation order is random (avoids any systematic bias)
    random.shuffle(all_fens)
    all_fens = all_fens[:args.positions]

    # ── Evaluate with Stockfish ───────────────────────────────────
    print(f"\n[2/2] Evaluating positions...")
    labeled = evaluate_positions(all_fens, args.stockfish, args.eval_depth)

    # ── Write CSV ─────────────────────────────────────────────────
    out = Path(args.output)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fen", "cp"])
        writer.writerows(labeled)

    print(f"\n✓ Saved {len(labeled):,} labeled positions to: {out.resolve()}")
    print(f"\nNext step:")
    print(f"  python texel_train.py --data {args.output}")


if __name__ == "__main__":
    main()