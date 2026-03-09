"""
texel_train.py — Texel tuning for chess piece-square tables using PyTorch.

Loads the dataset produced by texel_generate.py, trains a linear model
whose weights ARE the PST values, then writes the tuned tables to
tuned_constants.py ready to drop into your constants.py.

The model is intentionally simple:
    score = feature_vector @ weights

where feature_vector encodes the board position and weights are the
PST/piece-value parameters. This is exactly how your evaluate() works —
the linear structure means PyTorch can compute exact gradients with no
approximation.

Usage:
    python texel_train.py
    python texel_train.py --data texel_data.csv --epochs 1000 --lr 1.0

Install dependencies:
    pip install torch chess
"""

import torch
import torch.optim as optim
import chess
import csv
import numpy as np
import argparse
import sys
from pathlib import Path


# ── Weight layout ────────────────────────────────────────────────────────────
#
#  Index range    Content
#  ─────────────  ────────────────────────────────────────────────────────────
#  [  0 ..  63]   PAWN_TABLE      (row-major, 8×8)
#  [ 64 .. 127]   KNIGHT_TABLE
#  [128 .. 191]   BISHOP_TABLE
#  [192 .. 255]   ROOK_TABLE
#  [256 .. 319]   QUEEN_TABLE
#  [320 .. 383]   KING_TABLE      (middlegame: total_material >= 1500)
#  [384 .. 447]   KING_END_TABLE  (endgame:    total_material <  1500)
#  [448]          piece value: PAWN
#  [449]          piece value: KNIGHT
#  [450]          piece value: BISHOP
#  [451]          piece value: ROOK
#  [452]          piece value: QUEEN
#
#  Total: 453 learnable parameters.

N_WEIGHTS = 453

PST_OFFSETS = {
    chess.PAWN:   0,
    chess.KNIGHT: 64,
    chess.BISHOP: 128,
    chess.ROOK:   192,
    chess.QUEEN:  256,
    # King handled separately below
}
KING_MG_OFFSET = 320
KING_EG_OFFSET = 384

VALUE_INDICES = {
    chess.PAWN:   448,
    chess.KNIGHT: 449,
    chess.BISHOP: 450,
    chess.ROOK:   451,
    chess.QUEEN:  452,
}

# Centipawn values used only for king-table threshold detection,
# NOT as learnable parameters in this dict.
_CP = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
       chess.ROOK: 500, chess.QUEEN: 900}


# ── Feature extraction ───────────────────────────────────────────────────────

def _pst_index(sq: int, is_white: bool) -> int:
    """
    Map a python-chess square + side to the row-major index used in
    your constants.py PST tables.

    Your layout: squares[r][c], r=0 → rank 8 (black's back), r=7 → rank 1.
    White pieces use table[r][c] directly.
    Black pieces use table[7-r][c]  (vertical mirror).

    python-chess: square_rank(sq) = 0 → rank 1, 7 → rank 8.
    """
    file = chess.square_file(sq)   # 0 = a-file
    rank = chess.square_rank(sq)   # 0 = rank 1

    # White: their row r = 7 - rank  →  pst_index = (7 - rank) * 8 + file
    # Black: mirrored → table[7-r][c] where r = 7-rank → 7-r = rank
    #                   pst_index = rank * 8 + file
    row = (7 - rank) if is_white else rank
    return row * 8 + file


def extract_features(fen: str) -> np.ndarray:
    """
    Return the 453-element feature vector F such that F @ W = engine score.

    For each piece on the board:
      • White piece at PST index i → F[offset + i] += 1
      • Black piece at mirrored PST index i → F[offset + i] -= 1
      • Piece value weight index j → F[j] += 1 (white) or -= 1 (black)

    The king table used (mg vs eg) is determined by total non-king material,
    matching the threshold in your search.py evaluate() function.
    """
    board    = chess.Board(fen)
    features = np.zeros(N_WEIGHTS, dtype=np.float32)

    # Decide which king table to use (binary phase, matches your evaluate())
    total_mat = sum(
        len(board.pieces(pt, chess.WHITE)) * _CP[pt] +
        len(board.pieces(pt, chess.BLACK)) * _CP[pt]
        for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    )
    king_offset = KING_EG_OFFSET if total_mat < 1500 else KING_MG_OFFSET

    for piece_type in chess.PIECE_TYPES:
        if piece_type == chess.KING:
            for sq in board.pieces(chess.KING, chess.WHITE):
                features[king_offset + _pst_index(sq, True)]  += 1.0
            for sq in board.pieces(chess.KING, chess.BLACK):
                features[king_offset + _pst_index(sq, False)] -= 1.0
        else:
            pst_off = PST_OFFSETS[piece_type]
            val_idx = VALUE_INDICES[piece_type]
            for sq in board.pieces(piece_type, chess.WHITE):
                features[pst_off + _pst_index(sq, True)] += 1.0
                features[val_idx]                         += 1.0
            for sq in board.pieces(piece_type, chess.BLACK):
                features[pst_off + _pst_index(sq, False)] -= 1.0
                features[val_idx]                          -= 1.0

    return features


# ── Weight initialisation ────────────────────────────────────────────────────

def load_initial_weights() -> np.ndarray:
    """
    Seed the weight vector from your existing constants.py tables so that
    training starts from your current hand-tuned values, not random noise.
    This means epoch 0 already plays at your current strength.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from constants import (
            PAWN_TABLE, KNIGHT_TABLE, BISHOP_TABLE,
            ROOK_TABLE, QUEEN_TABLE, KING_TABLE, KING_END_TABLE,
            PIECE_VALUES, PAWN, KNIGHT, BISHOP, ROOK, QUEEN,
        )
    except ImportError:
        print("  Warning: could not import constants.py — starting from scratch.")
        return np.zeros(N_WEIGHTS, dtype=np.float32)

    w = np.zeros(N_WEIGHTS, dtype=np.float32)
    for table, offset in [
        (PAWN_TABLE,      0),
        (KNIGHT_TABLE,   64),
        (BISHOP_TABLE,  128),
        (ROOK_TABLE,    192),
        (QUEEN_TABLE,   256),
        (KING_TABLE,    320),
        (KING_END_TABLE, 384),
    ]:
        for r in range(8):
            for c in range(8):
                w[offset + r * 8 + c] = float(table[r][c])

    w[448] = float(PIECE_VALUES[PAWN])
    w[449] = float(PIECE_VALUES[KNIGHT])
    w[450] = float(PIECE_VALUES[BISHOP])
    w[451] = float(PIECE_VALUES[ROOK])
    w[452] = float(PIECE_VALUES[QUEEN])

    return w


# ── Loss ─────────────────────────────────────────────────────────────────────

def sigmoid_loss(pred: torch.Tensor, target: torch.Tensor, K: float = 0.004) -> torch.Tensor:
    """
    Texel sigmoid MSE loss.

    Maps centipawn scores to win-probability space [0, 1] before comparing.
    This is essential — raw MSE on centipawns would over-weight outliers
    (e.g. +2000 cp positions) and under-weight the critical ±200 cp range
    where most practical games are decided.

    K is the sigmoid scaling factor. Smaller K → softer sigmoid.
    0.004 corresponds to ~250 cp → ~73% win probability, which is reasonable.
    """
    pred_p   = torch.sigmoid(K * pred)
    target_p = torch.sigmoid(K * target)
    return ((pred_p - target_p) ** 2).mean()


# ── Weight → table conversion ────────────────────────────────────────────────

def weights_to_tables(w: np.ndarray) -> dict:
    def to_table(segment):
        return [[round(float(segment[r * 8 + c])) for c in range(8)] for r in range(8)]

    return {
        "PAWN_TABLE":     to_table(w[0:64]),
        "KNIGHT_TABLE":   to_table(w[64:128]),
        "BISHOP_TABLE":   to_table(w[128:192]),
        "ROOK_TABLE":     to_table(w[192:256]),
        "QUEEN_TABLE":    to_table(w[256:320]),
        "KING_TABLE":     to_table(w[320:384]),
        "KING_END_TABLE": to_table(w[384:448]),
        "PIECE_VALUES": {
            "PAWN":   round(float(w[448])),
            "KNIGHT": round(float(w[449])),
            "BISHOP": round(float(w[450])),
            "ROOK":   round(float(w[451])),
            "QUEEN":  round(float(w[452])),
        },
    }


def write_output(tables: dict, path: str):
    pv = tables["PIECE_VALUES"]
    table_names = ["PAWN_TABLE", "KNIGHT_TABLE", "BISHOP_TABLE",
                   "ROOK_TABLE", "QUEEN_TABLE", "KING_TABLE", "KING_END_TABLE"]

    lines = [
        "# ──────────────────────────────────────────────────────────────",
        "# Texel-tuned weights — generated by texel_train.py",
        "# Copy these blocks into constants.py, replacing the existing tables.",
        "# ──────────────────────────────────────────────────────────────",
        "",
        "PIECE_VALUES = {",
        f"    PAWN:   {pv['PAWN']},",
        f"    KNIGHT: {pv['KNIGHT']},",
        f"    BISHOP: {pv['BISHOP']},",
        f"    ROOK:   {pv['ROOK']},",
        f"    QUEEN:  {pv['QUEEN']},",
        "    KING:   0",
        "}",
        "",
    ]

    for name in table_names:
        table = tables[name]
        lines.append(f"{name} = [")
        for row in table:
            lines.append(f"[{', '.join(f'{v:4d}' for v in row)}],")
        lines.append("]")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Texel tuning: optimise chess piece-square tables with PyTorch."
    )
    parser.add_argument("--data",   default="texel_data.csv",
                        help="CSV from texel_generate.py (default: texel_data.csv)")
    parser.add_argument("--epochs", type=int,   default=1000,
                        help="Training epochs (default: 1000)")
    parser.add_argument("--lr",     type=float, default=1.0,
                        help="Adam learning rate in centipawn units (default: 1.0)")
    parser.add_argument("--batch",  type=int,   default=4096,
                        help="Mini-batch size (default: 4096)")
    parser.add_argument("--K",      type=float, default=0.004,
                        help="Sigmoid scaling factor (default: 0.004)")
    parser.add_argument("--l2",     type=float, default=0.0001,
                        help="L2 regularization strength for PST weights (default: 0.0001). "
                             "Higher values keep PST magnitudes smaller and more stable. "
                             "Try 0.001 if tables still look too large.")
    parser.add_argument("--output", default="tuned_constants.py",
                        help="Output file (default: tuned_constants.py)")
    args = parser.parse_args()

    # ── Load dataset ─────────────────────────────────────────────
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"✗ Data file not found: {data_path}")
        print("  Run texel_generate.py first.")
        sys.exit(1)

    print(f"Loading dataset from {data_path}...")
    fens, scores = [], []
    with open(data_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fens.append(row["fen"])
            scores.append(float(row["cp"]))

    n = len(fens)
    print(f"  {n:,} positions loaded.")
    if n < 10_000:
        print("  Warning: dataset is small (<10k). Consider generating more positions.")

    # ── Extract features ─────────────────────────────────────────
    print(f"\nExtracting features ({N_WEIGHTS} weights per position)...")
    feat_list = []
    for i, fen in enumerate(fens):
        feat_list.append(extract_features(fen))
        if (i + 1) % 5000 == 0:
            print(f"  {i+1:>8,} / {n:,}", end="\r", flush=True)

    X = torch.tensor(np.array(feat_list, dtype=np.float32))   # [n, 453]
    y = torch.tensor(np.array(scores,    dtype=np.float32))   # [n]
    print(f"\n  Feature matrix: {tuple(X.shape)}  dtype: {X.dtype}")

    # ── Initialise weights from existing constants.py ────────────
    print("\nLoading initial weights from constants.py...")
    init_w = load_initial_weights()
    W = torch.tensor(init_w, requires_grad=True)

    # Baseline loss (before any training)
    with torch.no_grad():
        baseline = sigmoid_loss(X @ W, y, K=args.K).item()
    print(f"  Baseline loss (epoch 0): {baseline:.6f}")

    # ── Optimiser ────────────────────────────────────────────────
    optimizer = optim.Adam([W], lr=args.lr)

    best_loss = baseline
    best_W    = W.detach().clone()

    print(f"\nTraining  |  epochs={args.epochs}  lr={args.lr}  batch={args.batch}  K={args.K}  l2={args.l2}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        perm       = torch.randperm(n)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n, args.batch):
            idx  = perm[start : start + args.batch]
            X_b  = X[idx]
            y_b  = y[idx]

            pred = X_b @ W

            # Sigmoid MSE loss (Texel tuning objective)
            sig_loss = sigmoid_loss(pred, y_b, K=args.K)

            # L2 regularization on PST weights only (indices 0-447).
            # Piece values (448-452) are NOT regularized — we want those
            # to find their natural scale. The PST weights are penalized
            # toward zero, which prevents the large magnitude explosion
            # seen without regularization.
            l2_reg = args.l2 * (W[:448] ** 2).mean()

            loss = sig_loss + l2_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Soft-clamp piece values to sane range (50–2000 cp)
            # Prevents the model from zeroing out piece values
            with torch.no_grad():
                W[448:453].clamp_(50.0, 2000.0)

            epoch_loss += sig_loss.item()   # track sigmoid loss only, not l2 term
            n_batches  += 1

        avg_loss = epoch_loss / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_W    = W.detach().clone()
            torch.save(best_W, "best_weights.pt")   # checkpoint

        if epoch % 100 == 0 or epoch == 1:
            # Also print max PST magnitude so you can see regularization working
            pst_max = W[:448].abs().max().item()
            improvement = (baseline - best_loss) / baseline * 100
            print(f"  Epoch {epoch:5d}  |  loss: {avg_loss:.6f}  "
                  f"|  best: {best_loss:.6f}  ({improvement:+.2f}%)  "
                  f"|  max PST: {pst_max:.1f}")

    print("-" * 65)
    print(f"\nTraining complete.  Best loss: {best_loss:.6f}  "
          f"({(baseline - best_loss)/baseline*100:+.2f}% improvement)")

    # ── Export tuned weights ──────────────────────────────────────
    tables = weights_to_tables(best_W.numpy())
    write_output(tables, args.output)
    print(f"\n✓ Tuned weights written to: {Path(args.output).resolve()}")

    # ── Print summary of changes ──────────────────────────────────
    print("\n── Piece value changes ────────────────────────────────────────")
    init_tables = weights_to_tables(init_w)
    pv_old = init_tables["PIECE_VALUES"]
    pv_new = tables["PIECE_VALUES"]
    for name in ["PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN"]:
        old = pv_old[name]
        new = pv_new[name]
        arrow = "↑" if new > old else "↓" if new < old else "="
        print(f"  {name:<7} {old:>5} → {new:>5}  {arrow} {new - old:+d}")

    # PST magnitude report — flag any suspiciously large values
    print("\n── PST magnitude check ────────────────────────────────────────")
    table_names_check = ["PAWN_TABLE", "KNIGHT_TABLE", "BISHOP_TABLE",
                         "ROOK_TABLE", "QUEEN_TABLE", "KING_TABLE", "KING_END_TABLE"]
    for name in table_names_check:
        flat   = [v for row in tables[name] for v in row]
        mx     = max(flat)
        mn     = min(flat)
        warn   = "  ⚠ large values" if (mx > 100 or mn < -100) else ""
        print(f"  {name:<16}  min: {mn:>5}  max: {mx:>5}{warn}")

    print("\n── Instructions ───────────────────────────────────────────────")
    print(f"  1. Open: {Path(args.output).resolve()}")
    print(f"  2. Copy the PIECE_VALUES and *_TABLE blocks into constants.py")
    print(f"  3. Run your engine and benchmark against the old version")
    print("")


if __name__ == "__main__":
    main()