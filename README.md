# Chess Engine

A fully functional chess engine written in Python from scratch. Features a terminal interface and a Pygame GUI with a dark luxury theme. Built incrementally — each optimization layer is documented below with its purpose and the speed/strength tradeoff it introduces.

**Estimated strength: ~1400–1800 ELO**

![Python](https://img.shields.io/badge/Python-3.11%2B-blue) ![PyPy](https://img.shields.io/badge/PyPy-3.11-orange) ![Pygame](https://img.shields.io/badge/Pygame-2.6-green)

---

## Quick Start

**Terminal (PyPy recommended for full speed):**
```bash
pypy main.py
```

**GUI:**
```bash
# Step 1 — download piece images (run once)
python download_pieces.py

# Step 2 — install pygame
pip install pygame

# Step 3 — launch
python gui.py
```

The GUI (`gui.py`) runs under CPython for Pygame compatibility. It launches `engine_main.py` as a PyPy subprocess and communicates over stdin/stdout, so the engine still runs at full PyPy speed.

---

## Project Structure

| File | Description |
|---|---|
| `constants.py` | Piece types, move flags, Zobrist tables, piece-square tables |
| `board.py` | Board state, make/undo move, Zobrist hashing, attack detection |
| `movegen.py` | Legal move generation for all piece types and special moves |
| `search.py` | Minimax, alpha-beta, quiescence, transposition table, iterative deepening |
| `main.py` | Terminal game loop with dynamic depth scaling |
| `engine_main.py` | Subprocess engine with a clean stdin/stdout protocol for the GUI |
| `gui.py` | Pygame GUI — PNG piece images, move selection, threaded engine |
| `download_pieces.py` | Downloads the Wikimedia Commons chess piece PNG set |

---

## How It Works

Each feature below was added incrementally. Understanding the chain — and the tradeoffs each link introduces — explains how a naive exhaustive search becomes a ~1600 ELO engine.

---

### 1. Depth-Limited Minimax

**What it is:**
Minimax models chess as a two-player zero-sum game. White tries to *maximise* the score; black tries to *minimise* it. The engine explores all sequences of moves to a fixed depth, calls `evaluate()` at the leaves, and propagates the best score back up. This lives in `search.py → find_best_move()` and `minimax()`.

**Why we added it:**
Without search, the engine can only look one move ahead — it has no concept of consequences. Minimax gives it the ability to reason about the future and avoid moves that look good immediately but lead to a worse position several turns later.

**The tradeoff:**
Depth is exponential. With ~35 legal moves per position, depth 5 requires evaluating roughly 35⁵ ≈ 52 million nodes. Every extra depth multiplies work by ~35. This is the fundamental bottleneck every other feature below works to break.

---

### 2. Move Integer Bit Encoding

**What it is:**
Every chess move is packed into a single Python integer using bit shifts. The layout, defined in `movegen.py`:

```
bits  0–2:  from_row
bits  3–5:  from_col
bits  6–8:  to_row
bits  9–11: to_col
bits 12–14: flag  (normal / en passant / castling / promotion type)
```

Encoding and decoding are just shifts and masks:
```python
def encode_move(r1, c1, r2, c2, flag=FLAG_NORMAL):
    return r1 | (c1 << 3) | (r2 << 6) | (c2 << 9) | (flag << 12)

def decode_move(move):
    return move & 7, (move>>3)&7, (move>>6)&7, (move>>9)&7, (move>>12)&7
```

The eight possible flags (`FLAG_NORMAL`, `FLAG_EN_PASSANT`, `FLAG_CASTLE_KINGSIDE`, `FLAG_CASTLE_QUEENSIDE`, and four promotion flags) are all encoded here — no separate object needed for any special move type.

**Why we added it:**
Python objects have significant overhead. Representing each move as a tuple or named object would slow down the millions of move comparisons, list insertions, and transposition table lookups the search performs. A single integer is as cheap as Python gets.

**The tradeoff:**
Essentially none. The only cost is slightly less readable code. The performance gain throughout the search tree is real, and PyPy's JIT particularly loves tight integer operations.

---

### 3. Piece-Square Tables

**What it is:**
Each piece type has a hardcoded 8×8 table of positional bonuses in `constants.py`. Knights are rewarded for central squares, bishops for long diagonals, rooks for the 7th rank and open files. The king has two tables — `KING_TABLE` for the middlegame (incentivising castling and staying behind pawns) and `KING_END_TABLE` for the endgame (incentivising centralisation). The `evaluate()` function in `search.py` switches between them based on total material remaining:

```python
table = KING_END_TABLE if total_material < 1500 else KING_TABLE
```

Black's table values are mirrored using `table[7 - r][c]` so the same tables work for both sides without duplication.

**Why we added it:**
Pure material counting makes the engine play nonsensically — it has no preference for developing knights, no instinct to castle, no understanding that a centralised piece is stronger than a passive one. Piece-square tables encode these principles directly into `evaluate()` with essentially zero search cost.

**The tradeoff:**
Marginally slower evaluation per node (~1–2% overhead), but zero extra nodes searched. The ELO gain is entirely from playing more principled chess, not from deeper search.

---

### 4. Alpha-Beta Pruning

**What it is:**
An extension of minimax that maintains two bounds — `alpha` (the best white can guarantee) and `beta` (the best black can guarantee). When the search proves a subtree cannot affect the final result — because one side already has something better — that entire subtree is pruned. The cutoff condition `beta <= alpha` is checked after every move in `minimax()`. The original alpha and beta values are saved at the start of each node to correctly classify the result for the transposition table.

**Why we added it:**
Alpha-beta is the most impactful algorithmic change possible for a minimax engine. In the best case (perfect move ordering) it reduces the tree from O(b^d) to O(b^(d/2)) — mathematically equivalent to doubling searchable depth for free. Even with average ordering it cuts the tree to roughly O(b^(3d/4)).

**The tradeoff:**
Alpha-beta always returns the exact same result as plain minimax — it is not an approximation. The only cost is a two-line comparison at every node. This is as close to a free lunch as algorithmic optimisation gets.

---

### 5. Move Ordering

**What it is:**
Alpha-beta only prunes when it finds a good move *early*. If moves are searched in random order, almost nothing gets pruned. Move ordering in `search.py → score_move()` sorts moves before searching using two heuristics:

- **MVV-LVA** (Most Valuable Victim, Least Valuable Attacker): captures are scored as `10000 + victim_value × 10 − attacker_value`. Capturing a queen with a pawn scores higher than capturing a pawn with a queen, ensuring the best captures are tried first.
- **TT move first**: the best move stored in the transposition table for this position (from a previous search) is placed at the front of the list, since it is very likely to cause an early cutoff again.

**Why we added it:**
Alpha-beta without move ordering barely prunes anything in practice. With good ordering the engine approaches the theoretical O(b^(d/2)) limit. This compounds with every other feature — better ordering means deeper effective search for the same time budget.

**The tradeoff:**
A sort at every node adds some overhead. But the additional pruning from better ordering saves far more time than the sort costs. Net result: ~40–50% faster search, ~100 ELO stronger.

---

### 6. Quiescence Search

**What it is:**
When the main search hits depth 0 and calls `evaluate()`, it may be looking at a position mid-capture — for example, a queen just took a pawn but the opponent can immediately recapture the queen next move. Evaluating here wildly overestimates the position. Quiescence search in `search.py → quiescence()` resolves this by continuing to search all captures (and en passant) beyond the depth limit until the position is "quiet" — no captures available. It also uses **delta pruning**: if `stand_pat + 900 < alpha`, no single capture can raise the score enough to matter, and the subtree is pruned immediately.

**Why we added it:**
Without quiescence search, the engine blunders constantly near its horizon — it sees material gains that evaporate one move later. This is the *horizon effect* and is one of the most damaging weaknesses of naive depth-limited search. Quiescence search essentially eliminates this class of blunder.

**The tradeoff:**
Unlike every other feature in this list, quiescence search makes the engine *slower* — it deliberately evaluates more nodes. At a tactically sharp position the quiescence tree can grow large. This is the only feature here that consciously trades speed for strength. The ~200 ELO gain makes it worth it by a significant margin.

---

### 7. Transposition Table (Zobrist Hashing)

**What it is:**
Chess positions can be reached by many different move orders. Without a transposition table, the engine re-searches the same position from scratch every time it arrives there via a different path — wasting enormous work.

The transposition table in `search.py` is a dictionary mapping position hash → `(depth, score, flag, best_move)`. Each position is identified by a **Zobrist hash** — a 64-bit integer built from `constants.py → ZOBRIST_TABLE`, a 13×8×8 array of random 64-bit numbers (one per piece type per square), seeded at `794613` for reproducibility. The hash is updated incrementally in `board.py → make_move()` and `undo_move()` by XORing in/out only the squares that changed, so recomputing from scratch is never needed.

The `flag` field distinguishes three cases needed for correct alpha-beta integration:
- `TT_EXACT` — the stored score is exact
- `TT_LOWER_BOUND` — the score caused a beta cutoff; real score is at least this high
- `TT_UPPER_BOUND` — no move improved alpha; real score is at most this high

The same Zobrist hash powers **threefold repetition detection**: `board.position_history` counts occurrences per hash throughout the game, and `minimax()` returns 0 (draw) when a position has been seen twice already on the current path.

**Why we added it:**
The transposition table gives the engine *memory* across search paths. It avoids redundant re-search, improves move ordering via the stored best move, and enables iterative deepening to compound across iterations. The table persists across the entire game, so the engine also benefits from positions encountered earlier in the game.

**The tradeoff:**
Memory proportional to unique positions encountered. A small lookup cost at every node. The savings from avoiding re-search dwarf the overhead — net result: ~30–50% faster search, ~150 ELO stronger.

---

### 8. Iterative Deepening

**What it is:**
Instead of searching directly to depth N, `find_best_move()` searches depth 1, then 2, then 3, up to the target depth. Each completed iteration's best move is preserved and inserted at the front of the move list for the next iteration:

```python
for depth in range(1, max_depth + 1):
    priority_move = best_move if best_move is not None else tt_move
    if priority_move in legal_moves:
        legal_moves.remove(priority_move)
        legal_moves.insert(0, priority_move)
```

The progress is visible in the terminal output:
```
  depth 1 -> score +30  best: Moved ♟ from e2 to e4
  depth 2 -> score +15  best: Moved ♞ from g1 to f3
  depth 3 -> score +25  best: ...
```

**Why we added it:**
Shallower searches are cheap (depth 1 ≈ 35 nodes; depth 4 ≈ 1.5M nodes; depth 5 ≈ 52M nodes) and they populate the transposition table with good move estimates that make the final search significantly more efficient. Without iterative deepening, the depth-5 search has no move ordering guidance from prior iterations. With it, the best move from the previous iteration bubbles to the front at every node, maximising alpha-beta cutoffs throughout the tree.

**The tradeoff:**
~33% extra work from re-searching shallower depths. In return, the deepest iteration runs faster due to better ordering, and a valid best move is always available if the search is cut short. The net ELO gain from better ordering at depth is real despite the overhead.

---

### 9. PyPy JIT Compiler

**What it is:**
PyPy is an alternative Python interpreter with a Just-In-Time compiler. Rather than interpreting bytecode line by line (CPython), PyPy profiles hot code paths and compiles them to native machine code at runtime. No code changes are required — the same `.py` files run on both interpreters.

The GUI bridges the two runtimes via `engine_main.py`, which implements a clean text protocol over stdin/stdout:

```
newgame           → ok
move e2e4         → ok | illegal
go <depth>        → bestmove e7e5
legal             → e2e4 d2d4 ... | none
status            → ongoing white | check black | checkmate white | stalemate | draw repetition
quit
```

`gui.py` spawns `engine_main.py` under PyPy and communicates through this protocol in a background thread, so the GUI stays responsive while the engine is thinking.

**Why we added it:**
The search consists almost entirely of tight Python loops — iterating over move lists, indexing into 8×8 arrays, XORing 64-bit integers. These are exactly the patterns JIT compilation excels at. The speedup is **4–7×** with zero code changes — equivalent to gaining 1–2 extra depth levels for free, which compounds through every other feature.

**The tradeoff:**
PyPy doesn't support C-extension packages (notably Pygame), which necessitates the subprocess split. PyPy also has a longer startup time and a brief JIT warmup on the first few searches. For sustained play, the warmup cost is negligible.

---

### 10. Texel Tuning (PST Optimisation)

**What it is:**
Texel tuning is a machine-learning method for optimising piece-square table values using real game data. Rather than setting PST values by hand, a PyTorch model treats every entry in every table as a learnable parameter and adjusts them to minimise prediction error across hundreds of thousands of evaluated positions.

The model is intentionally simple — it is exactly the engine's `evaluate()` function expressed as a linear operation:

```
score = feature_vector @ weights
```

where `feature_vector` encodes which pieces are on which squares and `weights` are the 453 PST/piece-value parameters (7 tables × 64 squares + 5 piece values). Because the model is linear and matches the engine's eval exactly, PyTorch's gradients are exact — there is no approximation.

The loss function maps centipawn scores into win-probability space via a sigmoid before comparing, which prevents large-score outliers from dominating training:

```python
pred_p   = torch.sigmoid(K * pred)
target_p = torch.sigmoid(K * target)
loss     = ((pred_p - target_p) ** 2).mean()
```

L2 regularisation is applied to PST weights only (not piece values) to prevent table entries from growing to unrealistic magnitudes.

**Data sources:**
Two separate datasets were generated and trained independently:

- **Self-play** (`texel_generate.py`): Stockfish plays randomised games at depth 6, collecting ~200k quiet positions evaluated at depth 10.
- **Lichess** (`texel_generate_lichess.py`): Streams a monthly Lichess PGN dump, filters to games above 1800 ELO, extracts quiet positions at regular intervals, and evaluates them with Stockfish.

**Blending:**
Rather than using one dataset's output directly, `constants.py` defines all three source tables explicitly — `HANDCRAFT_*`, `SELFPLAY_*`, `LICHESS_*` — and computes the final tables as a weighted blend at import time:

```python
PAWN_TABLE = _blend(HANDCRAFT_PAWN_TABLE, SELFPLAY_PAWN_TABLE, LICHESS_PAWN_TABLE,
                    w_hand=0.25, w_self=0.375, w_lich=0.375)
```

Blend weights were chosen per-table based on the quality of each source. For example, both tuned rook tables showed the Lichess version correctly capturing 7th-rank and open-file bonuses while the self-play version was almost entirely negative (an overfitting artifact), so Lichess is weighted more heavily there. The king tables are hand-crafted only — king safety depends on pawn shelter and open files in a way that a context-free PST cannot encode, so the tuned versions were discarded.

**Why we added it:**
Hand-crafted PST values encode human intuition but miss subtleties that only emerge from large amounts of real game data. Texel tuning is a principled way to let the data refine the tables while keeping the evaluation function completely unchanged — no new features, no slower evaluation, just better numbers.

**The tradeoff:**
The tuning pipeline requires Stockfish, ~200k positions, and several hours of compute per dataset. The evaluation function itself is identical — there is no runtime cost at all. The blend approach also means the hand-crafted values act as an anchor, preventing the tuned data from introducing noise into tables where it performed poorly (notably king safety).

**Running the pipeline:**

```bash
# Install dependencies
pip install torch chess zstandard numpy

# Generate self-play data (~2–3 hours)
python texel_generate.py --stockfish "path/to/stockfish" --positions 200000

# Or generate from a Lichess PGN dump (~2–3 hours)
python texel_generate_lichess.py --pgn lichess_db_standard_rated_2017-02.pgn.zst \
    --stockfish "path/to/stockfish" --positions 200000 --min-elo 1800

# Train
python texel_train.py --data texel_data.csv --epochs 2000 --lr 1.0 --l2 0.001
```

Lichess PGN dumps are available at [database.lichess.org](https://database.lichess.org). Stockfish binaries are available at [stockfishchess.org/download](https://stockfishchess.org/download/).

---

## Dynamic Depth Scaling

As pieces come off the board the search space shrinks, so `main.py` automatically increases depth in the endgame:

```python
if total_pieces <= 4:    depth = DEPTH + 5
elif total_pieces <= 6:  depth = DEPTH + 4
elif total_pieces <= 8:  depth = DEPTH + 3
elif total_pieces <= 12: depth = DEPTH + 2
elif total_pieces <= 20: depth = DEPTH + 1
else:                    depth = DEPTH
```

With `DEPTH = 4`, this gives depth 9 in a K+P vs K endgame — enough to convert most theoretically won endings correctly.

---

## Speed vs. Strength Summary

| Feature | Speed Impact | ELO Added | Running Total |
|---|---|---|---|
| Depth-limited minimax (depth 5) | Baseline | ~600 | ~600 |
| Move integer encoding | Negligible | — | ~600 |
| Piece-square tables | ~1% slower | ~50 | ~650 |
| Alpha-beta pruning | **~75% faster** | ~150 | ~800 |
| Move ordering | **~45% faster** | ~100 | ~900 |
| Quiescence search | ~30% slower | **~200** | ~1100 |
| Transposition table | **~40% faster** | ~150 | ~1250 |
| Iterative deepening | ~10% slower | ~75 | ~1325 |
| PyPy JIT | **~5× faster** | ~150 | ~1475 |
| Texel tuning | No runtime cost | ~50–100 | ~1550 |

---

## Chess Rules Implemented

- All piece movements (pawns, knights, bishops, rooks, queens, kings)
- Pawn double push and en passant capture
- Promotions to queen, rook, bishop, or knight
- Kingside and queenside castling with per-side rights tracking
- Check detection used to filter pseudo-legal moves
- Checkmate and stalemate detection
- Threefold repetition draw detection via Zobrist position history

---

## Credits

Piece images from [Wikimedia Commons](https://commons.wikimedia.org/wiki/Category:SVG_chess_pieces) — Colin M.L. Burnett, CC BY-SA 3.0. Downloaded automatically by `download_pieces.py`.
