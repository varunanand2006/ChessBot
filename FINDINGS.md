# Numbers and findings

The performance numbers this project produced, and some endgame results that came
out of the GPU-generated 5-piece tablebases. Every tablebase figure below was
cross-checked against the Lichess (Gaviota DTM) tablebase; see
[`cuda/PROFILING.md`](cuda/PROFILING.md) for the full record.

## Performance numbers

| | |
|---|---|
| Perft (correctness) | startpos→6 = 119,060,324 nodes, Kiwipete→5 = 193,690,690, exact vs the Chess Programming Wiki |
| Move generation | ~21 M nodes/s (perft, single-threaded, `-O3 -march=native`) |
| Search | ~3.5 M nodes/s alpha-beta (flat-array TT, quiescence) |
| GPU kernel speedup | 17.2× naive→optimized (9,872 ms → 575 ms on KQKR), each step re-gated bit-exact |
| GPU vs CPU | ~1,096× faster than the same memoryless sweep single-threaded on CPU |
| GPU throughput | 342.8 M positions/s at 5-man scale |
| 5-piece tablebases | all 28 distinct-piece pawnless materials solved on an RTX 4090 |
| Scale | 5.87 billion positions indexed, 4.33 billion legal, ~313 billion GPU thread-executions |
| External verification | 28/28 materials, 896/896 sample positions match Lichess, zero mismatches |
| TB-probing in search | node reductions of ~10,900× (KRK d12), ~19,900× (KQK d12), ~18,500× (KQKR d10) vs the heuristic |

## Endgame findings (from the 28 five-piece tablebases)

The tablebases store perfect play — exact distance-to-mate for every legal
position. Ranking the 28 materials by how often the game is a draw (a metric that
doesn't depend on whose move it is) surfaces some real chess truths.

### 1. An extra minor piece usually isn't enough to win

The most-drawn materials are all "one side up a single minor piece," and the
defender holds a fortress ~80% of the time:

| material | draws | what it confirms |
|---|---:|---|
| KBN vs KB | 87.1% | bishop+knight vs bishop — almost always a draw |
| KBN vs KR | 84.2% | two minors don't beat a rook |
| KRN vs KR | 79.1% | rook+knight vs rook — the classic drawn fortress |
| KRB vs KR | 77.6% | rook+bishop vs rook — drawn (second-rank / Cochrane defense) |

These are exact fractions of drawn positions over every legal configuration
(156–175 million each), not estimates. R+minor vs R and two-minors vs one being
usually drawn is textbook endgame theory, confirmed here empirically to the
position.

### 2. The deepest mate sits inside a mostly-drawn ending

KBN vs KN is 82.9% draws, yet its won positions include a forced mate-in-107
(213 plies) — the deepest in the set and one of the deepest pawnless 5-man mates
known. The longest forced win in all 28 tables lives in a material that is usually
drawn: the win is rare, and converting it takes 107 moves of exact play. Matched
to Gaviota exactly.

### 3. More material can mean a longer mate

Three of the four "vs a lone king" materials mate almost immediately, but one is
very different:

| material | mate-in | note |
|---|---:|---|
| KQRB vs K | 5 | queen present → trivial |
| KQRN vs K | 5 | queen present → trivial |
| KQBN vs K | 7 | queen present → trivial |
| KRBN vs K | 29 | no queen → rook+bishop+knight must coordinate, ~6× deeper |

Rook + bishop + knight vs a bare king is more total material than queen + two
pieces, yet its worst-case mate is ~6× longer, because no single piece forces the
mate and the extra pieces make stalemate a constant hazard.

### 4. Queen + minor vs queen is nearly a coin flip

| material | draws | decisive |
|---|---:|---:|
| KQN vs KQ | 56.2% | 43.8% |
| KQB vs KQ | 53.8% | 46.2% |

Being up a bishop or knight in a queen ending wins only about half the time — the
defending queen generates enough perpetual-check and fortress resources to hold
the other half.

### 5. Rook + minor vs queen: the queen usually wins, and the wins are long

| material | decisive | deepest mate |
|---|---:|---:|
| KRN vs KQ | 69.1% | mate-in-69 |
| KRB vs KQ | 63.0% | mate-in-70 |

Queen vs rook+minor is winning for the queen in ~two-thirds of positions, and the
wins are among the deepest in the set (mate-in-69/70) — long endgames rather than
quick conversions.

## Full table — all 28 materials

Sorted by draw fraction. "decisive" = won or lost for one side; mate-in = deepest
forced mate in the material. Every row verified against Lichess (32 samples,
including each table's two deepest mates).

| material | draws | decisive | mate-in | legal positions |
|---|---:|---:|---:|---:|
| KBN vs KB | 87.1% | 12.9% | 39 | 174,949,839 |
| KBN vs KR | 84.2% | 15.8% | 41 | 167,207,085 |
| KBN vs KN | 82.9% | 17.1% | 107 | 179,977,181 |
| KRN vs KR | 79.1% | 20.9% | 41 | 160,061,214 |
| KRB vs KR | 77.6% | 22.4% | 65 | 156,171,346 |
| KQN vs KQ | 56.2% | 43.8% | 41 | 133,981,893 |
| KQB vs KQ | 53.8% | 46.2% | 33 | 130,009,252 |
| KRB vs KQ | 37.0% | 63.0% | 70 | 142,499,084 |
| KRN vs KQ | 30.9% | 69.1% | 69 | 146,388,952 |
| KRN vs KB | 18.6% | 81.4% | 31 | 167,803,968 |
| KRB vs KB | 18.1% | 81.9% | 30 | 163,914,100 |
| KQB vs KR | 16.2% | 83.8% | 40 | 143,681,514 |
| KQN vs KR | 15.7% | 84.3% | 41 | 147,654,155 |
| KRN vs KN | 14.2% | 85.8% | 37 | 172,831,310 |
| KRB vs KN | 14.1% | 85.9% | 40 | 168,941,442 |
| KQR vs KQ | 13.1% | 86.9% | 67 | 124,149,900 |
| KQN vs KB | 12.4% | 87.6% | 17 | 155,396,909 |
| KQB vs KB | 12.0% | 88.0% | 17 | 151,424,268 |
| KQN vs KN | 10.9% | 89.1% | 21 | 160,424,251 |
| KQR vs KR | 10.7% | 89.3% | 34 | 137,822,162 |
| KQB vs KN | 10.5% | 89.5% | 21 | 156,451,610 |
| KQR vs KB | 7.3% | 92.7% | 29 | 145,564,916 |
| KQR vs KN | 5.0% | 95.0% | 40 | 150,592,258 |
| KBN vs KQ | 4.3% | 95.7% | 53 | 153,534,823 |
| KQRB vs K | 1.2% | 98.8% | 5 | 152,000,134 |
| KQRN vs K | 1.0% | 99.0% | 5 | 155,000,267 |
| KQBN vs K | 0.7% | 99.3% | 7 | 160,420,548 |
| KRBN vs K | 0.3% | 99.7% | 29 | 171,777,043 |

> Win/loss counts are per side-to-move (both sides to move are distinct
> positions), so the draw fraction is the cleanest cross-material signal — it
> doesn't depend on whose turn it is. Raw per-material W/L/D are in the batch log.
