// Phase 2 / Step 2.2 — retrograde DTM solver for a 3-man endgame.
//
// Value encoding (mate-score, from the side-to-move's perspective), int16:
//     v == 0  : draw
//     v  > 0  : WIN  for side to move, mate in (MATE - v) plies
//     v  < 0  : LOSS for side to move, mated in (MATE + v) plies
//   so v = +(MATE-1) is mate-in-1, v = -MATE is "checkmated now" (loss in 0).
//
// Two solvers are provided and must agree exactly (the correctness gate):
//   * solve_sweep — iterated negamax fixpoint: each pass recomputes every
//     position's value as max over legal moves of "age(-child)". This is the
//     dense-sweep-to-convergence shape the CUDA phase implements on the GPU.
//   * solve_bfs   — textbook win/loss retrograde BFS by distance over the
//     REVERSE move graph. A structurally different algorithm, so agreement
//     between the two is strong evidence both are correct.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "tb_comb_index.hpp"
#include "tb_index.hpp"

namespace tb {

constexpr int MATE = 30000;

struct Table {
    std::vector<int16_t> value;  // index-aligned with the Index
    std::size_t wins = 0, losses = 0, draws = 0;
    int max_win_dtm = 0;   // longest forced mate (plies) the winning side delivers
    int max_loss_dtm = 0;  // longest defense (plies) before being mated
    int passes = 0;        // sweep iterations to converge (0 for bfs)
};

inline int win_dtm(int16_t v)  { return MATE - v; }   // valid when v > 0
inline int loss_dtm(int16_t v) { return MATE + v; }   // valid when v < 0

// Move a mate score one ply further from mate (toward zero). The retrograde
// step shared by every solver: a parent's value is the best age(-child) over
// its moves. Shared here so the dense and combinatorial solvers use one
// definition (and one mate-distance convention).
inline int age(int x) { return x > 0 ? x - 1 : (x < 0 ? x + 1 : 0); }

Table solve_sweep(const Index& idx);
Table solve_bfs(const Index& idx);

// Memoryless dense sweep over a COMBINATORIAL index — the GPU-faithful shape:
// O(N) memory (only the int16 value array), moves regenerated every pass, in
// place, iterated to a fixpoint. This is what makes 5-man feasible: the
// materialized forward graph the dense solvers build is billions of edges (tens
// of GB) at 5 men, whereas this stores nothing per edge. It is also the
// single-thread CPU baseline the CUDA sweep will be measured against.
//
// Distinct-piece materials only (capture-exit detection identifies the removed
// piece by which (color,type) bitboard emptied). Capture-exit boundary values
// come from the sub-materials, solved by the dense solve_sweep (they are 4-man
// or smaller, so materializing their graph is fine).
Table solve_sweep_comb(const CombIndex& idx);

}  // namespace tb
