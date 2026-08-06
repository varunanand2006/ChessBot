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

inline bool is_win(int16_t v)  { return v > 0; }
inline bool is_loss(int16_t v) { return v < 0; }
inline int  win_dtm(int16_t v)  { return MATE - v; }   // valid when v > 0
inline int  loss_dtm(int16_t v) { return MATE + v; }   // valid when v < 0

Table solve_sweep(const Index& idx);
Table solve_bfs(const Index& idx);

}  // namespace tb
