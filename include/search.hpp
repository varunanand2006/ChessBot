// Alpha-beta search — a port of python/search.py, same shape and heuristics:
//   * negamax written as explicit White(max)/Black(min) branches over a
//     White-relative evaluation (see eval.hpp),
//   * alpha-beta pruning with a transposition table (exact/lower/upper bounds),
//   * quiescence search over captures with side-aware stand-pat + delta pruning,
//   * MVV-LVA + TT/prev-best move ordering,
//   * iterative deepening at the root,
//   * threefold-repetition detection via a game+search position history.
//
// The TT and the repetition history are game-global state (mirroring the Python
// module-level dicts): they persist across successive find_best_move calls in a
// game and are only reset by new_game(). A std::unordered_map keyed by the
// Zobrist hash is the faithful port of the Python dict; a fixed-size array TT is
// the standard optimization to make later if search throughput becomes a focus.

#pragma once

#include <cstdint>

#include "move.hpp"
#include "position.hpp"

namespace search {

struct SearchResult {
    Move     best_move = MOVE_NONE;
    int      score     = 0;  // White's perspective (centipawns / mate score)
    int      depth     = 0;  // last depth completed
    uint64_t nodes     = 0;  // nodes visited (search + quiescence)
};

// Reset all game-global state: transposition table + repetition history.
void new_game();

// Enable/disable tablebase probing at search nodes (default: off). When on, a
// node whose material is in a generated endgame table returns the exact result
// instead of searching further — an exact cutoff that prunes the tree. Off by
// default so ordinary searches never pay the table build cost; the endgame
// benchmark and its test turn it on explicitly.
void set_use_tablebase(bool on);

// Record a position actually reached in the game (call once per real move,
// including the initial position) so the search can see threefold repetitions.
void history_add(uint64_t key);

// Iterative deepening to max_depth. `verbose` prints a per-depth line like the
// Python root. The position is returned byte-identical (search unmakes all moves).
SearchResult find_best_move(Position& pos, int max_depth, bool verbose = false);

}  // namespace search
