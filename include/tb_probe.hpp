// Tablebase probing for the search (Phase 4 hookup).
//
// At a search leaf whose material is small enough to have a generated table,
// the exact endgame result replaces the heuristic evaluation. This module is
// the adapter: given a Position, decide whether it is a supported material and,
// if so, return the exact win/draw/loss + distance-to-mate.
//
// Result is SIDE-TO-MOVE relative (like the tablebase itself). The search
// converts it to its own White-relative mate-score scale — this module stays
// independent of the search's scoring constants.
//
// Supported materials mirror tb::Index: pawnless, <= 4 men, distinct (color,
// type) extras. Anything else (pawns, 5+ men, identical pieces) returns
// found=false and the search falls back to the heuristic eval.
//
// Tables are built lazily and cached by material on first probe. Building a
// 4-man table (e.g. KQKR ~2.5M positions) takes seconds; every probe after is
// an O(1) index + array lookup. Probing is OFF in the search by default — see
// search::set_use_tablebase — so ordinary games/tests never pay the build cost.

#pragma once

#include <cstdint>

#include "position.hpp"

namespace tb {

struct ProbeResult {
    bool found = false;  // material supported AND position present in a table
    int  wdl   = 0;      // side-to-move relative: +1 win, 0 draw, -1 loss
    int  dtm   = 0;      // plies to mate (deliver if win / be mated if loss); 0 if draw
};

// Probe `pos`. Builds+caches the material's table on first use.
ProbeResult probe(const Position& pos);

// Number of distinct material tables built so far (for benchmarking/warmup).
uint64_t tables_built();

// Drop all cached tables (free memory). Cached tables otherwise persist for the
// life of the process, reused across games.
void clear();

}  // namespace tb
