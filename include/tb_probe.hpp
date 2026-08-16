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

#include <string>

#include "position.hpp"

namespace tb {

struct ProbeResult {
    bool found = false;  // material supported AND position present in a table
    int  wdl   = 0;      // side-to-move relative: +1 win, 0 draw, -1 loss
    int  dtm   = 0;      // plies to mate (deliver if win / be mated if loss); 0 if draw
};

// Directory searched for persisted `<MATERIAL>.tb` files (canonical name, e.g.
// "KQKR.tb"). Empty (the default) disables disk probing. When set, probe() uses
// an on-disk table if one exists for the position's material — this is how a
// GPU-generated 5-man table gets used without regenerating it. Loaded tables are
// cached; materials with no file are remembered so probing isn't retried.
void set_table_dir(const std::string& dir);

// Probe `pos`. Prefers an on-disk table (any persisted material, incl. 5-man),
// else builds+caches the <=4-man dense table in RAM. Cached tables persist for
// the life of the process, reused across games.
ProbeResult probe(const Position& pos);

}  // namespace tb
