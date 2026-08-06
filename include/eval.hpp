// Static evaluation — a direct port of the Python engine's handcrafted +
// Texel-blended evaluation (python/search.py `evaluate`, python/constants.py
// piece values & piece-square tables).
//
// Convention (same as the Python engine and the search that consumes it):
// the score is always from WHITE's perspective — positive favours White,
// negative favours Black — NOT side-to-move relative. The search's minimax
// keeps explicit White(max)/Black(min) branches to match.
//
// Terms: material + piece-square tables + a small in-check penalty + an
// endgame king-distance term (drive the losing king toward a corner). The
// blended values live in src/eval_tables.inc, generated from the Python
// constants by python/gen_eval_tables.py so they match byte-for-byte.

#pragma once

#include "position.hpp"

namespace eval {

// Centipawns, from White's perspective.
int evaluate(const Position& pos);

// Blended value of a piece type (centipawns; King/None -> 0). Exposed for move
// ordering (MVV-LVA) so it shares the single generated value table.
int piece_value(PieceType pt);

}  // namespace eval
