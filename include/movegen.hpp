// Move generation: pseudo-legal generation plus a make/unmake legality filter.
//
// Correctness-first design (per the phase plan): generate pseudo-legal moves,
// then keep only those that don't leave our own king in check by actually
// making the move and testing. This filter handles pins and the awkward
// en-passant discovered-check case for free — after the move is made, the
// board reflects reality and is_attacked() sees any newly-exposed slider.
// Pin-aware generation (skipping the make/unmake) is a later optimization; it
// must reproduce these results exactly, which Step 8's perft will enforce.

#pragma once

#include "move.hpp"
#include "position.hpp"
#include "types.hpp"

namespace movegen {

// Is square `s` attacked by any piece of color `by`, given current occupancy?
bool is_attacked(const Position& pos, Square s, Color by);

// Square of `c`'s king (assumes exactly one king of that color).
Square king_square(const Position& pos, Color c);

// Is the side to move currently in check?
bool in_check(const Position& pos);

// Pseudo-legal moves (may leave own king in check). const — does not touch pos.
void generate_pseudo_legal(const Position& pos, MoveList& list);

// Fully legal moves. Takes Position by non-const reference because it filters
// via make/unmake in place (pos is byte-identical on return). This is the
// allocation-free path perft/search will use.
void generate_legal(Position& pos, MoveList& list);

}  // namespace movegen
