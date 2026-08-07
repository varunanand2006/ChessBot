// Perft: recursive leaf-node counting, the correctness gate for move
// generation. perft(N) is the number of legal move sequences of length N from
// a position; matching the published reference counts exercises every corner
// of make/unmake and legality filtering at once.

#pragma once

#include <cstdint>

#include "position.hpp"

namespace perft {

// Count leaf nodes at the given depth. Mutates pos via make/unmake but leaves
// it byte-identical on return.
uint64_t perft(Position& pos, int depth);

// Divide: print each root move's UCI string and its perft(depth-1) subtotal,
// then the total. This is THE debugging tool — compare subtotals against a
// reference and recurse into whichever root move differs. Returns the total.
uint64_t divide(Position& pos, int depth);

}  // namespace perft
