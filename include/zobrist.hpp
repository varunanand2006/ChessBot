// Zobrist hashing keys.
//
// A position's hash is the XOR of one random 64-bit key per (color, piece,
// square) present, plus keys for side-to-move, castling rights, and the
// en-passant file. Because XOR is its own inverse, the key updates
// incrementally in make/unmake by XORing only the squares that changed.
//
// Phase 1 has no transposition table yet, but the key is introduced now as a
// cheap correctness invariant: after any make(), the incrementally-maintained
// key must equal a from-scratch recomputation. A mismatch is an immediate,
// loud signal that make/unmake touched state inconsistently — far easier to
// catch here than as a subtle search bug later.
//
// Keys are generated at COMPILE TIME (constexpr splitmix64) from a fixed seed,
// so the table is reproducible build-to-build — the same reproducibility the
// Python engine got from seeding its RNG.

#pragma once

#include <cstdint>

#include "types.hpp"

namespace zobrist {
namespace detail {

// Same seed lineage as the Python engine's Zobrist table (794613), so hashes
// are comparable in spirit across the two implementations.
constexpr uint64_t SEED = 794613ULL;

// splitmix64 — a well-distributed constexpr-friendly PRNG step.
constexpr uint64_t next(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

struct Tables {
    uint64_t piece[2][NUM_PIECE_TYPES][64];
    uint64_t castling[16];  // indexed by the 4-bit castling-rights mask
    uint64_t ep_file[8];    // indexed by file of the en-passant target
    uint64_t side;          // XORed in when Black is to move
};

constexpr Tables make() {
    Tables t{};
    uint64_t s = SEED;
    for (int c = 0; c < 2; ++c)
        for (int pt = 0; pt < NUM_PIECE_TYPES; ++pt)
            for (int sq = 0; sq < 64; ++sq)
                t.piece[c][pt][sq] = next(s);
    for (int i = 0; i < 16; ++i) t.castling[i] = next(s);
    for (int i = 0; i < 8; ++i)  t.ep_file[i]  = next(s);
    t.side = next(s);
    return t;
}

}  // namespace detail

inline constexpr detail::Tables T = detail::make();

constexpr uint64_t piece(Color c, PieceType pt, Square s) {
    return T.piece[color_index(c)][type_index(pt)][sq_index(s)];
}
constexpr uint64_t castling(uint8_t rights) { return T.castling[rights]; }
constexpr uint64_t ep_file(int file)        { return T.ep_file[file]; }
constexpr uint64_t side()                   { return T.side; }

}  // namespace zobrist
