// Sliding-piece attacks: rook, bishop, queen.
//
// Two implementations live here on purpose:
//
//   * rook_ref / bishop_ref  — the slow, obviously-correct ray-walk. It walks
//     each direction square by square and stops at the first blocker. This is
//     the permanent correctness oracle: it is used to BUILD the magic tables
//     at init, and the test suite checks the magics against it forever. It is
//     never deleted, even once magics work — a wrong magic fails silently and
//     surfaces as a bogus bug far downstream, so the reference stays.
//
//   * rook_attacks / bishop_attacks / queen_attacks — the fast magic-bitboard
//     path used everywhere in movegen/search. O(1): one masked multiply, one
//     shift, one table load.
//
// Why magic bitboards and not BMI2 PEXT (which is simpler and this CPU has):
// CUDA devices have no PEXT instruction, and the later retrograde tablebase
// phase needs sliding attacks in device code. Plain multiply-shift magics port
// to the GPU verbatim; PEXT would force a second, divergent implementation.
//
// init() MUST be called once at program start before any *_attacks query.

#pragma once

#include "slider_device.hpp"
#include "types.hpp"

namespace slider {

// Build the magic tables. Idempotent; safe to call more than once.
void init();

// Fill `out` with pointers to the internal magic entries + attack tables, for
// uploading to the GPU (see slider_device.hpp). init() must have run first. The
// pointers alias slider.cpp's static storage (stable for the process lifetime),
// so the caller copies from them to device memory and rebinds.
void get_device_sliders(DeviceSliders& out);

// Reference ray-walking attacks (the oracle). sq is a 0..63 square index.
// `occ` is the full board occupancy; blockers stop the ray (inclusive).
Bitboard rook_ref(int sq, Bitboard occ);
Bitboard bishop_ref(int sq, Bitboard occ);

// Fast magic-bitboard attacks. Requires init() to have run.
Bitboard rook_attacks(Square s, Bitboard occ);
Bitboard bishop_attacks(Square s, Bitboard occ);
Bitboard queen_attacks(Square s, Bitboard occ);

}  // namespace slider
