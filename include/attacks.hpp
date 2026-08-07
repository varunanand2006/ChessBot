// Precomputed non-sliding attack tables: knight, king, pawn.
//
// This is the bitboard analog of the Python engine's per-move offset loops
// (KNIGHT_OFFSETS / KING_OFFSETS with a `0 <= rr < 8` bounds check). Here the
// same offset+bounds logic runs ONCE at compile time via constexpr, baking a
// ready-made attack bitboard for every square into the binary.
//
// Rationale: each table is 64 * 8 bytes = 512 B (pawn table 1 KiB for both
// colors) — trivial, and it lives hot in L1. In exchange, an attack query at
// runtime is a single array load with zero branches, versus the 8-iteration
// offset loop with a bounds test per step that the Python version pays on
// every move generated. The alternative (compute on the fly) costs those
// branches in the movegen hot loop for no memory saving worth mentioning.
//
// Header-only constexpr so the tables are compile-time constants the optimizer
// can fold through; this mirrors keeping the precomputed data in one place, as
// constants.py did in the Python project.

#pragma once

#include <array>

#include "types.hpp"

namespace attacks {

namespace detail {

// bit / on_board / *_from are CH_HD for two reasons: they build the host tables
// below at compile time, AND they are the DEVICE path of the attack accessors —
// nvcc cannot ODR-use the host KNIGHT/KING/PAWN arrays in device code, so on the
// GPU the accessors recompute through these (see the accessors' __CUDA_ARCH__
// split). The knight/king offset lists are FUNCTION-LOCAL constexpr arrays rather
// than namespace-scope ones for the same reason: a namespace-scope host array
// indexed at runtime in device code is exactly the ODR-use that fails to compile.
CH_HD constexpr Bitboard bit(int file, int rank) {
    return Bitboard{1} << (rank * 8 + file);
}

CH_HD constexpr bool on_board(int file, int rank) {
    return file >= 0 && file < 8 && rank >= 0 && rank < 8;
}

CH_HD constexpr Bitboard knight_from(int sq) {
    constexpr int DF[8] = {+1, +2, +2, +1, -1, -2, -2, -1};
    constexpr int DR[8] = {+2, +1, -1, -2, -2, -1, +1, +2};
    Bitboard bb = 0;
    const int f = sq & 7, r = sq >> 3;
    for (int i = 0; i < 8; ++i) {
        const int tf = f + DF[i], tr = r + DR[i];
        if (on_board(tf, tr)) bb |= bit(tf, tr);
    }
    return bb;
}

CH_HD constexpr Bitboard king_from(int sq) {
    constexpr int DF[8] = {-1, 0, +1, -1, +1, -1, 0, +1};
    constexpr int DR[8] = {+1, +1, +1, 0, 0, -1, -1, -1};
    Bitboard bb = 0;
    const int f = sq & 7, r = sq >> 3;
    for (int i = 0; i < 8; ++i) {
        const int tf = f + DF[i], tr = r + DR[i];
        if (on_board(tf, tr)) bb |= bit(tf, tr);
    }
    return bb;
}

// Pawn ATTACKS (captures only — pushes are not attacks and are generated
// separately). White attacks toward rank+1, Black toward rank-1, each by one
// file left and right. Offsets are a local array (not an initializer_list) so
// the loop is plainly device-compilable.
CH_HD constexpr Bitboard pawn_from(Color c, int sq) {
    const int dfs[2] = {-1, +1};
    Bitboard bb = 0;
    const int f = sq & 7, r = sq >> 3;
    const int dr = (c == Color::White) ? +1 : -1;
    for (int i = 0; i < 2; ++i) {
        const int tf = f + dfs[i], tr = r + dr;
        if (on_board(tf, tr)) bb |= bit(tf, tr);
    }
    return bb;
}

constexpr std::array<Bitboard, 64> make_knight() {
    std::array<Bitboard, 64> a{};
    for (int s = 0; s < 64; ++s) a[s] = knight_from(s);
    return a;
}

constexpr std::array<Bitboard, 64> make_king() {
    std::array<Bitboard, 64> a{};
    for (int s = 0; s < 64; ++s) a[s] = king_from(s);
    return a;
}

constexpr std::array<std::array<Bitboard, 64>, 2> make_pawn() {
    std::array<std::array<Bitboard, 64>, 2> a{};
    for (int s = 0; s < 64; ++s) {
        a[color_index(Color::White)][s] = pawn_from(Color::White, s);
        a[color_index(Color::Black)][s] = pawn_from(Color::Black, s);
    }
    return a;
}

}  // namespace detail

// The tables. Compile-time constants; every entry is a fully-formed attack set.
inline constexpr std::array<Bitboard, 64> KNIGHT = detail::make_knight();
inline constexpr std::array<Bitboard, 64> KING   = detail::make_king();
inline constexpr std::array<std::array<Bitboard, 64>, 2> PAWN = detail::make_pawn();

// Query accessors. Constexpr + Square-typed so callers stay readable and the
// index math is centralized here. HOST reads the precomputed table (one branch-
// free load in the movegen hot loop); DEVICE recomputes via the *_from helpers,
// because nvcc cannot ODR-use the host KNIGHT/KING/PAWN arrays in device code
// ("identifier attacks::KNIGHT is undefined in device code"). The recompute is a
// handful of ALU ops per query; the alternative — uploading these 512 B/1 KiB
// tables to __constant__ and binding them per device TU — is Phase-4 plumbing we
// don't need for correctness, and on the GPU the ALU path is competitive anyway.
CH_HD constexpr Bitboard knight(Square s) {
#ifdef __CUDA_ARCH__
    return detail::knight_from(sq_index(s));
#else
    return KNIGHT[sq_index(s)];
#endif
}
CH_HD constexpr Bitboard king(Square s) {
#ifdef __CUDA_ARCH__
    return detail::king_from(sq_index(s));
#else
    return KING[sq_index(s)];
#endif
}
CH_HD constexpr Bitboard pawn(Color c, Square s) {
#ifdef __CUDA_ARCH__
    return detail::pawn_from(c, sq_index(s));
#else
    return PAWN[color_index(c)][sq_index(s)];
#endif
}

}  // namespace attacks
