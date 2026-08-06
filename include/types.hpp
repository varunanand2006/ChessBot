// Core value types for the engine: bitboards, colors, piece types, squares.
//
// Everything here is flat and trivially copyable — no methods with state, no
// virtuals, no hierarchies. These types are shared verbatim with device code
// in the later CUDA phase, so they must stay POD-friendly and header-only.
//
// Square mapping: Little-Endian Rank-File (LERF).
//   index = rank * 8 + file,  file 0 = a-file, rank 0 = rank 1.
//   a1 = 0, h1 = 7, a8 = 56, h8 = 63.
// This is the standard bitboard convention: a "west" shift is >>1, "north" is
// <<8, so sliding/pawn move generation later reads like the textbook.

#pragma once

#include <cstdint>

using Bitboard = uint64_t;

// ---------------------------------------------------------------------------
// Color
// ---------------------------------------------------------------------------
enum class Color : uint8_t {
    White = 0,
    Black = 1,
    None  = 2,
};

constexpr Color operator~(Color c) {
    // Flip side to move. Valid only for White/Black.
    return static_cast<Color>(static_cast<uint8_t>(c) ^ 1u);
}

constexpr int color_index(Color c) { return static_cast<int>(c); }

// ---------------------------------------------------------------------------
// Piece type (color-agnostic). Ordering matches conventional piece values and
// is used directly as an index into Position::by_type[6].
// ---------------------------------------------------------------------------
enum class PieceType : uint8_t {
    Pawn   = 0,
    Knight = 1,
    Bishop = 2,
    Rook   = 3,
    Queen  = 4,
    King   = 5,
    None   = 6,
};

constexpr int NUM_PIECE_TYPES = 6;
constexpr int type_index(PieceType pt) { return static_cast<int>(pt); }

// ---------------------------------------------------------------------------
// Square (LERF). 64 real squares plus a None sentinel (e.g. no en-passant).
// ---------------------------------------------------------------------------
enum class Square : uint8_t {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
    None,
};

constexpr int NUM_SQUARES = 64;

constexpr int sq_index(Square s) { return static_cast<int>(s); }

constexpr Square make_square(int file, int rank) {
    return static_cast<Square>(rank * 8 + file);
}

constexpr int file_of(Square s) { return sq_index(s) & 7; }
constexpr int rank_of(Square s) { return sq_index(s) >> 3; }

// Single-bit bitboard for a square.
constexpr Bitboard square_bb(Square s) {
    return Bitboard{1} << sq_index(s);
}

// ---------------------------------------------------------------------------
// Castling rights: a 4-bit mask. Combinable with bitwise OR.
// ---------------------------------------------------------------------------
enum CastlingRights : uint8_t {
    NO_CASTLING = 0,
    WHITE_OO    = 1,  // white kingside  (e1-g1)
    WHITE_OOO   = 2,  // white queenside (e1-c1)
    BLACK_OO    = 4,  // black kingside  (e8-g8)
    BLACK_OOO   = 8,  // black queenside (e8-c8)
    ANY_CASTLING = WHITE_OO | WHITE_OOO | BLACK_OO | BLACK_OOO,
};
