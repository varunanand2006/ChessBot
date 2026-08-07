// Position: the full board state as a struct of bitboards.
//
// Data-oriented layout — no redundant mailbox array (that would be extra state
// to keep in sync across make/unmake and, later, across the CUDA retrograde
// kernels). Piece-on-square is derived from the bitboards on demand. If perft
// profiling later shows piece lookup is hot, a mailbox can be added then, with
// a measured justification.

#pragma once

#include <string>
#include <string_view>

#include "move.hpp"
#include "types.hpp"

struct Position {
    // Piece occupancy split two ways. To get e.g. white knights:
    //   by_type[Knight] & by_color[White]
    // occupied() is the union. Storing both axes (type and color) rather than
    // 12 per-(color,type) boards keeps the struct small (8 boards) while still
    // making the common queries single ANDs.
    Bitboard by_type[NUM_PIECE_TYPES] = {};  // indexed by PieceType 0..5
    Bitboard by_color[2]              = {};  // indexed by Color   0..1

    Color  side_to_move   = Color::White;
    uint8_t castling       = NO_CASTLING;    // CastlingRights mask
    Square ep_square      = Square::None;    // en-passant TARGET square, or None
    uint16_t halfmove_clock = 0;             // plies since last pawn move/capture
    uint16_t fullmove_number = 1;            // starts at 1, increments after Black
    uint64_t zobrist       = 0;              // incrementally-maintained hash key

    Bitboard occupied() const { return by_color[0] | by_color[1]; }

    // Piece/color on a square, or None if empty. Linear scan over 6 boards.
    PieceType piece_type_on(Square s) const;
    Color     color_on(Square s) const;

    // Remove all pieces / reset to an empty board (used by FEN parsing).
    void clear();
};

// FEN (Forsyth-Edwards Notation).
// set_fen returns false and leaves pos in an unspecified state on malformed
// input (no exceptions — hot-path-friendly and CUDA-portable error handling).
bool        set_fen(Position& pos, std::string_view fen);
std::string to_fen(const Position& pos);

// Recompute the Zobrist key from scratch. Used to seed a freshly-parsed
// position and, in tests, as the oracle the incremental key is checked against.
uint64_t compute_zobrist(const Position& pos);

// ---------------------------------------------------------------------------
// make / unmake
//
// Design choice: real make/unmake with an explicit state stack, NOT copy-board.
//   * No ~80-byte Position copy per node — make mutates in place, unmake
//     reverses it.
//   * The later CUDA retrograde tablebase phase is built on *unmove*
//     generation (retrograde analysis walks moves backward). Having an
//     explicit unmake primitive on the host is exactly what those device
//     kernels mirror; copy-board would leave nothing to carry forward.
//
// The caller owns the stack: it keeps a StateInfo per ply (e.g. a fixed
// StateInfo[MAX_PLY] array) and passes the matching slot to make/unmake. This
// keeps Position small and the stack allocation-free.
// ---------------------------------------------------------------------------
struct StateInfo {
    uint64_t  key;             // Zobrist key BEFORE the move (restored on unmake)
    uint16_t  halfmove_clock;  // halfmove clock before the move
    uint8_t   castling;        // castling rights before the move
    Square    ep_square;       // en-passant target before the move
    PieceType captured;        // piece captured by this move, or None
};

void make_move(Position& pos, Move m, StateInfo& st);
void unmake_move(Position& pos, Move m, const StateInfo& st);
