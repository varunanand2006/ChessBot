// Step 3 gate: precomputed knight/king/pawn attack tables.
//   knight from a1 = 2 squares, from d4 = 8
//   king   from a1 = 3 squares, from d4 = 8
//
// Also checks pawn attacks and a-/h-file edge behavior (no wraparound), and
// asserts at compile time that the tables are genuinely constexpr.

#include <bit>
#include <cstdio>
#include <string_view>

#include "attacks.hpp"
#include "types.hpp"

namespace {

int g_failures = 0;

void check_count(std::string_view name, Bitboard bb, int expected) {
    const int got = std::popcount(bb);
    if (got != expected) {
        std::printf("[FAIL] %-28s expected %d, got %d\n", name.data(), expected, got);
        ++g_failures;
    } else {
        std::printf("[ OK ] %-28s = %d\n", name.data(), got);
    }
}

void check_true(std::string_view name, bool cond) {
    if (!cond) { std::printf("[FAIL] %s\n", name.data()); ++g_failures; }
    else       { std::printf("[ OK ] %s\n", name.data()); }
}

}  // namespace

// Compile-time proof the tables exist at compile time (Step 3's whole point).
static_assert(std::popcount(attacks::knight(Square::A1)) == 2);
static_assert(std::popcount(attacks::knight(Square::D4)) == 8);
static_assert(std::popcount(attacks::king(Square::A1)) == 3);
static_assert(std::popcount(attacks::king(Square::D4)) == 8);

int main() {
    // --- Gate: knight ---
    check_count("knight a1", attacks::knight(Square::A1), 2);
    check_count("knight d4", attacks::knight(Square::D4), 8);

    // --- Gate: king ---
    check_count("king a1", attacks::king(Square::A1), 3);
    check_count("king d4", attacks::king(Square::D4), 8);

    // --- Extra corners/edges to catch wraparound bugs ---
    check_count("knight h8", attacks::knight(Square::H8), 2);
    check_count("knight b1", attacks::knight(Square::B1), 3);
    check_count("king h8",   attacks::king(Square::H8), 3);
    check_count("king e1",   attacks::king(Square::E1), 5);

    // Exact target sets: knight a1 attacks exactly b3 and c2.
    check_true("knight a1 -> {b3,c2}",
               attacks::knight(Square::A1) ==
                   (square_bb(Square::B3) | square_bb(Square::C2)));

    // --- Pawn attacks ---
    // White pawn on e4 attacks d5 and f5.
    check_true("white pawn e4 -> {d5,f5}",
               attacks::pawn(Color::White, Square::E4) ==
                   (square_bb(Square::D5) | square_bb(Square::F5)));
    // Black pawn on e5 attacks d4 and f4.
    check_true("black pawn e5 -> {d4,f4}",
               attacks::pawn(Color::Black, Square::E5) ==
                   (square_bb(Square::D4) | square_bb(Square::F4)));
    // Edge: white pawn on a2 attacks only b3 (no wrap to h-file).
    check_count("white pawn a2 (edge)", attacks::pawn(Color::White, Square::A2), 1);
    check_true("white pawn a2 -> b3",
               attacks::pawn(Color::White, Square::A2) == square_bb(Square::B3));
    // White pawn on the 8th rank has no forward attacks.
    check_count("white pawn a8 (no forward)", attacks::pawn(Color::White, Square::A8), 0);

    if (g_failures == 0) {
        std::printf("\nAll attack-table tests passed.\n");
        return 0;
    }
    std::printf("\n%d attack-table test(s) failed.\n", g_failures);
    return 1;
}
