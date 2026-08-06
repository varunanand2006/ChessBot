// Step 7 gate: legal move generation compiles and produces moves. Full
// correctness is verified by perft in Step 8; here we only sanity-check that
// generation runs, restores the board, and gets the obvious cases right.

#include <cstdio>
#include <string_view>

#include "movegen.hpp"
#include "position.hpp"
#include "slider.hpp"

namespace {

int g_failures = 0;

void expect(std::string_view name, bool cond) {
    if (!cond) { std::printf("[FAIL] %s\n", name.data()); ++g_failures; }
    else       { std::printf("[ OK ] %s\n", name.data()); }
}

bool same(const Position& a, const Position& b) {
    for (int i = 0; i < NUM_PIECE_TYPES; ++i)
        if (a.by_type[i] != b.by_type[i]) return false;
    return a.by_color[0] == b.by_color[0] && a.by_color[1] == b.by_color[1] &&
           a.side_to_move == b.side_to_move && a.castling == b.castling &&
           a.ep_square == b.ep_square && a.zobrist == b.zobrist;
}

int legal_count(std::string_view fen) {
    Position pos;
    set_fen(pos, fen);
    const Position before = pos;
    MoveList list;
    movegen::generate_legal(pos, list);
    if (!same(pos, before)) { std::printf("[FAIL] board mutated by generate_legal: %s\n", fen.data()); ++g_failures; }
    // Every generated move must, when made, leave our king safe.
    const Color us = pos.side_to_move, them = ~us;
    for (Move m : list) {
        StateInfo st;
        make_move(pos, m, st);
        if (movegen::is_attacked(pos, movegen::king_square(pos, us), them)) {
            std::printf("[FAIL] illegal move survived filter: %s in %s\n",
                        move_to_uci(m).c_str(), fen.data());
            ++g_failures;
        }
        unmake_move(pos, m, st);
    }
    return list.size();
}

}  // namespace

int main() {
    slider::init();

    // Startpos has exactly 20 legal moves: 8 pawns x2 (single+double) + 2
    // knights x2. This is structurally certain, not a recalled perft value.
    const int n_start = legal_count("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    std::printf("startpos legal moves = %d\n", n_start);
    expect("startpos == 20", n_start == 20);

    // Produce moves for the other standard positions (counts printed for eyeball
    // review; Step 8 checks them against the wiki). Just require > 0 and a valid
    // board round-trip, which legal_count already asserts.
    struct Case { std::string_view name, fen; };
    const Case cases[] = {
        {"kiwipete",  "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"},
        {"position3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"},
        {"position4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"},
        {"position5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"},
    };
    for (const auto& c : cases) {
        const int n = legal_count(c.fen);
        std::printf("%-10s legal moves = %d\n", c.name.data(), n);
        expect(std::string_view(c.name), n > 0);
    }

    // Targeted special-move sanity.
    // A position where en passant is the ONLY pawn capture available.
    {
        Position pos; set_fen(pos, "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
        MoveList list; movegen::generate_legal(pos, list);
        bool has_ep = false;
        for (Move m : list) if (is_en_passant(m)) has_ep = true;
        expect("en passant generated", has_ep);
    }
    // Castling available for both sides on an open back rank.
    {
        Position pos; set_fen(pos, "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        MoveList list; movegen::generate_legal(pos, list);
        int castles = 0;
        for (Move m : list)
            if (move_flag(m) == MoveFlag::KingCastle || move_flag(m) == MoveFlag::QueenCastle) ++castles;
        expect("two castle moves for white", castles == 2);
    }
    // Cannot castle THROUGH check: a rook on e-file attacks... use f1 attacked.
    // Black rook on f8 attacks f1, so white O-O (king passes f1) is illegal,
    // but O-O-O is still legal.
    {
        Position pos; set_fen(pos, "4kr2/8/8/8/8/8/8/R3K2R w KQ - 0 1");
        MoveList list; movegen::generate_legal(pos, list);
        bool oo = false, ooo = false;
        for (Move m : list) {
            if (move_flag(m) == MoveFlag::KingCastle)  oo = true;
            if (move_flag(m) == MoveFlag::QueenCastle) ooo = true;
        }
        expect("O-O blocked through check", !oo);
        expect("O-O-O still legal", ooo);
    }
    // In-check detection: white king on e1, black rook on e8 -> in check.
    {
        Position pos; set_fen(pos, "4r3/8/8/8/8/8/8/4K3 w - - 0 1");
        expect("in_check true", movegen::in_check(pos));
    }

    if (g_failures == 0) { std::printf("\nAll movegen sanity tests passed.\n"); return 0; }
    std::printf("\n%d movegen sanity test(s) failed.\n", g_failures);
    return 1;
}
