// Chess engine driver.
//
// Phase 1 exposes a perft CLI — the tool used to validate move generation:
//   chess perft  <depth> [FEN...]   leaf-node count at depth
//   chess divide <depth> [FEN...]   per-root-move subtotals (debugging)
// FEN may be given as the remaining args (unquoted, space-separated); if
// omitted, the standard start position is used.

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

#include "perft.hpp"
#include "position.hpp"
#include "slider.hpp"
#include "tb_index.hpp"
#include "tb_solve.hpp"
#include "types.hpp"

namespace {

constexpr const char* START_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// Join argv[first..argc) into a single space-separated FEN string.
std::string join_fen(int argc, char** argv, int first) {
    if (first >= argc) return START_FEN;
    std::string fen = argv[first];
    for (int i = first + 1; i < argc; ++i) {
        fen += ' ';
        fen += argv[i];
    }
    return fen;
}

int run_perft(int argc, char** argv, bool divide) {
    const int depth = std::atoi(argv[2]);
    const std::string fen = join_fen(argc, argv, 3);

    Position pos;
    if (!set_fen(pos, fen)) {
        std::fprintf(stderr, "invalid FEN: %s\n", fen.c_str());
        return 1;
    }

    const auto t0 = std::chrono::steady_clock::now();
    uint64_t nodes = divide ? perft::divide(pos, depth) : perft::perft(pos, depth);
    const auto t1 = std::chrono::steady_clock::now();

    const double secs = std::chrono::duration<double>(t1 - t0).count();
    if (!divide) std::printf("perft(%d) = %llu\n", depth, static_cast<unsigned long long>(nodes));
    const double nps = secs > 0 ? static_cast<double>(nodes) / secs : 0.0;
    std::printf("time %.3fs   %.2f Mnps\n", secs, nps / 1e6);
    return 0;
}

// Parse a single white-piece letter into a PieceType for the 3-man tablebase.
bool parse_piece(const char* s, PieceType& pt) {
    if (!s || s[0] == '\0' || s[1] != '\0') return false;
    switch (s[0]) {
        case 'Q': case 'q': pt = PieceType::Queen;  return true;
        case 'R': case 'r': pt = PieceType::Rook;   return true;
        case 'B': case 'b': pt = PieceType::Bishop; return true;
        case 'N': case 'n': pt = PieceType::Knight; return true;
        default: return false;
    }
}

int run_tb(int argc, char** argv) {
    PieceType wp;
    if (argc < 3 || !parse_piece(argv[2], wp)) {
        std::fprintf(stderr, "usage: chess tb <Q|R|B|N>\n");
        return 1;
    }

    tb::Index idx(wp);
    const tb::Table t = tb::solve_sweep(idx);

    // type_index order: Pawn0 Knight1 Bishop2 Rook3 Queen4 -> letter table below.
    std::printf("K%cK  indices=%zu  W=%zu L=%zu D=%zu  maxWinDTM=%d plies (%d moves)\n",
                " NBRQ"[type_index(wp)], idx.size(), t.wins, t.losses, t.draws,
                t.max_win_dtm, (t.max_win_dtm + 1) / 2);

    // Show a deepest-forced-mate position (the hardest win) as a concrete,
    // externally-verifiable FEN. Longest mate => smallest positive value.
    if (t.wins > 0) {
        std::size_t deepest = 0;
        for (std::size_t i = 0; i < t.value.size(); ++i)
            if (t.value[i] > 0 && tb::win_dtm(t.value[i]) == t.max_win_dtm) { deepest = i; break; }
        std::printf("deepest win (%d-ply mate): %s\n",
                    t.max_win_dtm, to_fen(idx.decode(deepest)).c_str());
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    slider::init();  // build magic tables once at startup

    if (argc >= 3 && std::strcmp(argv[1], "perft") == 0)  return run_perft(argc, argv, false);
    if (argc >= 3 && std::strcmp(argv[1], "divide") == 0) return run_perft(argc, argv, true);
    if (argc >= 2 && std::strcmp(argv[1], "tb") == 0)     return run_tb(argc, argv);

    std::printf("ChessEngine v0.1.0\n");
    std::printf("usage:\n");
    std::printf("  chess perft  <depth> [FEN...]\n");
    std::printf("  chess divide <depth> [FEN...]\n");
    std::printf("  chess tb     <Q|R|B|N>\n");
    return 0;
}
