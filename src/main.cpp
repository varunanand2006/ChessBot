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

}  // namespace

int main(int argc, char** argv) {
    slider::init();  // build magic tables once at startup

    if (argc >= 3 && std::strcmp(argv[1], "perft") == 0)  return run_perft(argc, argv, false);
    if (argc >= 3 && std::strcmp(argv[1], "divide") == 0) return run_perft(argc, argv, true);

    std::printf("ChessEngine v0.1.0\n");
    std::printf("usage:\n");
    std::printf("  chess perft  <depth> [FEN...]\n");
    std::printf("  chess divide <depth> [FEN...]\n");
    return 0;
}
