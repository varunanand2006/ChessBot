#include "perft.hpp"

#include <cstdio>

#include "move.hpp"
#include "movegen.hpp"

namespace perft {

uint64_t perft(Position& pos, int depth) {
    MoveList list;
    movegen::generate_legal(pos, list);

    // Bulk counting: at the last ply the number of leaves is exactly the number
    // of legal moves, so we skip make/unmake for the whole final level. This is
    // a standard perft optimization and yields the same counts as the published
    // reference (which counts nodes at the given depth), for a large speedup.
    if (depth <= 1) return static_cast<uint64_t>(list.size());

    uint64_t nodes = 0;
    for (const Move m : list) {
        StateInfo st;
        make_move(pos, m, st);
        nodes += perft(pos, depth - 1);
        unmake_move(pos, m, st);
    }
    return nodes;
}

uint64_t divide(Position& pos, int depth) {
    MoveList list;
    movegen::generate_legal(pos, list);

    uint64_t total = 0;
    for (const Move m : list) {
        StateInfo st;
        make_move(pos, m, st);
        const uint64_t n = (depth <= 1) ? 1 : perft(pos, depth - 1);
        unmake_move(pos, m, st);
        std::printf("%s: %llu\n", move_to_uci(m).c_str(), static_cast<unsigned long long>(n));
        total += n;
    }
    std::printf("\nNodes: %llu\n", static_cast<unsigned long long>(total));
    return total;
}

}  // namespace perft
