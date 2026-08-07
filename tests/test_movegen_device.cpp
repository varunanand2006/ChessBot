// Phase 2 gate (host half) — the DEVICE-shaped move generator
// (movegen_dev::generate_legal over Position + slider::DeviceSliders, in
// movegen_device.hpp) produces the SAME legal move set as the authoritative
// movegen::generate_legal, at every node of full perft trees.
//
// movegen_dev is CH_HD, so this runs it on the HOST and diffs it against the
// reference over the standard perft positions (startpos, Kiwipete, positions
// 3/4/5) — which exercise castling, en passant, and promotions, the parts most
// likely to diverge — plus a pawnless endgame like the sweep actually uses. A
// full local Phase 2 gate with no GPU; cuda/movegen_check.cu confirms the device
// reproduces this host result.
//
// At each node: sort both legal move lists and require them identical. The tree
// is driven by the REFERENCE make/unmake so a bug in the device make/unmake
// can't hide the tree from the diff.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "movegen.hpp"
#include "movegen_device.hpp"
#include "position.hpp"
#include "slider.hpp"
#include "types.hpp"

namespace {

int g_failures = 0;
uint64_t g_nodes = 0;
void fail(const char* m) { std::printf("[FAIL] %s\n", m); ++g_failures; }

slider::DeviceSliders g_sliders;  // host-side; aliases slider.cpp tables.

bool same_moves(const MoveList& a, const MoveList& b) {
    if (a.size() != b.size()) return false;
    std::vector<Move> x(a.begin(), a.end()), y(b.begin(), b.end());
    std::sort(x.begin(), x.end());
    std::sort(y.begin(), y.end());
    return x == y;
}

// Walk with the reference movegen; at every node compare dev vs ref legal sets.
void walk(Position& pos, int depth) {
    if (g_failures) return;

    MoveList ref;
    movegen::generate_legal(pos, ref);

    MoveList dev;
    movegen_dev::generate_legal(pos, dev, g_sliders);

    ++g_nodes;
    if (!same_moves(ref, dev)) {
        std::printf("  divergence: ref=%d dev=%d moves at %s\n",
                    ref.size(), dev.size(), to_fen(pos).c_str());
        fail("device legal set != reference");
        return;
    }
    if (depth <= 1) return;

    for (const Move m : ref) {
        StateInfo st;
        make_move(pos, m, st);
        walk(pos, depth - 1);
        unmake_move(pos, m, st);
        if (g_failures) return;
    }
}

void run(const char* fen, int depth, const char* name) {
    Position pos;
    if (!set_fen(pos, fen)) { fail("bad FEN"); return; }
    const uint64_t before = g_nodes;
    walk(pos, depth);
    std::printf("[ %s ] %-10s depth=%d  nodes=%llu\n",
                g_failures ? "FAIL" : " OK ", name, depth,
                (unsigned long long)(g_nodes - before));
}

}  // namespace

int main(int argc, char** argv) {
    slider::init();
    slider::get_device_sliders(g_sliders);
    const bool slow = (argc > 1 && std::string(argv[1]) == "slow");

    // Standard perft positions — full move-type coverage.
    run("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", slow ? 5 : 4, "startpos");
    run("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", slow ? 4 : 3, "kiwipete");
    run("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", slow ? 5 : 4, "pos3");
    run("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", slow ? 4 : 3, "pos4");
    run("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", slow ? 4 : 3, "pos5");

    // A pawnless endgame — the material the retrograde sweep actually sweeps.
    run("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", slow ? 6 : 5, "KRKR");

    if (g_failures == 0) {
        std::printf("Device movegen matches reference on all %llu nodes.\n",
                    (unsigned long long)g_nodes);
        return 0;
    }
    std::printf("\n%d movegen device check(s) failed.\n", g_failures);
    return 1;
}
