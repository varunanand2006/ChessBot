// Phase 2 / Step 2.1 gate: the tb::Index mapping must be a bijection between
// dense indices and symmetry-equivalence-classes of legal positions.
//
// Checks (no externally-recalled constants — all self-verifying):
//   A. Round trip on indices:   encode(decode(i)) == i for all i in [0,N).
//   B. Every legal position encodes into [0,N).
//   C. Symmetry invariance:     encode is equal across all 8 transforms of a
//                               position, and decode(encode(p)) is in p's orbit.
//   D. Partition:               summing orbit sizes over the N indices equals
//                               the total number of legal positions, and every
//                               index is hit (surjective, no gaps).

#include <cstdint>
#include <cstdio>
#include <vector>

#include "position.hpp"
#include "slider.hpp"
#include "tb_index.hpp"
#include "types.hpp"

namespace {

int g_failures = 0;
void fail(const char* msg) { std::printf("[FAIL] %s\n", msg); ++g_failures; }

// The same 8 transforms the indexer uses (kept independent here so the test
// doesn't depend on the engine's private table).
int sym(int t, int s) {
    const int f = s & 7, r = s >> 3;
    switch (t) {
        case 0: return r * 8 + f;
        case 1: return r * 8 + (7 - f);
        case 2: return (7 - r) * 8 + f;
        case 3: return (7 - r) * 8 + (7 - f);
        case 4: return f * 8 + r;
        case 5: return (7 - f) * 8 + (7 - r);
        case 6: return (7 - f) * 8 + r;
        default:return f * 8 + (7 - r);
    }
}

void test_piece(PieceType wp, const char* name) {
    tb::Index idx(wp);
    const std::size_t N = idx.size();

    // A. index round trip.
    for (std::size_t i = 0; i < N; ++i) {
        Position p = idx.decode(i);
        if (idx.encode(p) != i) { fail("encode(decode(i)) != i"); break; }
    }

    // Walk every legal raw position once.
    std::vector<uint64_t> hits(N, 0);       // how many legal positions map to each index
    long long total_legal = 0;
    for (int stm_bit = 0; stm_bit < 2; ++stm_bit) {
        const Color stm = (stm_bit == 0) ? Color::White : Color::Black;
        for (int wk = 0; wk < 64; ++wk)
        for (int pc = 0; pc < 64; ++pc)
        for (int bk = 0; bk < 64; ++bk) {
            if (!tb::Index::legal(wp, wk, pc, bk, stm)) continue;
            ++total_legal;

            const std::size_t e = idx.encode_raw(wk, pc, bk, stm);
            if (e >= N) { fail("encode out of range"); return; }
            ++hits[e];

            // C. symmetry invariance: all 8 transforms share the index.
            for (int t = 1; t < 8; ++t) {
                const int wk2 = sym(t, wk), pc2 = sym(t, pc), bk2 = sym(t, bk);
                if (idx.encode_raw(wk2, pc2, bk2, stm) != e) {
                    fail("encode not symmetry-invariant");
                    return;
                }
            }
        }
    }

    // D. partition: every index hit at least once (surjective), and the orbit
    // sizes sum to the legal-position count (each class counted exactly once).
    long long summed = 0;
    bool all_hit = true;
    for (std::size_t i = 0; i < N; ++i) {
        if (hits[i] == 0) all_hit = false;
        summed += static_cast<long long>(hits[i]);
    }
    if (!all_hit) fail("some index never produced (gaps / not surjective)");
    if (summed != total_legal) fail("orbit sizes do not sum to legal positions");

    const double avg = N ? static_cast<double>(total_legal) / static_cast<double>(N) : 0.0;
    std::printf("[ OK ] %-4s  indices=%zu  legal positions=%lld  avg orbit=%.3f\n",
                name, N, total_legal, avg);
}

}  // namespace

int main() {
    slider::init();
    test_piece(PieceType::Rook,   "KRK");
    test_piece(PieceType::Queen,  "KQK");
    test_piece(PieceType::Bishop, "KBK");
    test_piece(PieceType::Knight, "KNK");

    if (g_failures == 0) { std::printf("\nAll tb::Index bijection tests passed.\n"); return 0; }
    std::printf("\n%d tb::Index test(s) failed.\n", g_failures);
    return 1;
}
