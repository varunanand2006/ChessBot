// Phase 2 gate: tb::Index is a bijection between dense indices and symmetry
// equivalence classes of legal positions, for pawnless materials.
//
// Generalized to any pawnless material (2 kings + distinct extra pieces).
// Checks per material (no recalled constants):
//   A. index round trip:  encode(decode(i)) == i for all i in [0,N).
//   B. symmetry invariance: transforming a position by any of the 8 D4 elements
//      leaves its index unchanged.
//   C. partition:  sum of orbit sizes over the N indices equals the number of
//      legal positions found by independent enumeration (surjective + exact).
//
// 3-man materials run in the default suite; 4-man (millions of positions) is a
// separate "slow" entry.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "slider.hpp"
#include "tb_index.hpp"
#include "types.hpp"

namespace {

int g_failures = 0;
void fail(const char* m) { std::printf("[FAIL] %s\n", m); ++g_failures; }

// Independent copy of the 8 D4 transforms (so the test doesn't lean on the
// engine's private table).
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

void check(std::vector<tb::Piece> extras, const char* name) {
    tb::Index idx(std::move(extras));
    const std::size_t N = idx.size();
    const int men = idx.men();

    long long orbit_sum = 0;
    for (std::size_t i = 0; i < N; ++i) {
        int sq[4];
        idx.raw_squares(i, sq);
        const Color stm = idx.side_to_move(i);

        // A. round trip.
        if (idx.encode(idx.decode(i)) != i) { fail("encode(decode(i)) != i"); return; }

        // B. symmetry invariance + orbit size (distinct transformed tuples).
        uint32_t codes[8];
        for (int g = 0; g < 8; ++g) {
            int sq2[4];
            uint32_t code = 0;
            for (int k = 0; k < men; ++k) { sq2[k] = sym(g, sq[k]); code = code * 64u + static_cast<uint32_t>(sq2[k]); }
            codes[g] = code;
            if (idx.encode_squares(sq2, stm) != i) { fail("index not symmetry-invariant"); return; }
        }
        int distinct = 0;
        for (int g = 0; g < 8; ++g) {
            bool seen = false;
            for (int h = 0; h < g; ++h) if (codes[h] == codes[g]) { seen = true; break; }
            if (!seen) ++distinct;
        }
        orbit_sum += distinct;
    }

    // C. Independent count of legal positions by full enumeration.
    long long legal_total = 0;
    uint64_t space = 1;
    for (int i = 0; i < men; ++i) space *= 64;
    int sq[4];
    for (int stm_bit = 0; stm_bit < 2; ++stm_bit) {
        const Color stm = (stm_bit == 0) ? Color::White : Color::Black;
        for (uint64_t s = 0; s < space; ++s) {
            uint64_t t = s;
            for (int k = men - 1; k >= 0; --k) { sq[k] = static_cast<int>(t % 64); t /= 64; }
            if (tb::Index::legal(idx.extras(), sq, stm)) ++legal_total;
        }
    }

    if (orbit_sum != legal_total) fail("orbit sizes do not sum to legal positions");

    const double avg = N ? static_cast<double>(legal_total) / static_cast<double>(N) : 0.0;
    std::printf("[ OK ] %-6s  men=%d  indices=%zu  legal=%lld  avg orbit=%.3f\n",
                name, men, N, legal_total, avg);
}

}  // namespace

int main(int argc, char** argv) {
    slider::init();
    const bool slow = (argc >= 2 && std::strcmp(argv[1], "slow") == 0);

    using PT = PieceType;
    // 3-man (fast).
    check({{Color::White, PT::Rook}},   "KRK");
    check({{Color::White, PT::Queen}},  "KQK");
    check({{Color::White, PT::Bishop}}, "KBK");
    check({{Color::White, PT::Knight}}, "KNK");

    if (slow) {
        std::printf("--- slow (4-man) ---\n");
        check({{Color::White, PT::Queen}, {Color::Black, PT::Rook}},  "KQKR");
        check({{Color::White, PT::Rook},  {Color::Black, PT::Knight}},"KRKN");
        check({{Color::White, PT::Rook},  {Color::White, PT::Bishop}},"KRBK");
    }

    if (g_failures == 0) { std::printf("\nAll tb::Index tests passed%s.\n", slow ? " (incl. 4-man)" : ""); return 0; }
    std::printf("\n%d tb::Index test(s) failed.\n", g_failures);
    return 1;
}
