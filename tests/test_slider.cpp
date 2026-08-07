// Step 4 gate (MANDATORY): the magic-bitboard rook/bishop attacks must agree
// with the slow ray-walking reference on all 64 squares across 10,000 random
// occupancy masks each, for both pieces. Incorrect magics fail silently and
// surface as plausible-looking bugs far downstream, so this gate is strict.

#include <cstdint>
#include <cstdio>
#include <initializer_list>

#include "slider.hpp"
#include "types.hpp"

namespace {

// Same xorshift64 style PRNG as the engine, but a DIFFERENT seed so the test
// occupancies are independent of the magic-search stream.
uint64_t g_rng = 0xD1B54A32D192ED03ULL;
uint64_t rand64() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return g_rng;
}
// A realistic-ish sparse occupancy (AND of two draws) plus, occasionally, a
// dense one — exercise both light and heavy blocker configurations.
uint64_t rand_occ() {
    uint64_t o = rand64() & rand64();
    if ((rand64() & 3) == 0) o |= rand64();  // sometimes dense
    return o;
}

}  // namespace

int main() {
    slider::init();

    constexpr int TRIALS = 10000;
    long long checks = 0;
    int rook_fail = 0, bishop_fail = 0;

    for (int sq = 0; sq < 64; ++sq) {
        const Square s = static_cast<Square>(sq);

        // Deterministic corner cases first: empty and full boards.
        for (Bitboard occ : {Bitboard{0}, ~Bitboard{0}}) {
            if (slider::rook_attacks(s, occ) != slider::rook_ref(sq, occ)) ++rook_fail;
            if (slider::bishop_attacks(s, occ) != slider::bishop_ref(sq, occ)) ++bishop_fail;
            checks += 2;
        }

        for (int t = 0; t < TRIALS; ++t) {
            const Bitboard occ = rand_occ();

            const Bitboard rm = slider::rook_attacks(s, occ);
            const Bitboard rr = slider::rook_ref(sq, occ);
            if (rm != rr) {
                if (rook_fail < 5) {
                    std::printf("[FAIL] rook sq=%d occ=%016llx magic=%016llx ref=%016llx\n",
                                sq, (unsigned long long)occ,
                                (unsigned long long)rm, (unsigned long long)rr);
                }
                ++rook_fail;
            }

            const Bitboard bm = slider::bishop_attacks(s, occ);
            const Bitboard br = slider::bishop_ref(sq, occ);
            if (bm != br) {
                if (bishop_fail < 5) {
                    std::printf("[FAIL] bishop sq=%d occ=%016llx magic=%016llx ref=%016llx\n",
                                sq, (unsigned long long)occ,
                                (unsigned long long)bm, (unsigned long long)br);
                }
                ++bishop_fail;
            }
            checks += 2;
        }
    }

    std::printf("checked %lld attack sets (64 squares x %d occ x 2 pieces + corners)\n",
                checks, TRIALS);
    if (rook_fail == 0 && bishop_fail == 0) {
        std::printf("[ OK ] magic == reference on all squares and occupancies\n");
        return 0;
    }
    std::printf("[FAIL] rook mismatches=%d bishop mismatches=%d\n", rook_fail, bishop_fail);
    return 1;
}
