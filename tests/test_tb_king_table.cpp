// Step 5.2 gate: the king-anchored canonical king-pair table (tb::KingTable)
// folds king pairs by the SAME symmetry as the already-verified tb::Index — so
// the combinatorial index inherits a symmetry we trust.
//
// Primary gate (no recalled constants): tb::KingTable and tb::Index (built for
// bare kings) must induce the IDENTICAL partition of the legal king pairs into
// symmetry classes — same number of classes, same grouping. That is exactly
// what it means for the king-anchored folding to match the whole-tuple folding.
//
// Secondary structural checks: legality, self-canonical reps, the recorded
// transform actually reaches the canonical rep, round-trips. Plus two soft
// anchors printed for sanity (3612 legal pairs, 462 canonical configs).

#include <cstdint>
#include <cstdio>
#include <vector>

#include "slider.hpp"
#include "tb_index.hpp"
#include "tb_king_table.hpp"
#include "types.hpp"

namespace {

int g_failures = 0;
void fail(const char* m) { std::printf("[FAIL] %s\n", m); ++g_failures; }
void expect(bool ok, const char* m) { if (!ok) fail(m); }

// Independent D4 transforms (same formulas as the engine; the test must not lean
// on either the KingTable's or the Index's private copy).
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

bool adjacent(int a, int b) {
    const int fa = a & 7, ra = a >> 3, fb = b & 7, rb = b >> 3;
    const int df = fa > fb ? fa - fb : fb - fa;
    const int dr = ra > rb ? ra - rb : rb - ra;
    return df <= 1 && dr <= 1;  // includes a==b, filtered separately
}

}  // namespace

int main() {
    slider::init();
    tb::KingTable kt;

    // --- Structural checks over all raw pairs -----------------------------
    long long legal_pairs = 0;
    for (int wk = 0; wk < 64; ++wk) {
        for (int bk = 0; bk < 64; ++bk) {
            const bool legal = (wk != bk) && !adjacent(wk, bk);
            const int id = kt.id_of(wk, bk);

            if (!legal) {
                if (id != -1) fail("illegal pair has an id");
                continue;
            }
            ++legal_pairs;
            if (id < 0 || id >= kt.count()) { fail("legal pair id out of range"); continue; }

            // The recorded transform must carry (wk,bk) to the canonical rep.
            const int g = kt.transform_of(wk, bk);
            int cwk, cbk;
            kt.kings_of(id, cwk, cbk);
            if (sym(g, wk) != cwk || sym(g, bk) != cbk)
                fail("transform does not reach canonical rep");
        }
    }

    // Every canonical rep must be self-canonical (min-code of its orbit is
    // itself) and reached by the identity transform.
    for (int id = 0; id < kt.count(); ++id) {
        int wk, bk;
        kt.kings_of(id, wk, bk);
        uint32_t best = 0xFFFFFFFFu;
        for (int t = 0; t < 8; ++t) {
            const uint32_t code = static_cast<uint32_t>(sym(t, wk)) * 64u +
                                  static_cast<uint32_t>(sym(t, bk));
            if (code < best) best = code;
        }
        if (best != static_cast<uint32_t>(wk * 64 + bk)) fail("canonical rep is not self-canonical");
        if (kt.transform_of(wk, bk) != 0) fail("canonical rep transform is not identity");
        if (kt.id_of(wk, bk) != id) fail("kings_of/id_of not inverse");
    }

    // --- Primary gate: same partition as tb::Index (bare kings) -----------
    // Index for KK doubles every spatial class by side-to-move; its spatial
    // canonicalization (whole-tuple min-code over 2 men) is exactly a king pair.
    tb::Index kk(std::vector<tb::Piece>{});  // empty extras => bare kings
    if (kk.size() % 2 != 0) fail("KK index size not even (stm should double it)");
    if (static_cast<long long>(kk.size() / 2) != kt.count())
        fail("KingTable count != tb::Index KK spatial-class count");

    // Build the map KingTable-id -> Index-class over all legal pairs, and verify
    // it is a consistent bijection (the two partitions coincide exactly).
    std::vector<long long> kt_to_idx(static_cast<std::size_t>(kt.count()), -1);
    std::vector<int>       idx_to_kt(kk.size(), -1);
    for (int wk = 0; wk < 64; ++wk) {
        for (int bk = 0; bk < 64; ++bk) {
            if (wk == bk || adjacent(wk, bk)) continue;
            int sq[2] = {wk, bk};
            const std::size_t ii = kk.encode_squares(sq, Color::White);
            const int kid = kt.id_of(wk, bk);

            if (kt_to_idx[static_cast<std::size_t>(kid)] == -1)
                kt_to_idx[static_cast<std::size_t>(kid)] = static_cast<long long>(ii);
            else if (kt_to_idx[static_cast<std::size_t>(kid)] != static_cast<long long>(ii))
                { fail("one KingTable class maps to two Index classes"); }

            if (idx_to_kt[ii] == -1) idx_to_kt[ii] = kid;
            else if (idx_to_kt[ii] != kid)
                { fail("one Index class maps to two KingTable classes"); }
        }
    }

    std::printf("[info] legal king pairs = %lld (expect 3612)\n", legal_pairs);
    std::printf("[info] canonical king configs = %d (expect 462)\n", kt.count());
    expect(legal_pairs == 3612, "legal pair count != 3612");
    expect(kt.count() == 462, "canonical king config count != 462");

    if (g_failures == 0) {
        std::printf("All KingTable tests passed (partition matches tb::Index).\n");
        return 0;
    }
    std::printf("\n%d KingTable test(s) failed.\n", g_failures);
    return 1;
}
