// Step 5.3 gate: the combinatorial position index (tb::CombIndex) is a correct
// bijection, and — where both are defined — represents every legal position that
// the already-verified dense tb::Index does, with no collisions.
//
// Gate A (self-consistency, every material incl. duplicates + a 5-man):
//   for all i in [0,size): decode(i) -> (squares, stm) is a VALID position
//   (kings legal, pieces distinct & off the kings) and encode(that) == i.
//   This proves encode/decode are mutual inverses over the whole index space.
//
// Gate B (regression vs dense tb::Index, distinct-piece materials only):
//   enumerate every legal canonical position from the trusted dense index; each
//   must (B1) encode to a comb index whose decode is SYMMETRY-EQUIVALENT to it,
//   and (B2) land on a comb index NOT shared by any other dense class (no
//   collision — the one property that would corrupt a solver). Injectivity here
//   means: distinct legal orbits never merge.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "slider.hpp"
#include "tb_comb_index.hpp"
#include "tb_index.hpp"
#include "types.hpp"

namespace {

int g_failures = 0;
void fail(const char* m) { std::printf("[FAIL] %s\n", m); ++g_failures; }

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
    const int df = (a & 7) - (b & 7), dr = (a >> 3) - (b >> 3);
    const int af = df < 0 ? -df : df, ar = dr < 0 ? -dr : dr;
    return af <= 1 && ar <= 1;
}

// Does some D4 transform carry tuple `a` (men squares) exactly onto `b`?
// Distinct-piece materials only (slot-wise compare, no multiset needed).
bool same_orbit(const int* a, const int* b, int men) {
    for (int g = 0; g < 8; ++g) {
        bool ok = true;
        for (int i = 0; i < men; ++i) if (sym(g, a[i]) != b[i]) { ok = false; break; }
        if (ok) return true;
    }
    return false;
}

// ---- Gate A: full self-bijection sweep over [0,size) ---------------------
void gate_a_full(std::vector<tb::Piece> extras, const char* name) {
    tb::CombIndex idx(std::move(extras));
    const int men = idx.men();
    const std::size_t N = idx.size();

    for (std::size_t i = 0; i < N; ++i) {
        int sq[8];
        Color stm;
        idx.decode(i, sq, &stm);

        // Validity: kings non-adjacent/distinct; pieces distinct & off kings.
        if (sq[0] == sq[1] || adjacent(sq[0], sq[1])) { fail("decoded kings illegal"); return; }
        for (int a = 0; a < men; ++a)
            for (int b = a + 1; b < men; ++b)
                if (sq[a] == sq[b]) { fail("decoded squares not distinct"); return; }

        if (idx.encode(sq, stm) != i) { fail("encode(decode(i)) != i"); return; }
    }
    std::printf("[ OK ] Gate A  %-7s  men=%d  size=%zu\n", name, men, N);
}

// ---- Gate A: sampled sweep (for spaces too big to fully enumerate) -------
void gate_a_sampled(std::vector<tb::Piece> extras, const char* name, std::size_t samples) {
    tb::CombIndex idx(std::move(extras));
    const int men = idx.men();
    const std::size_t N = idx.size();
    const std::size_t step = N > samples ? N / samples : 1;

    for (std::size_t i = 0; i < N; i += step) {
        int sq[8];
        Color stm;
        idx.decode(i, sq, &stm);
        if (sq[0] == sq[1] || adjacent(sq[0], sq[1])) { fail("decoded kings illegal"); return; }
        for (int a = 0; a < men; ++a)
            for (int b = a + 1; b < men; ++b)
                if (sq[a] == sq[b]) { fail("decoded squares not distinct"); return; }
        if (idx.encode(sq, stm) != i) { fail("encode(decode(i)) != i"); return; }
    }
    std::printf("[ OK ] Gate A* %-7s  men=%d  size=%zu  (sampled ~%zu)\n", name, men, N, samples);
}

// ---- Gate B: regression against the trusted dense tb::Index --------------
void gate_b(std::vector<tb::Piece> extras, const char* name) {
    tb::Index     dense(extras);
    tb::CombIndex comb(std::move(extras));
    const int men = dense.men();

    // Collision detector over the comb space (one bit per index).
    std::vector<char> hit(comb.size(), 0);

    for (std::size_t d = 0; d < dense.size(); ++d) {
        int sq[8];
        dense.raw_squares(d, sq);
        const Color stm = dense.side_to_move(d);

        const std::size_t c = comb.encode(sq, stm);
        if (c >= comb.size()) { fail("comb index out of range"); return; }

        // B1: round-trip through comb yields a symmetry-equivalent position.
        int sq2[8];
        Color stm2;
        comb.decode(c, sq2, &stm2);
        if (stm2 != stm) { fail("comb decode flipped side to move"); return; }
        if (!same_orbit(sq, sq2, men)) { fail("comb decode not symmetry-equivalent"); return; }

        // B2: no two distinct dense classes share a comb index.
        if (hit[c]) { fail("comb index collision between distinct legal orbits"); return; }
        hit[c] = 1;
    }
    std::printf("[ OK ] Gate B  %-7s  dense=%zu  comb=%zu\n", name, dense.size(), comb.size());
}

// ---- Position path: decode_pos -> encode(Position) round-trips -----------
// The raw-squares gates never touch encode(const Position&); this does. For a
// DUPLICATE material (KRRK) it is the only coverage of the same-(color,type)
// peel loop that disambiguates identical pieces when reading them off a board.
void gate_position(std::vector<tb::Piece> extras, const char* name, std::size_t samples) {
    tb::CombIndex idx(std::move(extras));
    const std::size_t N = idx.size();
    const std::size_t step = (samples && N > samples) ? N / samples : 1;
    for (std::size_t i = 0; i < N; i += step) {
        const Position pos = idx.decode_pos(i);
        if (idx.encode(pos) != i) { fail("encode(decode_pos(i)) != i"); return; }
    }
    std::printf("[ OK ] Pos    %-7s  size=%zu%s\n", name, N, step > 1 ? "  (sampled)" : "");
}

}  // namespace

int main(int argc, char** argv) {
    slider::init();
    const bool slow = (argc >= 2 && std::strcmp(argv[1], "slow") == 0);

    using PT = PieceType;
    const Color W = Color::White, B = Color::Black;

    // Fast tier: 3-man — full self-bijection + full regression vs dense index.
    gate_a_full({{W, PT::Rook}},  "KRK");
    gate_a_full({{W, PT::Queen}}, "KQK");
    gate_b({{W, PT::Rook}},  "KRK");
    gate_b({{W, PT::Queen}}, "KQK");
    gate_position({{W, PT::Rook}}, "KRK", 0);  // full: exercises encode(Position)

    if (slow) {
        std::printf("--- slow (4-man + duplicate + 5-man) ---\n");
        // 4-man distinct: both gates (full sweep + full regression).
        gate_a_full({{W, PT::Queen}, {B, PT::Rook}},   "KQKR");
        gate_b({{W, PT::Queen}, {B, PT::Rook}},        "KQKR");

        // Duplicate pieces: dense tb::Index can't build these, so Gate A only
        // (proves the identical-piece subset machinery is a bijection).
        gate_a_full({{W, PT::Rook}, {W, PT::Rook}},    "KRRK");
        // ...and the ONLY test of encode(Position) with identical pieces.
        gate_position({{W, PT::Rook}, {W, PT::Rook}},  "KRRK", 0);

        // First 5-man: space too big to fully sweep — sampled bijection + the
        // headline that it BUILDS with a sane, non-overflowing size.
        gate_a_sampled({{W, PT::Queen}, {W, PT::Rook}, {B, PT::Rook}}, "KQRKR", 2'000'000);
    }

    if (g_failures == 0) {
        std::printf("\nAll CombIndex tests passed%s.\n", slow ? " (incl. 4-man/dup/5-man)" : "");
        return 0;
    }
    std::printf("\n%d CombIndex test(s) failed.\n", g_failures);
    return 1;
}
