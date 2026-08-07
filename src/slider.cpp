#include "slider.hpp"

#include <bit>
#include <cstdio>
#include <cstdlib>

namespace slider {

namespace {

// --- small local bit helpers ------------------------------------------------
constexpr Bitboard bit(int file, int rank) {
    return Bitboard{1} << (rank * 8 + file);
}
constexpr bool on_board(int file, int rank) {
    return file >= 0 && file < 8 && rank >= 0 && rank < 8;
}

// Direction sets as (file delta, rank delta).
constexpr int ROOK_DF[4]   = {+1, -1, 0, 0};
constexpr int ROOK_DR[4]   = {0, 0, +1, -1};
constexpr int BISHOP_DF[4] = {+1, +1, -1, -1};
constexpr int BISHOP_DR[4] = {+1, -1, +1, -1};

// Generic ray-walk: OR in every square along each direction, stopping at (and
// including) the first occupied square. This is the reference oracle.
Bitboard ray_attacks(int sq, Bitboard occ, const int* df, const int* dr) {
    Bitboard result = 0;
    const int f = sq & 7, r = sq >> 3;
    for (int d = 0; d < 4; ++d) {
        int ff = f + df[d], rr = r + dr[d];
        while (on_board(ff, rr)) {
            const Bitboard b = bit(ff, rr);
            result |= b;
            if (occ & b) break;  // blocker: include it, then stop
            ff += df[d];
            rr += dr[d];
        }
    }
    return result;
}

// Relevant-occupancy masks: the squares whose contents can block this piece,
// which EXCLUDES the board edges (a piece on the edge of a ray can't block
// anything beyond it, so those bits are irrelevant to the attack set). Fewer
// mask bits => smaller per-square table.
Bitboard rook_mask(int sq) {
    Bitboard m = 0;
    const int f = sq & 7, r = sq >> 3;
    for (int ff = f + 1; ff <= 6; ++ff) m |= bit(ff, r);
    for (int ff = f - 1; ff >= 1; --ff) m |= bit(ff, r);
    for (int rr = r + 1; rr <= 6; ++rr) m |= bit(f, rr);
    for (int rr = r - 1; rr >= 1; --rr) m |= bit(f, rr);
    return m;
}
Bitboard bishop_mask(int sq) {
    Bitboard m = 0;
    const int f = sq & 7, r = sq >> 3;
    for (int ff = f + 1, rr = r + 1; ff <= 6 && rr <= 6; ++ff, ++rr) m |= bit(ff, rr);
    for (int ff = f + 1, rr = r - 1; ff <= 6 && rr >= 1; ++ff, --rr) m |= bit(ff, rr);
    for (int ff = f - 1, rr = r + 1; ff >= 1 && rr <= 6; --ff, ++rr) m |= bit(ff, rr);
    for (int ff = f - 1, rr = r - 1; ff >= 1 && rr >= 1; --ff, --rr) m |= bit(ff, rr);
    return m;
}

// --- magic table storage ----------------------------------------------------
struct Magic {
    Bitboard  mask    = 0;        // relevant-occupancy mask
    Bitboard  magic   = 0;        // multiplier
    Bitboard* attacks = nullptr;  // points into ROOK_TABLE / BISHOP_TABLE
    unsigned  shift   = 0;        // 64 - popcount(mask)

    unsigned index(Bitboard occ) const {
        return static_cast<unsigned>(((occ & mask) * magic) >> shift);
    }
};

// Shared attack tables. Sizes are the well-known fancy-magic totals: the sum
// over all 64 squares of 2^(relevant bits). Rook relevant bits peak at 12
// (corner-free interior of both rays), bishop at 9. These are compile-time
// constants, not heap — the whole engine avoids heap below the search root.
constexpr int ROOK_TABLE_SIZE   = 102400;
constexpr int BISHOP_TABLE_SIZE = 5248;

Magic    ROOK[64];
Magic    BISHOP[64];
Bitboard ROOK_TABLE[ROOK_TABLE_SIZE];
Bitboard BISHOP_TABLE[BISHOP_TABLE_SIZE];

// --- deterministic sparse PRNG for magic search -----------------------------
// xorshift64. Seeded with a fixed constant so the generated magics — and thus
// the whole build — are reproducible. Sparse candidates (AND of three draws)
// have few set bits, which empirically find collision-free magics fastest.
uint64_t g_rng = 0x0123456789ABCDEFULL;
uint64_t next_rand() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return g_rng;
}
uint64_t sparse_rand() { return next_rand() & next_rand() & next_rand(); }

// Find a magic for one square and fill its slice of the shared table.
// Returns the number of table entries consumed (2^bits).
int init_square(bool bishop, int sq, Bitboard* table_base, int offset) {
    const Bitboard mask = bishop ? bishop_mask(sq) : rook_mask(sq);
    const int  bits  = std::popcount(mask);
    const int  size  = 1 << bits;
    const unsigned shift = static_cast<unsigned>(64 - bits);

    // Enumerate every occupancy subset of the mask (Carry-Rippler trick) and
    // precompute its reference attack set.
    static Bitboard occ_subset[4096];  // 4096 = 2^12, the rook maximum
    static Bitboard reference[4096];
    int n = 0;
    Bitboard sub = 0;
    do {
        occ_subset[n] = sub;
        reference[n]  = bishop ? bishop_ref(sq, sub) : rook_ref(sq, sub);
        ++n;
        sub = (sub - mask) & mask;
    } while (sub != 0);

    // Trial magics until one maps all subsets with no *destructive* collision.
    // Two subsets may share an index only if they have the same attack set
    // ("constructive" collision) — that keeps the table minimal.
    static Bitboard used[4096];
    Magic& m = bishop ? BISHOP[sq] : ROOK[sq];
    m.mask  = mask;
    m.shift = shift;
    m.attacks = table_base + offset;

    for (long long attempt = 0;; ++attempt) {
        const Bitboard magic = sparse_rand();
        // Heuristic reject: a good magic spreads mask bits into the high byte.
        if (std::popcount((mask * magic) & 0xFF00000000000000ULL) < 6) continue;

        for (int i = 0; i < size; ++i) used[i] = 0;
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            const unsigned idx =
                static_cast<unsigned>(((occ_subset[i] & mask) * magic) >> shift);
            if (used[idx] == 0) {
                used[idx] = reference[i];
            } else if (used[idx] != reference[i]) {
                ok = false;  // destructive collision
                break;
            }
        }
        if (ok) {
            m.magic = magic;
            for (int i = 0; i < size; ++i) m.attacks[i] = used[i];
            break;
        }
        if (attempt > 100'000'000LL) {  // never hit in practice with sparse draws
            std::fprintf(stderr, "slider::init: no magic found for square %d\n", sq);
            std::abort();
        }
    }
    return size;
}

bool g_initialized = false;

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
Bitboard rook_ref(int sq, Bitboard occ)   { return ray_attacks(sq, occ, ROOK_DF, ROOK_DR); }
Bitboard bishop_ref(int sq, Bitboard occ) { return ray_attacks(sq, occ, BISHOP_DF, BISHOP_DR); }

void init() {
    if (g_initialized) return;
    int r_off = 0, b_off = 0;
    for (int sq = 0; sq < 64; ++sq) {
        r_off += init_square(false, sq, ROOK_TABLE, r_off);
        b_off += init_square(true, sq, BISHOP_TABLE, b_off);
    }
    // Sanity: the offsets must exactly fill the tables. If these ever fire, the
    // mask/relevant-bit logic drifted from the hard-coded table sizes.
    if (r_off != ROOK_TABLE_SIZE || b_off != BISHOP_TABLE_SIZE) {
        std::fprintf(stderr, "slider::init: table size mismatch rook=%d bishop=%d\n",
                     r_off, b_off);
        std::abort();
    }
    g_initialized = true;
}

Bitboard rook_attacks(Square s, Bitboard occ) {
    const Magic& m = ROOK[sq_index(s)];
    return m.attacks[m.index(occ)];
}
Bitboard bishop_attacks(Square s, Bitboard occ) {
    const Magic& m = BISHOP[sq_index(s)];
    return m.attacks[m.index(occ)];
}
Bitboard queen_attacks(Square s, Bitboard occ) {
    return rook_attacks(s, occ) | bishop_attacks(s, occ);
}

void get_device_sliders(DeviceSliders& out) {
    if (!g_initialized) init();
    // Process-lifetime storage the returned pointers alias. Converting each
    // internal Magic (which holds an absolute pointer into the shared table) to
    // a self-contained {offset} form so it survives a memcpy to the device.
    static DeviceMagic rook_dev[64];
    static DeviceMagic bishop_dev[64];
    for (int sq = 0; sq < 64; ++sq) {
        rook_dev[sq]   = {ROOK[sq].mask,   ROOK[sq].magic,
                          static_cast<int>(ROOK[sq].attacks - ROOK_TABLE),   ROOK[sq].shift};
        bishop_dev[sq] = {BISHOP[sq].mask, BISHOP[sq].magic,
                          static_cast<int>(BISHOP[sq].attacks - BISHOP_TABLE), BISHOP[sq].shift};
    }
    out.rook              = rook_dev;
    out.bishop            = bishop_dev;
    out.rook_table        = ROOK_TABLE;
    out.bishop_table      = BISHOP_TABLE;
    out.rook_table_size   = ROOK_TABLE_SIZE;
    out.bishop_table_size = BISHOP_TABLE_SIZE;
}

}  // namespace slider
