// Device-side retrograde sweep — the GPU port of solve_sweep_comb's per-node
// update (src/tb_solve_comb.cpp). This header holds the CH_HD update function
// that IS the kernel body: for one live position it decodes the index, generates
// legal moves, and takes best = max over moves of age(-child), looking child
// values up in the same-material value array or in a capture-exit sub-table.
//
// Faithful to the CPU oracle, with three deliberate, gate-verified choices:
//   1. Sub-tables use the CombIndex layout (comb_encode), not the dense Index.
//      DTM is position-intrinsic and comb==dense is proven (test_tb_comb_solve),
//      so this returns identical child values without porting the dense indexer.
//   2. The driver iterates Jacobi (ping-pong buffers) rather than the CPU's
//      in-place Gauss-Seidel. Both converge to the same unique DTM fixpoint; only
//      the pass COUNT differs, and the gate compares the fixpoint.
//   3. Distinct pieces only (like solve_sweep_comb): capture detection is "which
//      (color,type) bitboard emptied", ambiguous for duplicates.
//
// Because it is CH_HD, test_tb_sweep_device runs the whole sweep on the HOST with
// these functions and diffs the fixpoint against solve_sweep_comb — a local
// Phase 3 gate with no GPU. cuda/sweep_check.cu runs the same update in a kernel.

#pragma once

#include <cstdint>

#include "cuda_compat.hpp"
#include "movegen_device.hpp"
#include "position.hpp"
#include "slider_device.hpp"
#include "tb_comb_index_device.hpp"
#include "tb_solve.hpp"   // tb::MATE (constexpr; device-usable)
#include "types.hpp"

namespace tb {

// Per-position liveness/terminal class, filled once on the host (mirrors the
// ST_* enum in solve_sweep_comb). The device kernel updates only SW_SOLVE nodes.
enum : uint8_t { SW_ILLEGAL = 0, SW_SOLVE = 1, SW_MATE = 2, SW_STALE = 3 };

// A capture-exit sub-table for one piece group (the material with that group's
// piece removed), in CombIndex layout. draw=1 means the capture reaches bare KK.
struct DeviceSub {
    int            draw;    // 1 => exit to bare KK (child value 0)
    const int16_t* value;   // comb-keyed sub-table values (null iff draw)
    DeviceMaterial mat;     // sub material descriptor (for the sub encode)
};

// age(): move a mate score one ply further from mate. Local CH_HD copy of
// tb::age (which is host-only inline); identical definition + mate convention.
CH_HD inline int sweep_age(int x) { return x > 0 ? x - 1 : (x < 0 ? x + 1 : 0); }

// Build a Position (bitboards + side to move only) from decoded squares. Value-
// initialization gives empty boards, NO_CASTLING, ep=None, clocks 0 — exactly a
// cleared tablebase board, matching CombIndex::make_position(with_zobrist=false).
CH_HD inline Position sweep_make_position(const int* squares, Color stm,
                                          const DeviceMaterial& mat) {
    Position p{};
    const Bitboard wk = square_bb(static_cast<Square>(squares[0]));
    const Bitboard bk = square_bb(static_cast<Square>(squares[1]));
    p.by_type[type_index(PieceType::King)] |= wk | bk;
    p.by_color[color_index(Color::White)]  |= wk;
    p.by_color[color_index(Color::Black)]  |= bk;
    for (int g = 0; g < mat.num_groups; ++g) {
        const DeviceGroup& G = mat.groups[g];
        for (int j = 0; j < G.count; ++j) {
            const Bitboard b = square_bb(static_cast<Square>(squares[2 + G.slots[j]]));
            p.by_type[G.type]   |= b;
            p.by_color[G.color] |= b;
        }
    }
    p.side_to_move = stm;
    return p;
}

// Encode a Position of `mat`'s material into its comb index (distinct pieces:
// each group is one piece, its square is the single set bit of its bitboard).
// Mirrors CombIndex::encode(Position) for the distinct case.
CH_HD inline uint64_t sweep_encode_position(const Position& p,
                                            const DeviceKingTable& kt,
                                            const DeviceMaterial& mat) {
    int sq[kDevMaxMen];
    sq[0] = ctz_dev(p.by_type[type_index(PieceType::King)] & p.by_color[color_index(Color::White)]);
    sq[1] = ctz_dev(p.by_type[type_index(PieceType::King)] & p.by_color[color_index(Color::Black)]);
    for (int g = 0; g < mat.num_groups; ++g) {
        const DeviceGroup& G = mat.groups[g];  // count==1 (distinct pieces)
        sq[2 + G.slots[0]] = ctz_dev(p.by_type[G.type] & p.by_color[G.color]);
    }
    return comb_encode(sq, (p.side_to_move == Color::Black) ? 1 : 0, kt, mat);
}

// The per-node retrograde update. Precondition: state[i] == SW_SOLVE. Returns the
// node's new value = max over legal moves of age(-child). subs has one entry per
// group (subs[g] = material minus group g's piece).
//
// `v_old` is __restrict__ because the child lookups (v_old[encode(child)]) are a
// scattered retrograde gather — the worst-latency reads in the kernel. The
// restrict promise is TRUE: within a pass v_old is a distinct ping-pong buffer
// that never aliases v_new/state/subs, and no local aliases it either. That lets
// nvcc route the gather through the read-only data cache (LDG), which suits
// scattered reads better than an ordinary global load. Pure compiler hint —
// correctness is unchanged (the host gate still diffs the fixpoint bit-for-bit);
// the benefit is GPU-only and shows up as l2/tex hit rate in Nsight. On host
// (GCC) __restrict__ is a no-op-ish aliasing hint, equally harmless.
CH_HD inline int16_t sweep_update(uint64_t i, const int16_t* __restrict__ v_old,
                                  const DeviceKingTable& kt,
                                  const DeviceMaterial& mat,
                                  const slider::DeviceSliders& S,
                                  const DeviceSub* subs) {
    int sq[kDevMaxMen];
    int bit;
    comb_decode(i, sq, &bit, kt, mat);
    Position pos = sweep_make_position(sq, bit ? Color::Black : Color::White, mat);

    MoveList ml;
    movegen_dev::generate_legal(pos, ml, S);

    int best = -(MATE + 1);
    for (int k = 0; k < ml.size(); ++k) {
        const Move m = ml[k];
        StateInfo st;
        movegen_dev::make_move(pos, m, st);

        // Which group's piece did this move capture (its bitboard went empty)?
        int captured = -1;
        for (int g = 0; g < mat.num_groups; ++g) {
            const DeviceGroup& G = mat.groups[g];
            if ((pos.by_type[G.type] & pos.by_color[G.color]) == 0) { captured = g; break; }
        }

        int child;
        if (captured < 0) {
            child = v_old[sweep_encode_position(pos, kt, mat)];
        } else {
            const DeviceSub& s = subs[captured];
            // Hoist to a restrict local so the sub-table probe (also a scattered
            // gather) gets the same read-only-cache treatment as the v_old gather.
            const int16_t* __restrict__ sv = s.value;
            child = s.draw ? 0 : sv[sweep_encode_position(pos, kt, s.mat)];
        }

        movegen_dev::unmake_move(pos, m, st);

        const int cand = sweep_age(-child);
        if (cand > best) best = cand;
    }
    return static_cast<int16_t>(best);
}

}  // namespace tb
