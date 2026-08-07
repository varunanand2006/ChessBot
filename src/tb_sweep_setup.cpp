#include "tb_sweep_setup.hpp"

#include <cstdio>
#include <cstdlib>
#include <utility>

#include "movegen.hpp"
#include "slider.hpp"
#include "tb_index.hpp"
#include "tb_solve.hpp"

namespace tb {

namespace {

// Distinct (color,type) required — capture detection identifies the removed
// piece by which bitboard emptied, ambiguous for duplicates (same as
// solve_sweep_comb).
void require_distinct(const std::vector<Piece>& extras) {
    for (std::size_t a = 0; a < extras.size(); ++a)
        for (std::size_t b = a + 1; b < extras.size(); ++b)
            if (extras[a].color == extras[b].color && extras[a].type == extras[b].type) {
                std::fprintf(stderr, "tb_sweep_setup: duplicate pieces unsupported\n");
                std::abort();
            }
}

}  // namespace

SweepSetup build_sweep_setup(const CombIndex& idx) {
    const std::vector<Piece>& extras = idx.extras();
    require_distinct(extras);

    SweepSetup s;
    s.N = idx.size();
    idx.fill_device(s.kt, s.mat);

    // --- Pass 0: classify (mirrors solve_sweep_comb) -----------------------
    s.state.assign(s.N, SW_ILLEGAL);
    s.v_init.assign(s.N, 0);
    int sq[CombIndex::kMaxMen];
    Color stm;
    for (std::size_t i = 0; i < s.N; ++i) {
        idx.decode(i, sq, &stm);
        if (!Index::legal(extras, sq, stm)) continue;  // SW_ILLEGAL
        Position pos = idx.make_position(sq, stm, /*with_zobrist=*/false);
        MoveList ml;
        movegen::generate_legal(pos, ml);
        if (ml.size() == 0) {
            if (movegen::in_check(pos)) {
                s.state[i]  = SW_MATE;
                s.v_init[i] = static_cast<int16_t>(-MATE);
            } else {
                s.state[i] = SW_STALE;  // v stays 0
            }
        } else {
            s.state[i] = SW_SOLVE;
        }
    }

    // --- Capture-exit sub-tables, one per group ----------------------------
    // Group g's piece is (color,type) from the device descriptor; the sub is the
    // material with that one piece removed. Solved with solve_sweep_comb so the
    // values are comb-keyed (== dense DTM, per test_tb_comb_solve).
    s.subs.resize(static_cast<std::size_t>(s.mat.num_groups));
    for (int g = 0; g < s.mat.num_groups; ++g) {
        const Color     gc = static_cast<Color>(s.mat.groups[g].color);
        const PieceType gt = static_cast<PieceType>(s.mat.groups[g].type);

        std::vector<Piece> rem;
        bool removed = false;
        for (const Piece& p : extras) {
            if (!removed && p.color == gc && p.type == gt) { removed = true; continue; }
            rem.push_back(p);
        }

        SweepSetup::HostSub sub;
        if (rem.empty()) {
            sub.draw = 1;  // capture reaches bare KK
        } else {
            CombIndex sub_idx(rem);
            Table t = solve_sweep_comb(sub_idx);
            sub.value = std::move(t.value);
            DeviceKingTable tmp_kt;  // discarded; sub encode uses the main kt
            sub_idx.fill_device(tmp_kt, sub.mat);
        }
        s.subs[static_cast<std::size_t>(g)] = std::move(sub);
    }

    return s;
}

std::vector<int16_t> sweep_host_reference(const CombIndex& idx, int* passes_out) {
    slider::init();
    slider::DeviceSliders sliders;
    slider::get_device_sliders(sliders);

    SweepSetup s = build_sweep_setup(idx);
    std::vector<DeviceSub> subs(static_cast<std::size_t>(s.mat.num_groups));
    s.device_subs(subs.data());

    // Jacobi ping-pong to a fixpoint — the exact shape the GPU driver runs.
    std::vector<int16_t> v_old = s.v_init;
    std::vector<int16_t> v_new = s.v_init;
    bool changed = true;
    int  passes  = 0;
    while (changed) {
        changed = false;
        for (std::size_t i = 0; i < s.N; ++i) {
            if (s.state[i] == SW_SOLVE) {
                const int16_t nv =
                    sweep_update(i, v_old.data(), s.kt, s.mat, sliders, subs.data());
                v_new[i] = nv;
                if (nv != v_old[i]) changed = true;
            } else {
                v_new[i] = v_old[i];
            }
        }
        std::swap(v_old, v_new);
        if (++passes > MATE) break;  // safety; converges long before this
    }
    if (passes_out) *passes_out = passes;
    return v_old;
}

}  // namespace tb
