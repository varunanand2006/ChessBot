// Memoryless dense retrograde sweep over a combinatorial index. See the header
// note for why this exists (5-man makes the materialized-graph solvers explode);
// this stores only the value array and regenerates moves each pass — the exact
// shape the CUDA kernel will implement, and the CPU baseline it must beat.

#include "tb_solve.hpp"

#include <cstdio>
#include <cstdlib>
#include <optional>

#include "movegen.hpp"
#include "position.hpp"
#include "tb_index.hpp"

namespace tb {

namespace {

// A capture-exit sub-table: the material with one extra removed, solved densely.
struct Sub {
    int                  removed = -1;
    bool                 draw = false;   // exit to bare KK
    std::optional<Index> idx;
    std::vector<int16_t> value;
};

// Per-position state precomputed once, so the sweep touches only live nodes and
// never re-runs legality/terminal classification each pass.
enum : uint8_t { ST_ILLEGAL = 0, ST_SOLVE = 1, ST_MATE = 2, ST_STALE = 3 };

}  // namespace

Table solve_sweep_comb(const CombIndex& idx) {
    const std::vector<Piece>& extras = idx.extras();

    // Distinct (color,type) required: capture detection below identifies the
    // removed piece by a bitboard going empty, which is ambiguous for duplicates.
    for (std::size_t a = 0; a < extras.size(); ++a)
        for (std::size_t b = a + 1; b < extras.size(); ++b)
            if (extras[a].color == extras[b].color && extras[a].type == extras[b].type) {
                std::fprintf(stderr, "solve_sweep_comb: duplicate pieces unsupported\n");
                std::abort();
            }

    // Build capture-exit sub-tables (dense, via the materialized solver — they
    // are <=4-man so their graphs are affordable).
    std::vector<Sub> subs;
    subs.reserve(extras.size());
    for (int e = 0; e < static_cast<int>(extras.size()); ++e) {
        Sub s;
        s.removed = e;
        std::vector<Piece> rem;
        for (int k = 0; k < static_cast<int>(extras.size()); ++k)
            if (k != e) rem.push_back(extras[static_cast<std::size_t>(k)]);
        if (rem.empty()) {
            s.draw = true;  // 3-man capture -> bare KK draw
        } else {
            Index sub(std::move(rem));
            Table st = solve_sweep(sub);
            s.value = std::move(st.value);
            s.idx.emplace(std::move(sub));
        }
        subs.push_back(std::move(s));
    }

    const std::size_t N = idx.size();
    std::vector<uint8_t> state(N, ST_ILLEGAL);
    std::vector<int16_t> v(N, 0);

    // Pass 0: classify every index once. Illegal indices (the combinatorial
    // space is deliberately loose) are pinned dead; checkmates/stalemates are
    // pinned to their terminal values; the rest are live nodes to iterate.
    int sq[CombIndex::kMaxMen];
    Color stm;
    for (std::size_t i = 0; i < N; ++i) {
        idx.decode(i, sq, &stm);
        if (!Index::legal(extras, sq, stm)) continue;  // ST_ILLEGAL
        Position pos = idx.make_position(sq, stm, /*with_zobrist=*/false);  // only bitboards used
        MoveList ml;
        movegen::generate_legal(pos, ml);
        if (ml.size() == 0) {
            if (movegen::in_check(pos)) { state[i] = ST_MATE; v[i] = static_cast<int16_t>(-MATE); }
            else                        { state[i] = ST_STALE; }  // v stays 0
        } else {
            state[i] = ST_SOLVE;
        }
    }

    // Iterate to a fixpoint (Gauss-Seidel in place, like the dense sweep). Each
    // live node = max over legal moves of age(-child). Child values are looked
    // up in v (same material) or in a sub-table (capture-exit). No edge is ever
    // stored: the move list is regenerated here every pass.
    bool changed = true;
    int passes = 0;
    while (changed) {
        changed = false;
        for (std::size_t i = 0; i < N; ++i) {
            if (state[i] != ST_SOLVE) continue;
            Position pos = idx.decode_pos(i);
            MoveList ml;
            movegen::generate_legal(pos, ml);

            int best = -(MATE + 1);
            for (const Move m : ml) {
                StateInfo st;
                make_move(pos, m, st);

                int captured = -1;
                for (std::size_t e = 0; e < extras.size(); ++e) {
                    const Bitboard bb = pos.by_type[type_index(extras[e].type)] &
                                        pos.by_color[color_index(extras[e].color)];
                    if (bb == 0) { captured = static_cast<int>(e); break; }
                }

                int child;
                if (captured < 0) {
                    child = v[idx.encode(pos)];
                } else {
                    const Sub& s = subs[static_cast<std::size_t>(captured)];
                    child = s.draw ? 0 : s.value[s.idx->encode(pos)];
                }
                unmake_move(pos, m, st);

                const int cand = age(-child);
                if (cand > best) best = cand;
            }
            if (static_cast<int16_t>(best) != v[i]) { v[i] = static_cast<int16_t>(best); changed = true; }
        }
        if (++passes > MATE) break;  // safety; converges long before this
    }

    Table t;
    t.passes = passes;
    // Stats over LEGAL positions only: the combinatorial space is intentionally
    // loose, so illegal indices (pinned 0) must not be counted as draws.
    for (std::size_t i = 0; i < N; ++i) {
        if (state[i] == ST_ILLEGAL) continue;
        const int16_t val = v[i];
        if (val > 0)      { ++t.wins;   if (win_dtm(val)  > t.max_win_dtm)  t.max_win_dtm  = win_dtm(val); }
        else if (val < 0) { ++t.losses; if (loss_dtm(val) > t.max_loss_dtm) t.max_loss_dtm = loss_dtm(val); }
        else              { ++t.draws; }
    }
    t.value = std::move(v);
    return t;
}

}  // namespace tb
