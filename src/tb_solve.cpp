#include "tb_solve.hpp"

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <optional>
#include <queue>

#include "movegen.hpp"
#include "position.hpp"

namespace tb {

namespace {

// Which retrograde algorithm to run. Both share the same graph builder and the
// same material-DAG recursion, so their agreement is a real cross-check and not
// two views of one implementation.
enum class Method { Sweep, Bfs };

Table solve_material(const Index& idx, Method method);

// A solved capture-exit sub-table. A 4-man capture removes one extra piece and
// lands in a 3-man material; a 3-man capture removes its only extra and lands in
// bare KK (always a draw). `removed` is the extra SLOT that this exit deletes.
//   draw == true   -> exit is KK, boundary value 0 (no table built).
//   draw == false  -> `idx`/`value` hold the solved sub-table to probe.
struct SubTable {
    int                  removed = -1;
    bool                 draw = false;
    std::optional<Index> idx;
    std::vector<int16_t> value;
};

// Compressed forward move graph over the index space.
//   For edge e in [start[i], start[i+1]):
//     target[e] >= 0 : an in-table child index (same material).
//     target[e] == -1: a capture-exit; boundary[e] is the child's value, taken
//                       from the child's side-to-move perspective (0 for a KK
//                       draw, or the probed DTM of a solved sub-table).
//   terminal[i]: 0 = has moves, 1 = checkmate (loss in 0), 2 = stalemate (draw).
struct Graph {
    std::vector<int32_t> start;
    std::vector<int32_t> target;
    std::vector<int16_t> boundary;
    std::vector<uint8_t> terminal;
};

Graph build_graph(const Index& idx, const std::vector<SubTable>& subs) {
    const std::size_t N = idx.size();
    const std::vector<Piece>& extras = idx.extras();

    Graph g;
    g.start.resize(N + 1);
    g.terminal.resize(N);
    g.start[0] = 0;

    for (std::size_t i = 0; i < N; ++i) {
        Position pos = idx.decode(i);
        MoveList ml;
        movegen::generate_legal(pos, ml);

        if (ml.size() == 0) {
            g.terminal[i] = movegen::in_check(pos) ? 1 : 2;
            g.start[i + 1] = g.start[i];
            continue;
        }
        g.terminal[i] = 0;
        for (const Move m : ml) {
            StateInfo st;
            make_move(pos, m, st);

            // A capture removes exactly one extra (kings are never captured), so
            // detect which extra's (color,type) bitboard went empty. Distinct
            // (color,type) per extra makes this unambiguous.
            int captured = -1;
            for (std::size_t e = 0; e < extras.size(); ++e) {
                const Bitboard bb = pos.by_type[type_index(extras[e].type)] &
                                    pos.by_color[color_index(extras[e].color)];
                if (bb == 0) { captured = static_cast<int>(e); break; }
            }

            int32_t tgt;
            int16_t bnd = 0;
            if (captured < 0) {
                tgt = static_cast<int32_t>(idx.encode(pos));  // same material
            } else {
                tgt = -1;  // capture-exit -> boundary value
                const SubTable* s = nullptr;
                for (const SubTable& c : subs)
                    if (c.removed == captured) { s = &c; break; }
                // KK exits stay 0; otherwise probe the solved sub-table.
                if (!s->draw) bnd = s->value[s->idx->encode(pos)];
            }
            unmake_move(pos, m, st);
            g.target.push_back(tgt);
            g.boundary.push_back(bnd);
        }
        g.start[i + 1] = static_cast<int32_t>(g.target.size());
    }
    return g;
}

// Move a mate score one ply further from mate (toward zero).
int age(int x) { return x > 0 ? x - 1 : (x < 0 ? x + 1 : 0); }

void fill_stats(Table& t) {
    for (const int16_t v : t.value) {
        if (v > 0) { ++t.wins;  if (win_dtm(v)  > t.max_win_dtm)  t.max_win_dtm  = win_dtm(v); }
        else if (v < 0) { ++t.losses; if (loss_dtm(v) > t.max_loss_dtm) t.max_loss_dtm = loss_dtm(v); }
        else ++t.draws;
    }
}

// ---- Core solver 1: iterated negamax sweep (the GPU dense-sweep shape) -------

Table run_sweep(const Index& idx, const Graph& g) {
    const std::size_t N = idx.size();

    std::vector<int16_t> v(N, 0);
    for (std::size_t i = 0; i < N; ++i)
        if (g.terminal[i] == 1) v[i] = static_cast<int16_t>(-MATE);  // checkmate
        // terminal==2 (stalemate) and non-terminal start at 0 (draw/unknown)

    // Iterate to a fixpoint. Each non-terminal is recomputed as the best (max)
    // aged negation of a child's value; capture-exits contribute their fixed
    // boundary value. Terminals are pinned.
    bool changed = true;
    int passes = 0;
    while (changed) {
        changed = false;
        for (std::size_t i = 0; i < N; ++i) {
            if (g.terminal[i]) continue;
            int best = -(MATE + 1);
            for (int32_t e = g.start[i]; e < g.start[i + 1]; ++e) {
                const int32_t t = g.target[e];
                const int c = (t < 0) ? g.boundary[e] : v[t];
                const int m = age(-c);
                if (m > best) best = m;
            }
            if (static_cast<int16_t>(best) != v[i]) {
                v[i] = static_cast<int16_t>(best);
                changed = true;
            }
        }
        if (++passes > MATE) break;  // safety; must converge well before this
    }

    Table t;
    t.value = std::move(v);
    t.passes = passes;
    fill_stats(t);
    return t;
}

// ---- Core solver 2: min-priority-queue retrograde over the reverse graph -----
//
// A position is finalized in nondecreasing DTM order. Capture-exits are folded
// in as fixed successor values at seed time, so a losing exit becomes a queued
// WIN candidate and a winning exit is a pre-counted opponent-win. A min-heap
// (not the 3-man FIFO) is required because those exits inject results at
// arbitrary DTM, out of natural discovery order.
struct Event {
    int32_t dtm;
    int32_t pos;
    bool    win;  // finalized value is a WIN for `pos` (else a LOSS)
};
bool operator>(const Event& a, const Event& b) { return a.dtm > b.dtm; }

Table run_bfs(const Index& idx, const Graph& g) {
    const std::size_t N = idx.size();

    // Reverse in-table edges (predecessors), CSR. Capture-exits have no reverse
    // edge; they are handled entirely at seed time below.
    std::vector<int32_t> pred_start(N + 1, 0);
    for (std::size_t i = 0; i < N; ++i)
        for (int32_t e = g.start[i]; e < g.start[i + 1]; ++e)
            if (g.target[e] >= 0) ++pred_start[g.target[e] + 1];
    for (std::size_t i = 0; i < N; ++i) pred_start[i + 1] += pred_start[i];
    std::vector<int32_t> pred(pred_start[N]);
    {
        std::vector<int32_t> cur(pred_start.begin(), pred_start.end());
        for (std::size_t i = 0; i < N; ++i)
            for (int32_t e = g.start[i]; e < g.start[i + 1]; ++e)
                if (g.target[e] >= 0) pred[cur[g.target[e]]++] = static_cast<int32_t>(i);
    }

    std::vector<int16_t> v(N, 0);            // 0 => draw/unknown
    std::vector<uint8_t> done(N, 0);
    std::vector<int32_t> nsucc(N, 0);        // total successors (all edges)
    std::vector<int32_t> opp_win_count(N, 0);// successors that are opponent-wins
    std::vector<int32_t> max_opp_dtm(N, -1); // slowest such opponent-win (for loss DTM)
    std::priority_queue<Event, std::vector<Event>, std::greater<Event>> pq;

    // Seed. A position is LOST only when EVERY successor is an opponent-win, so
    // any drawing/unresolved successor keeps opp_win_count below nsucc forever
    // and blocks a false loss — no explicit "draw block" flag needed.
    for (std::size_t i = 0; i < N; ++i) {
        if (g.terminal[i] == 1) { pq.push({0, static_cast<int32_t>(i), false}); continue; }
        if (g.terminal[i] == 2) { done[i] = 1; continue; }  // stalemate draw, pinned
        nsucc[i] = g.start[i + 1] - g.start[i];
        for (int32_t e = g.start[i]; e < g.start[i + 1]; ++e) {
            if (g.target[e] >= 0) continue;         // in-table child, via propagation
            const int16_t cv = g.boundary[e];       // capture-exit child value
            if (cv < 0) {                            // child loses -> capturing wins
                pq.push({loss_dtm(cv) + 1, static_cast<int32_t>(i), true});
            } else if (cv > 0) {                     // child wins -> opponent-win move
                ++opp_win_count[i];
                if (win_dtm(cv) > max_opp_dtm[i]) max_opp_dtm[i] = win_dtm(cv);
            }
            // cv == 0 (draw exit): counts in nsucc, never an opponent-win.
        }
    }
    // Positions whose successors are ALL opponent-wins purely via capture-exits.
    for (std::size_t i = 0; i < N; ++i)
        if (!done[i] && g.terminal[i] == 0 && nsucc[i] > 0 && opp_win_count[i] == nsucc[i])
            pq.push({max_opp_dtm[i] + 1, static_cast<int32_t>(i), false});

    while (!pq.empty()) {
        const Event ev = pq.top(); pq.pop();
        const int32_t i = ev.pos;
        if (done[i]) continue;
        done[i] = 1;
        v[i] = ev.win ? static_cast<int16_t>(MATE - ev.dtm)
                      : static_cast<int16_t>(ev.dtm - MATE);

        for (int32_t e = pred_start[i]; e < pred_start[i + 1]; ++e) {
            const int32_t Q = pred[e];
            if (done[Q]) continue;
            if (!ev.win) {
                // i is a LOSS: Q moves into it and WINS in ev.dtm+1.
                pq.push({ev.dtm + 1, Q, true});
            } else {
                // i is a WIN: one more opponent-win successor of Q. Q is LOST
                // once all of them are, with DTM set by the slowest (max).
                ++opp_win_count[Q];
                if (ev.dtm > max_opp_dtm[Q]) max_opp_dtm[Q] = ev.dtm;
                if (opp_win_count[Q] == nsucc[Q])
                    pq.push({max_opp_dtm[Q] + 1, Q, false});
            }
        }
    }
    // Anything still !done stays 0 (draw).

    Table t;
    t.value = std::move(v);
    fill_stats(t);
    return t;
}

// ---- Material-DAG recursion --------------------------------------------------

// Solve `idx`, first solving every capture-exit sub-material (recursively, with
// the SAME method) so captures probe real DTM boundary values. This is the DAG:
// KQKR depends on KQK and KRK; each 3-man depends only on KK (a draw).
Table solve_material(const Index& idx, Method method) {
    const std::vector<Piece>& extras = idx.extras();

    std::vector<SubTable> subs;
    subs.reserve(extras.size());
    for (int e = 0; e < static_cast<int>(extras.size()); ++e) {
        SubTable s;
        s.removed = e;
        std::vector<Piece> rem;
        for (int k = 0; k < static_cast<int>(extras.size()); ++k)
            if (k != e) rem.push_back(extras[static_cast<std::size_t>(k)]);

        if (rem.empty()) {
            s.draw = true;  // exit to bare KK
        } else {
            Index sub(std::move(rem));
            Table st = solve_material(sub, method);
            s.value = std::move(st.value);
            s.idx.emplace(std::move(sub));
        }
        subs.push_back(std::move(s));
    }

    const Graph g = build_graph(idx, subs);
    return (method == Method::Sweep) ? run_sweep(idx, g) : run_bfs(idx, g);
}

}  // namespace

Table solve_sweep(const Index& idx) { return solve_material(idx, Method::Sweep); }
Table solve_bfs(const Index& idx)   { return solve_material(idx, Method::Bfs); }

}  // namespace tb
