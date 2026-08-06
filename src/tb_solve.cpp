#include "tb_solve.hpp"

#include <queue>

#include "movegen.hpp"
#include "position.hpp"

namespace tb {

namespace {

// Compressed forward move graph over the index space.
//   target[start[i] .. start[i+1]) are the child indices of position i.
//   A target of -1 means the move leaves the material class (the rook is
//   captured -> KK), which is a draw.
//   terminal[i]: 0 = has moves, 1 = checkmate (loss in 0), 2 = stalemate (draw).
struct Graph {
    std::vector<int32_t> start;
    std::vector<int32_t> target;
    std::vector<uint8_t> terminal;
};

Graph build_graph(const Index& idx) {
    const std::size_t N = idx.size();
    const int wp_type = type_index(idx.white_piece());
    const int white   = color_index(Color::White);

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
            int32_t tgt;
            if ((pos.by_type[wp_type] & pos.by_color[white]) == 0) {
                tgt = -1;  // white piece captured -> KK draw
            } else {
                tgt = static_cast<int32_t>(idx.encode(pos));
            }
            unmake_move(pos, m, st);
            g.target.push_back(tgt);
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

}  // namespace

Table solve_sweep(const Index& idx) {
    const Graph g = build_graph(idx);
    const std::size_t N = idx.size();

    std::vector<int16_t> v(N, 0);
    for (std::size_t i = 0; i < N; ++i) {
        if (g.terminal[i] == 1) v[i] = static_cast<int16_t>(-MATE);  // checkmate
        // terminal==2 (stalemate) and non-terminal start at 0 (draw/unknown)
    }

    // Iterate to a fixpoint. Non-terminal positions are recomputed each pass as
    // the best (max) aged negation of a child's value; terminals are fixed.
    bool changed = true;
    int passes = 0;
    while (changed) {
        changed = false;
        for (std::size_t i = 0; i < N; ++i) {
            if (g.terminal[i]) continue;
            int best = -(MATE + 1);
            for (int32_t e = g.start[i]; e < g.start[i + 1]; ++e) {
                const int32_t t = g.target[e];
                const int c = (t < 0) ? 0 : v[t];  // capture -> KK draw
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

Table solve_bfs(const Index& idx) {
    const Graph g = build_graph(idx);
    const std::size_t N = idx.size();

    // Total successor count per position (includes capture/draw edges: those
    // permanently block a LOSS label, since they never become wins).
    std::vector<int32_t> nsucc(N, 0);
    for (std::size_t i = 0; i < N; ++i)
        nsucc[i] = g.terminal[i] ? 0 : (g.start[i + 1] - g.start[i]);

    // Reverse edges (predecessors), CSR, over in-table targets only.
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

    std::vector<int16_t> v(N, 0);       // 0 => draw/unknown
    std::vector<uint8_t> done(N, 0);
    std::vector<int32_t> win_count(N, 0);
    std::queue<int32_t> q;

    // Seed: checkmates are losses in 0. Stalemates are terminal draws (marked
    // done so they are never relabeled, but they emit no win/loss signal).
    for (std::size_t i = 0; i < N; ++i) {
        if (g.terminal[i] == 1) { v[i] = static_cast<int16_t>(-MATE); done[i] = 1; q.push(static_cast<int32_t>(i)); }
        else if (g.terminal[i] == 2) { done[i] = 1; }
    }

    while (!q.empty()) {
        const int32_t p = q.front(); q.pop();
        const bool p_loss = v[p] < 0;
        const int  d = p_loss ? loss_dtm(v[p]) : win_dtm(v[p]);

        for (int32_t e = pred_start[p]; e < pred_start[p + 1]; ++e) {
            const int32_t Q = pred[e];
            if (done[Q]) continue;
            if (p_loss) {
                // Q can move into a lost position -> Q wins in d+1.
                v[Q] = static_cast<int16_t>(MATE - (d + 1));
                done[Q] = 1;
                q.push(Q);
            } else {
                // p is a win for its mover; it is one more winning successor of Q.
                if (++win_count[Q] == nsucc[Q]) {
                    // Every successor of Q is a win for the opponent -> Q is lost,
                    // and the slowest such win (largest d, processed last) sets DTM.
                    v[Q] = static_cast<int16_t>((d + 1) - MATE);
                    done[Q] = 1;
                    q.push(Q);
                }
            }
        }
    }
    // Anything still !done stays 0 (draw).

    Table t;
    t.value = std::move(v);
    fill_stats(t);
    return t;
}

}  // namespace tb
