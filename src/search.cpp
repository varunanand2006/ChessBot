#include "search.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <unordered_map>

#include "eval.hpp"
#include "movegen.hpp"
#include "tb_probe.hpp"

namespace search {

namespace {

// Score bounds. INF fences alpha/beta; MATE is the checkmate magnitude (the
// Python engine's 99999). Both comfortably fit int and never overflow when the
// small per-ply mate offsets are added.
constexpr int INF  = 1'000'000;
constexpr int MATE = 99'999;

// Quiescence recursion cap (Python default qdepth=3).
constexpr int QDEPTH = 3;

// Delta-pruning margin: a queen. If even a full queen swing can't reach the
// window, stand pat (Python used 900).
constexpr int DELTA_MARGIN = 900;

// --- Transposition table (game-global, persists across moves) ----------------
enum TTFlag : uint8_t { TT_EXACT = 0, TT_LOWER_BOUND = 1, TT_UPPER_BOUND = 2 };

struct TTEntry {
    int     depth;
    int     score;
    uint8_t flag;
    Move    best_move;
};

std::unordered_map<uint64_t, TTEntry> g_tt;

// Repetition history: count of every Zobrist key seen along the game line plus
// the current search path. Mirrors the Python board.position_history dict, which
// make/undo bump up and down — so the search sees repetitions inside its own
// tree, not just in the played game.
std::unordered_map<uint64_t, int> g_history;

uint64_t g_nodes = 0;
bool g_use_tb = false;

// If tablebase probing is on and `pos` is a supported material, set white_score
// to the exact result on the search's White-relative mate scale and return true.
// Converting here (not in tb_probe) keeps the probe module free of search
// constants. Shorter mates score higher in magnitude, matching the search's own
// mate scores, so alpha-beta orders TB results correctly.
bool maybe_probe(const Position& pos, int& white_score) {
    if (!g_use_tb) return false;
    const tb::ProbeResult r = tb::probe(pos);
    if (!r.found) return false;

    int stm_score;
    if (r.wdl == 0)      stm_score = 0;
    else if (r.wdl > 0)  stm_score = MATE - r.dtm;   // side to move mates in r.dtm plies
    else                 stm_score = -MATE + r.dtm;  // side to move mated in r.dtm plies
    white_score = (pos.side_to_move == Color::White) ? stm_score : -stm_score;
    return true;
}

int history_count(uint64_t key) {
    const auto it = g_history.find(key);
    return it == g_history.end() ? 0 : it->second;
}

// Returns true and sets `out_score` when the stored bound is usable at this
// depth/window. Always exposes the stored move (out_move) for ordering.
bool tt_lookup(uint64_t key, int depth, int alpha, int beta,
               int& out_score, Move& out_move) {
    const auto it = g_tt.find(key);
    if (it == g_tt.end()) { out_move = MOVE_NONE; return false; }

    const TTEntry& e = it->second;
    out_move = e.best_move;
    if (e.depth >= depth) {
        if (e.flag == TT_EXACT) { out_score = e.score; return true; }
        if (e.flag == TT_LOWER_BOUND && e.score >= beta)  { out_score = e.score; return true; }
        if (e.flag == TT_UPPER_BOUND && e.score <= alpha) { out_score = e.score; return true; }
    }
    return false;
}

void tt_store(uint64_t key, int depth, int score, uint8_t flag, Move best_move) {
    g_tt[key] = TTEntry{depth, score, flag, best_move};
}

// --- Move ordering -----------------------------------------------------------
// MVV-LVA for captures (+ promotion bonus), matching Python score_move. Uses
// the board only for the victim/attacker piece types. En-passant scores as a
// quiet here (its victim square is empty) exactly as the Python original did.
int score_move(const Position& pos, Move m) {
    int score = 0;
    const PieceType victim = pos.piece_type_on(move_to(m));
    if (victim != PieceType::None) {
        const PieceType attacker = pos.piece_type_on(move_from(m));
        score += 10000 + (eval::piece_value(victim) * 10 - eval::piece_value(attacker));
    }
    if (is_promotion(m))
        score += eval::piece_value(promotion_type(m));
    return score;
}

// Order `ml` in place: `first` (TT or previous-best move) leads if present, then
// the remainder by descending MVV-LVA score. A small stack buffer of
// (score, move) pairs is sorted — no heap, no allocation on the search path.
void order_moves(const Position& pos, MoveList& ml, Move first) {
    std::array<std::pair<int, Move>, 256> scored;
    const int n = ml.size();
    for (int i = 0; i < n; ++i) {
        const Move m = ml[i];
        const int s = (m == first && first != MOVE_NONE) ? INF : score_move(pos, m);
        scored[static_cast<std::size_t>(i)] = {s, m};
    }
    std::stable_sort(scored.begin(), scored.begin() + n,
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    for (int i = 0; i < n; ++i) ml.moves[i] = scored[static_cast<std::size_t>(i)].second;
}

// --- Quiescence --------------------------------------------------------------
int quiescence(Position& pos, int alpha, int beta, int qdepth) {
    ++g_nodes;
    int tb_score;
    if (maybe_probe(pos, tb_score)) return tb_score;  // exact endgame result
    if (qdepth == 0) return eval::evaluate(pos);

    const int stand_pat = eval::evaluate(pos);

    // Side-aware stand-pat + delta pruning (scores are White-relative).
    if (pos.side_to_move == Color::White) {
        if (stand_pat + DELTA_MARGIN < alpha) return alpha;
        if (stand_pat >= beta) return beta;
        if (stand_pat > alpha) alpha = stand_pat;
    } else {
        if (stand_pat - DELTA_MARGIN > beta) return beta;
        if (stand_pat <= alpha) return alpha;
        if (stand_pat < beta) beta = stand_pat;
    }

    MoveList ml;
    movegen::generate_legal(pos, ml);
    if (ml.size() == 0) {
        if (movegen::in_check(pos)) return pos.side_to_move == Color::White ? -MATE : MATE;
        return 0;  // stalemate
    }

    // Captures (incl. en passant / promotion-captures) only, best first.
    MoveList caps;
    for (const Move m : ml) if (is_capture(m)) caps.add(m);
    if (caps.size() == 0) return stand_pat;  // quiet position: rely on stand-pat
    order_moves(pos, caps, MOVE_NONE);

    if (pos.side_to_move == Color::White) {
        for (const Move m : caps) {
            StateInfo st; make_move(pos, m, st);
            const int score = quiescence(pos, alpha, beta, qdepth - 1);
            unmake_move(pos, m, st);
            if (score >= beta) return beta;
            if (score > alpha) alpha = score;
        }
        return alpha;
    } else {
        for (const Move m : caps) {
            StateInfo st; make_move(pos, m, st);
            const int score = quiescence(pos, alpha, beta, qdepth - 1);
            unmake_move(pos, m, st);
            if (score <= alpha) return alpha;
            if (score < beta) beta = score;
        }
        return beta;
    }
}

// --- Minimax with alpha-beta + TT --------------------------------------------
int minimax(Position& pos, int depth, int alpha, int beta) {
    ++g_nodes;
    const int original_alpha = alpha;
    const int original_beta  = beta;

    // Threefold repetition: this position already seen twice -> draw. Checked
    // before the tablebase probe: a repetition is a draw regardless of a DTM win.
    if (history_count(pos.zobrist) >= 2) return 0;

    // Exact endgame result, if available, cuts off the search here.
    int tb_score;
    if (maybe_probe(pos, tb_score)) return tb_score;

    int tt_score;
    Move tt_move;
    if (tt_lookup(pos.zobrist, depth, alpha, beta, tt_score, tt_move))
        return tt_score;

    if (depth == 0) return quiescence(pos, alpha, beta, QDEPTH);

    MoveList ml;
    movegen::generate_legal(pos, ml);
    if (ml.size() == 0) {
        if (movegen::in_check(pos))
            return pos.side_to_move == Color::White ? (-MATE - depth) : (MATE + depth);
        return 0;  // stalemate
    }

    order_moves(pos, ml, tt_move);
    Move best_move = ml[0];

    if (pos.side_to_move == Color::White) {
        int max_score = -INF;
        for (const Move m : ml) {
            StateInfo st; make_move(pos, m, st);
            ++g_history[pos.zobrist];
            const int score = minimax(pos, depth - 1, alpha, beta);
            --g_history[pos.zobrist];
            unmake_move(pos, m, st);
            if (score > max_score) { max_score = score; best_move = m; }
            if (score > alpha) alpha = score;
            if (beta <= alpha) break;
        }
        const uint8_t flag = (max_score <= original_alpha) ? TT_UPPER_BOUND
                           : (max_score >= beta)           ? TT_LOWER_BOUND
                                                           : TT_EXACT;
        tt_store(pos.zobrist, depth, max_score, flag, best_move);
        return max_score;
    } else {
        int min_score = INF;
        for (const Move m : ml) {
            StateInfo st; make_move(pos, m, st);
            ++g_history[pos.zobrist];
            const int score = minimax(pos, depth - 1, alpha, beta);
            --g_history[pos.zobrist];
            unmake_move(pos, m, st);
            if (score < min_score) { min_score = score; best_move = m; }
            if (score < beta) beta = score;
            if (beta <= alpha) break;
        }
        const uint8_t flag = (min_score >= original_beta) ? TT_LOWER_BOUND
                           : (min_score <= alpha)         ? TT_UPPER_BOUND
                                                          : TT_EXACT;
        tt_store(pos.zobrist, depth, min_score, flag, best_move);
        return min_score;
    }
}

}  // namespace

void new_game() {
    g_tt.clear();
    g_history.clear();
}

void set_use_tablebase(bool on) { g_use_tb = on; }

void history_add(uint64_t key) { ++g_history[key]; }

SearchResult find_best_move(Position& pos, int max_depth, bool verbose) {
    g_nodes = 0;
    Move best_move  = MOVE_NONE;
    int  best_score = 0;

    for (int depth = 1; depth <= max_depth; ++depth) {
        MoveList ml;
        movegen::generate_legal(pos, ml);
        if (ml.size() == 0) return {MOVE_NONE, 0, depth, g_nodes};

        // Previous iteration's best leads; fall back to the TT move. The root is
        // never stored in the TT, so best_move from the last completed depth is
        // the reliable ordering seed.
        int dummy;
        Move tt_move;
        tt_lookup(pos.zobrist, depth, -INF, INF, dummy, tt_move);
        const Move priority = (best_move != MOVE_NONE) ? best_move : tt_move;
        order_moves(pos, ml, priority);

        int alpha = -INF, beta = INF;
        best_move = ml[0];

        if (pos.side_to_move == Color::White) {
            best_score = -INF;
            for (const Move m : ml) {
                StateInfo st; make_move(pos, m, st);
                ++g_history[pos.zobrist];
                const int score = minimax(pos, depth - 1, alpha, beta);
                --g_history[pos.zobrist];
                unmake_move(pos, m, st);
                if (score > best_score) { best_score = score; best_move = m; }
                alpha = std::max(alpha, score);
            }
        } else {
            best_score = INF;
            for (const Move m : ml) {
                StateInfo st; make_move(pos, m, st);
                ++g_history[pos.zobrist];
                const int score = minimax(pos, depth - 1, alpha, beta);
                --g_history[pos.zobrist];
                unmake_move(pos, m, st);
                if (score < best_score) { best_score = score; best_move = m; }
                beta = std::min(beta, score);
            }
        }

        if (verbose)
            std::printf("  depth %2d -> score %+6d   best: %s\n",
                        depth, best_score, move_to_uci(best_move).c_str());
    }

    return {best_move, best_score, max_depth, g_nodes};
}

}  // namespace search
