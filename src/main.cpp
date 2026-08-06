// Chess engine driver.
//
// Phase 1 exposes a perft CLI — the tool used to validate move generation:
//   chess perft  <depth> [FEN...]   leaf-node count at depth
//   chess divide <depth> [FEN...]   per-root-move subtotals (debugging)
// FEN may be given as the remaining args (unquoted, space-separated); if
// omitted, the standard start position is used.

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "bitboard.hpp"
#include "eval.hpp"
#include "movegen.hpp"
#include "perft.hpp"
#include "position.hpp"
#include "search.hpp"
#include "slider.hpp"
#include "tb_index.hpp"
#include "tb_solve.hpp"
#include "types.hpp"

namespace {

constexpr const char* START_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// Join argv[first..argc) into a single space-separated FEN string.
std::string join_fen(int argc, char** argv, int first) {
    if (first >= argc) return START_FEN;
    std::string fen = argv[first];
    for (int i = first + 1; i < argc; ++i) {
        fen += ' ';
        fen += argv[i];
    }
    return fen;
}

int run_perft(int argc, char** argv, bool divide) {
    const int depth = std::atoi(argv[2]);
    const std::string fen = join_fen(argc, argv, 3);

    Position pos;
    if (!set_fen(pos, fen)) {
        std::fprintf(stderr, "invalid FEN: %s\n", fen.c_str());
        return 1;
    }

    const auto t0 = std::chrono::steady_clock::now();
    uint64_t nodes = divide ? perft::divide(pos, depth) : perft::perft(pos, depth);
    const auto t1 = std::chrono::steady_clock::now();

    const double secs = std::chrono::duration<double>(t1 - t0).count();
    if (!divide) std::printf("perft(%d) = %llu\n", depth, static_cast<unsigned long long>(nodes));
    const double nps = secs > 0 ? static_cast<double>(nodes) / secs : 0.0;
    std::printf("time %.3fs   %.2f Mnps\n", secs, nps / 1e6);
    return 0;
}

bool letter_to_type(char c, PieceType& pt) {
    switch (std::toupper(static_cast<unsigned char>(c))) {
        case 'Q': pt = PieceType::Queen;  return true;
        case 'R': pt = PieceType::Rook;   return true;
        case 'B': pt = PieceType::Bishop; return true;
        case 'N': pt = PieceType::Knight; return true;
        default: return false;
    }
}

const char* type_letter(PieceType pt) {
    switch (pt) {
        case PieceType::Queen:  return "Q";
        case PieceType::Rook:   return "R";
        case PieceType::Bishop: return "B";
        case PieceType::Knight: return "N";
        default:                return "?";
    }
}

// Parse a tablebase material argument into the extra-piece list.
//   Legacy 3-man form: a single letter "Q|R|B|N" == KXK (a White piece).
//   General form: a Nalimov material name "KQKR", "KRKN", "KQK", ... where the
//   pieces before the second K are White's and those after are Black's.
// Rejects materials this engine's indexer cannot yet handle (see comments).
bool parse_material(const char* s, std::vector<tb::Piece>& extras, std::string& name) {
    if (!s || s[0] == '\0') return false;

    // Legacy single-letter KXK.
    if (s[1] == '\0') {
        PieceType pt;
        if (!letter_to_type(s[0], pt)) return false;
        extras.push_back({Color::White, pt});
        name = std::string("K") + type_letter(pt) + "K";
        return true;
    }

    std::string str(s);
    for (char& c : str) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if (str.size() < 3 || str[0] != 'K') return false;
    const std::size_t k2 = str.find('K', 1);
    if (k2 == std::string::npos) return false;
    if (str.find('K', k2 + 1) != std::string::npos) return false;  // exactly two kings

    for (std::size_t i = 1; i < k2; ++i) {
        PieceType pt;
        if (!letter_to_type(str[i], pt)) return false;
        extras.push_back({Color::White, pt});
    }
    for (std::size_t i = k2 + 1; i < str.size(); ++i) {
        PieceType pt;
        if (!letter_to_type(str[i], pt)) return false;
        extras.push_back({Color::Black, pt});
    }
    if (extras.empty() || extras.size() > 2) return false;  // 3- or 4-man only (direct index)

    // Distinct (color,type) required: identical pieces need unordered indexing.
    for (std::size_t a = 0; a < extras.size(); ++a)
        for (std::size_t b = a + 1; b < extras.size(); ++b)
            if (extras[a].color == extras[b].color && extras[a].type == extras[b].type) {
                std::fprintf(stderr, "tb: identical-piece materials (e.g. KRRK) unsupported yet\n");
                return false;
            }

    name = "K";
    for (const tb::Piece& p : extras) if (p.color == Color::White) name += type_letter(p.type);
    name += "K";
    for (const tb::Piece& p : extras) if (p.color == Color::Black) name += type_letter(p.type);
    return true;
}

int run_tb(int argc, char** argv) {
    std::vector<tb::Piece> extras;
    std::string name;
    if (argc < 3 || !parse_material(argv[2], extras, name)) {
        std::fprintf(stderr, "usage: chess tb <material>   e.g. Q, R, KQK, KQKR, KRKN\n");
        return 1;
    }

    tb::Index idx(std::move(extras));
    const tb::Table t = tb::solve_sweep(idx);

    std::printf("%-5s indices=%zu  W=%zu L=%zu D=%zu  maxWinDTM=%d plies (%d moves)\n",
                name.c_str(), idx.size(), t.wins, t.losses, t.draws,
                t.max_win_dtm, (t.max_win_dtm + 1) / 2);

    // Show a deepest-forced-mate position (the hardest win) as a concrete,
    // externally-verifiable FEN. Longest mate => smallest positive value.
    if (t.wins > 0) {
        std::size_t deepest = 0;
        for (std::size_t i = 0; i < t.value.size(); ++i)
            if (t.value[i] > 0 && tb::win_dtm(t.value[i]) == t.max_win_dtm) { deepest = i; break; }
        std::printf("deepest win (%d-ply mate): %s\n",
                    t.max_win_dtm, to_fen(idx.decode(deepest)).c_str());
    }
    return 0;
}

// ---- Interactive play + one-shot search ------------------------------------

// A position is illegal to search if the side NOT to move is in check: the
// previous move would have been illegal, and the engine (which only guards the
// mover's own king) would happily "capture" the exposed king and then evaluate a
// king-less board. Reject such FENs up front rather than crash.
bool position_legal(const Position& pos) {
    const Square their_king = movegen::king_square(pos, ~pos.side_to_move);
    return !movegen::is_attacked(pos, their_king, pos.side_to_move);
}

char piece_char(const Position& pos, Square s) {
    const PieceType t = pos.piece_type_on(s);
    if (t == PieceType::None) return '.';
    const char c = "PNBRQK"[type_index(t)];
    return pos.color_on(s) == Color::White ? c : static_cast<char>(std::tolower(c));
}

void print_board(const Position& pos) {
    for (int rank = 7; rank >= 0; --rank) {
        std::printf(" %d  ", rank + 1);
        for (int file = 0; file < 8; ++file)
            std::printf("%c ", piece_char(pos, make_square(file, rank)));
        std::printf("\n");
    }
    std::printf("    a b c d e f g h      %s to move\n",
                pos.side_to_move == Color::White ? "White" : "Black");
}

char promo_letter(PieceType pt) {
    switch (pt) {
        case PieceType::Knight: return 'n';
        case PieceType::Bishop: return 'b';
        case PieceType::Rook:   return 'r';
        default:                return 'q';
    }
}

// Match a user string ("e2e4", "e7e8q") to a legal move by from/to (+promotion).
// The flags (capture/castle/en-passant/promotion) come from the generated move,
// so the user never types them. A 4-char promotion defaults to queen.
Move match_user_move(const MoveList& ml, const std::string& in) {
    if (in.size() < 4) return MOVE_NONE;
    const int ff = in[0] - 'a', fr = in[1] - '1';
    const int tf = in[2] - 'a', tr = in[3] - '1';
    if (ff < 0 || ff > 7 || fr < 0 || fr > 7 || tf < 0 || tf > 7 || tr < 0 || tr > 7)
        return MOVE_NONE;
    const Square from = make_square(ff, fr), to = make_square(tf, tr);
    const char promo = in.size() >= 5 ? static_cast<char>(std::tolower(in[4])) : 0;

    Move fallback = MOVE_NONE;
    for (const Move m : ml) {
        if (move_from(m) != from || move_to(m) != to) continue;
        if (!is_promotion(m)) return m;
        const char letter = promo_letter(promotion_type(m));
        if (promo == 0) { if (letter == 'q') fallback = m; }  // default to queen
        else if (letter == promo) return m;
    }
    return fallback;
}

// Python's depth ladder: search deeper as material comes off the board.
int scaled_depth(const Position& pos, int base) {
    const int pieces = popcount(pos.occupied());
    if (pieces <= 4)  return base + 5;
    if (pieces <= 6)  return base + 4;
    if (pieces <= 8)  return base + 3;
    if (pieces <= 12) return base + 2;
    if (pieces <= 20) return base + 1;
    return base;
}

int run_search(int argc, char** argv) {
    const int depth = std::atoi(argv[2]);
    const std::string fen = join_fen(argc, argv, 3);
    Position pos;
    if (!set_fen(pos, fen)) { std::fprintf(stderr, "invalid FEN: %s\n", fen.c_str()); return 1; }
    if (!position_legal(pos)) {
        std::fprintf(stderr, "illegal position (side to move can capture the enemy king): %s\n", fen.c_str());
        return 1;
    }

    search::new_game();
    search::history_add(pos.zobrist);

    const auto t0 = std::chrono::steady_clock::now();
    const search::SearchResult r = search::find_best_move(pos, depth, /*verbose=*/true);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::printf("bestmove %s   score %+d (White's view)   nodes %llu   %.2fs   %.2f Mnps\n",
                move_to_uci(r.best_move).c_str(), r.score,
                static_cast<unsigned long long>(r.nodes), secs,
                secs > 0 ? r.nodes / secs / 1e6 : 0.0);
    return 0;
}

int run_play(int argc, char** argv) {
    // Optional [depth] then optional [FEN...]. Human plays White, engine Black
    // (matching the Python driver).
    int base_depth = 4;
    int fen_start = 2;
    if (argc >= 3 && std::isdigit(static_cast<unsigned char>(argv[2][0]))) {
        base_depth = std::atoi(argv[2]);
        fen_start = 3;
    }
    const std::string fen = join_fen(argc, argv, fen_start);
    Position pos;
    if (!set_fen(pos, fen)) { std::fprintf(stderr, "invalid FEN: %s\n", fen.c_str()); return 1; }
    if (!position_legal(pos)) {
        std::fprintf(stderr, "illegal position (side to move can capture the enemy king): %s\n", fen.c_str());
        return 1;
    }

    search::new_game();
    search::history_add(pos.zobrist);

    while (true) {
        std::printf("\n");
        print_board(pos);

        MoveList ml;
        movegen::generate_legal(pos, ml);
        if (ml.size() == 0) {
            if (movegen::in_check(pos))
                std::printf("Checkmate! %s wins.\n",
                            pos.side_to_move == Color::White ? "Black" : "White");
            else
                std::printf("Stalemate! Draw.\n");
            return 0;
        }

        if (pos.side_to_move == Color::White) {
            std::printf("Your move (e2e4, promo e7e8q): ");
            std::fflush(stdout);
            std::string line;
            if (!std::getline(std::cin, line)) { std::printf("\n(input closed)\n"); return 0; }
            // strip trailing whitespace/CR
            while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) line.pop_back();
            const Move m = match_user_move(ml, line);
            if (m == MOVE_NONE) { std::printf("Illegal / unparsed move.\n"); continue; }
            StateInfo st; make_move(pos, m, st);
            search::history_add(pos.zobrist);
        } else {
            const int depth = scaled_depth(pos, base_depth);
            std::printf("Bot thinking (depth %d)...\n", depth);
            const auto t0 = std::chrono::steady_clock::now();
            const search::SearchResult r = search::find_best_move(pos, depth, /*verbose=*/false);
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("  %s   score %+d   %.2fs   %llu nodes\n",
                        move_to_uci(r.best_move).c_str(), r.score, secs,
                        static_cast<unsigned long long>(r.nodes));
            StateInfo st; make_move(pos, r.best_move, st);
            search::history_add(pos.zobrist);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    slider::init();  // build magic tables once at startup

    if (argc >= 3 && std::strcmp(argv[1], "perft") == 0)  return run_perft(argc, argv, false);
    if (argc >= 3 && std::strcmp(argv[1], "divide") == 0) return run_perft(argc, argv, true);
    if (argc >= 2 && std::strcmp(argv[1], "tb") == 0)     return run_tb(argc, argv);
    if (argc >= 3 && std::strcmp(argv[1], "search") == 0) return run_search(argc, argv);
    if (argc >= 2 && std::strcmp(argv[1], "play") == 0)   return run_play(argc, argv);

    std::printf("ChessEngine v0.1.0\n");
    std::printf("usage:\n");
    std::printf("  chess perft  <depth> [FEN...]\n");
    std::printf("  chess divide <depth> [FEN...]\n");
    std::printf("  chess tb     <material>   e.g. Q, R, KQK, KQKR, KRKN\n");
    std::printf("  chess search <depth> [FEN...]      best move for a position\n");
    std::printf("  chess play   [depth] [FEN...]      play vs the engine (you are White)\n");
    return 0;
}
