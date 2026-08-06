#include "position.hpp"

#include <array>
#include <cctype>
#include <sstream>

#include "zobrist.hpp"

// ---------------------------------------------------------------------------
// Piece <-> FEN character mapping.
// ---------------------------------------------------------------------------
namespace {

// Index by PieceType; uppercase is White, lowercase is Black.
constexpr char PIECE_CHAR[NUM_PIECE_TYPES] = {'P', 'N', 'B', 'R', 'Q', 'K'};

// Decode a FEN piece letter. Returns false if not a piece character.
bool piece_from_char(char ch, PieceType& pt, Color& c) {
    c = std::isupper(static_cast<unsigned char>(ch)) ? Color::White : Color::Black;
    switch (std::tolower(static_cast<unsigned char>(ch))) {
        case 'p': pt = PieceType::Pawn;   return true;
        case 'n': pt = PieceType::Knight; return true;
        case 'b': pt = PieceType::Bishop; return true;
        case 'r': pt = PieceType::Rook;   return true;
        case 'q': pt = PieceType::Queen;  return true;
        case 'k': pt = PieceType::King;   return true;
        default:  return false;
    }
}

void put_piece(Position& pos, PieceType pt, Color c, Square s) {
    const Bitboard b = square_bb(s);
    pos.by_type[type_index(pt)]   |= b;
    pos.by_color[color_index(c)]  |= b;
}

}  // namespace

// ---------------------------------------------------------------------------
// Position helpers
// ---------------------------------------------------------------------------
void Position::clear() {
    *this = Position{};
    // Default member initializers already give an empty board with White to
    // move and fullmove 1; assign a fresh instance to reset every field.
}

PieceType Position::piece_type_on(Square s) const {
    const Bitboard b = square_bb(s);
    for (int i = 0; i < NUM_PIECE_TYPES; ++i) {
        if (by_type[i] & b) return static_cast<PieceType>(i);
    }
    return PieceType::None;
}

Color Position::color_on(Square s) const {
    const Bitboard b = square_bb(s);
    if (by_color[color_index(Color::White)] & b) return Color::White;
    if (by_color[color_index(Color::Black)] & b) return Color::Black;
    return Color::None;
}

// ---------------------------------------------------------------------------
// FEN parsing
// ---------------------------------------------------------------------------
bool set_fen(Position& pos, std::string_view fen) {
    Position tmp;  // parse into a scratch copy; only commit on full success
    tmp.clear();

    std::istringstream ss{std::string(fen)};
    std::string placement, side, castle, ep;

    // The first four fields are mandatory.
    if (!(ss >> placement >> side >> castle >> ep)) return false;

    // --- Field 1: piece placement (rank 8 down to rank 1) ---
    int rank = 7;
    int file = 0;
    for (char ch : placement) {
        if (ch == '/') {
            if (file != 8) return false;  // previous rank was not full
            --rank;
            file = 0;
            if (rank < 0) return false;
        } else if (ch >= '1' && ch <= '8') {
            file += ch - '0';
            if (file > 8) return false;
        } else {
            PieceType pt;
            Color c;
            if (!piece_from_char(ch, pt, c)) return false;
            if (file > 7 || rank < 0) return false;
            put_piece(tmp, pt, c, make_square(file, rank));
            ++file;
        }
    }
    if (rank != 0 || file != 8) return false;  // must end exactly on rank 1, file h

    // --- Field 2: side to move ---
    if (side == "w")      tmp.side_to_move = Color::White;
    else if (side == "b") tmp.side_to_move = Color::Black;
    else                  return false;

    // --- Field 3: castling rights ---
    tmp.castling = NO_CASTLING;
    if (castle != "-") {
        for (char ch : castle) {
            switch (ch) {
                case 'K': tmp.castling |= WHITE_OO;  break;
                case 'Q': tmp.castling |= WHITE_OOO; break;
                case 'k': tmp.castling |= BLACK_OO;  break;
                case 'q': tmp.castling |= BLACK_OOO; break;
                default:  return false;
            }
        }
    }

    // --- Field 4: en-passant target square ---
    if (ep == "-") {
        tmp.ep_square = Square::None;
    } else {
        if (ep.size() != 2) return false;
        const int f = ep[0] - 'a';
        const int r = ep[1] - '1';
        if (f < 0 || f > 7 || r < 0 || r > 7) return false;
        tmp.ep_square = make_square(f, r);
    }

    // --- Fields 5 & 6: halfmove clock and fullmove number (optional) ---
    // Some FENs (e.g. from EPD) omit these; default sensibly rather than fail.
    int halfmove = 0;
    int fullmove = 1;
    if (!(ss >> halfmove)) halfmove = 0;
    if (!(ss >> fullmove)) fullmove = 1;
    if (halfmove < 0 || fullmove < 1) return false;
    tmp.halfmove_clock  = static_cast<uint16_t>(halfmove);
    tmp.fullmove_number = static_cast<uint16_t>(fullmove);

    tmp.zobrist = compute_zobrist(tmp);
    pos = tmp;
    return true;
}

// ---------------------------------------------------------------------------
// FEN serialization
// ---------------------------------------------------------------------------
std::string to_fen(const Position& pos) {
    std::string out;

    // --- Field 1: piece placement ---
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            const Square s = make_square(file, rank);
            const PieceType pt = pos.piece_type_on(s);
            if (pt == PieceType::None) {
                ++empty;
                continue;
            }
            if (empty) {
                out += static_cast<char>('0' + empty);
                empty = 0;
            }
            char pc = PIECE_CHAR[type_index(pt)];
            if (pos.color_on(s) == Color::Black) {
                pc = static_cast<char>(std::tolower(static_cast<unsigned char>(pc)));
            }
            out += pc;
        }
        if (empty) out += static_cast<char>('0' + empty);
        if (rank > 0) out += '/';
    }

    // --- Field 2: side to move ---
    out += ' ';
    out += (pos.side_to_move == Color::White) ? 'w' : 'b';

    // --- Field 3: castling rights ---
    out += ' ';
    if (pos.castling == NO_CASTLING) {
        out += '-';
    } else {
        if (pos.castling & WHITE_OO)  out += 'K';
        if (pos.castling & WHITE_OOO) out += 'Q';
        if (pos.castling & BLACK_OO)  out += 'k';
        if (pos.castling & BLACK_OOO) out += 'q';
    }

    // --- Field 4: en-passant target ---
    out += ' ';
    if (pos.ep_square == Square::None) {
        out += '-';
    } else {
        out += static_cast<char>('a' + file_of(pos.ep_square));
        out += static_cast<char>('1' + rank_of(pos.ep_square));
    }

    // --- Fields 5 & 6 ---
    out += ' ';
    out += std::to_string(pos.halfmove_clock);
    out += ' ';
    out += std::to_string(pos.fullmove_number);

    return out;
}

// ---------------------------------------------------------------------------
// Zobrist
// ---------------------------------------------------------------------------
uint64_t compute_zobrist(const Position& pos) {
    uint64_t k = 0;
    for (int i = 0; i < NUM_SQUARES; ++i) {
        const Square s = static_cast<Square>(i);
        const PieceType pt = pos.piece_type_on(s);
        if (pt != PieceType::None) k ^= zobrist::piece(pos.color_on(s), pt, s);
    }
    if (pos.side_to_move == Color::Black) k ^= zobrist::side();
    k ^= zobrist::castling(pos.castling);
    if (pos.ep_square != Square::None) k ^= zobrist::ep_file(file_of(pos.ep_square));
    return k;
}

// ---------------------------------------------------------------------------
// make / unmake
// ---------------------------------------------------------------------------
namespace {

// Castling-rights update table. Starting from full rights, ANDing with
// CASTLE_MASK[from] & CASTLE_MASK[to] clears exactly the rights invalidated by
// this move: the king or a rook leaving its home square (from), OR a rook being
// captured on its home square (to). This one table handles all cases —
// including the easy-to-forget "capture a rook, lose the opponent's castling
// right" — with no special-casing.
constexpr std::array<uint8_t, 64> make_castle_mask() {
    std::array<uint8_t, 64> m{};
    for (int i = 0; i < 64; ++i) m[i] = ANY_CASTLING;
    m[sq_index(Square::E1)] &= static_cast<uint8_t>(~(WHITE_OO | WHITE_OOO));
    m[sq_index(Square::A1)] &= static_cast<uint8_t>(~WHITE_OOO);
    m[sq_index(Square::H1)] &= static_cast<uint8_t>(~WHITE_OO);
    m[sq_index(Square::E8)] &= static_cast<uint8_t>(~(BLACK_OO | BLACK_OOO));
    m[sq_index(Square::A8)] &= static_cast<uint8_t>(~BLACK_OOO);
    m[sq_index(Square::H8)] &= static_cast<uint8_t>(~BLACK_OO);
    return m;
}
constexpr std::array<uint8_t, 64> CASTLE_MASK = make_castle_mask();

}  // namespace

void make_move(Position& pos, Move m, StateInfo& st) {
    const Color     us     = pos.side_to_move;
    const Color     them   = ~us;
    const Square    from   = move_from(m);
    const Square    to     = move_to(m);
    const MoveFlag  flag   = move_flag(m);
    const PieceType moving = pos.piece_type_on(from);

    // Save the irreversible state so unmake can restore it verbatim.
    st.key            = pos.zobrist;
    st.castling       = pos.castling;
    st.ep_square      = pos.ep_square;
    st.halfmove_clock = pos.halfmove_clock;
    st.captured       = PieceType::None;

    uint64_t key = pos.zobrist;

    // Bitboard + key mutators. Lambdas inline to nothing under -O3; keeping the
    // key XOR next to the bitboard write makes it impossible to update one and
    // forget the other.
    auto put = [&](Color c, PieceType pt, Square s) {
        const Bitboard b = square_bb(s);
        pos.by_type[type_index(pt)]  |= b;
        pos.by_color[color_index(c)] |= b;
        key ^= zobrist::piece(c, pt, s);
    };
    auto take = [&](Color c, PieceType pt, Square s) {
        const Bitboard b = square_bb(s);
        pos.by_type[type_index(pt)]  &= ~b;
        pos.by_color[color_index(c)] &= ~b;
        key ^= zobrist::piece(c, pt, s);
    };
    auto move_pc = [&](Color c, PieceType pt, Square a, Square b) { take(c, pt, a); put(c, pt, b); };

    // Retire the old en-passant key; the square is reset and re-set below only
    // for a double push.
    if (pos.ep_square != Square::None) key ^= zobrist::ep_file(file_of(pos.ep_square));
    pos.ep_square = Square::None;

    // Remove the captured piece first (so a promotion can then occupy `to`).
    if (flag == MoveFlag::EnPassant) {
        // The captured pawn sits beside the mover's origin, not on `to`.
        const Square capsq = make_square(file_of(to), rank_of(from));
        st.captured = PieceType::Pawn;
        take(them, PieceType::Pawn, capsq);
    } else if (is_capture(m)) {
        st.captured = pos.piece_type_on(to);
        take(them, st.captured, to);
    }

    // Move (or promote, or castle) the moving piece.
    if (is_promotion(m)) {
        take(us, PieceType::Pawn, from);
        put(us, promotion_type(m), to);
    } else if (flag == MoveFlag::KingCastle || flag == MoveFlag::QueenCastle) {
        move_pc(us, PieceType::King, from, to);
        // Rook squares are fixed offsets from the king's destination.
        Square rfrom, rto;
        if (flag == MoveFlag::KingCastle) {
            rfrom = static_cast<Square>(sq_index(to) + 1);  // h-file rook
            rto   = static_cast<Square>(sq_index(to) - 1);  // f-file
        } else {
            rfrom = static_cast<Square>(sq_index(to) - 2);  // a-file rook
            rto   = static_cast<Square>(sq_index(to) + 1);  // d-file
        }
        move_pc(us, PieceType::Rook, rfrom, rto);
    } else {
        move_pc(us, moving, from, to);
        if (flag == MoveFlag::DoublePush) {
            // Set the en-passant target on every double push (perft convention).
            // Restricting it to "only when a capture is actually possible" is a
            // TT/repetition refinement for a later phase, noted deliberately.
            pos.ep_square = make_square(file_of(from), (rank_of(from) + rank_of(to)) / 2);
        }
    }
    if (pos.ep_square != Square::None) key ^= zobrist::ep_file(file_of(pos.ep_square));

    // Update castling rights (and their key contribution).
    key ^= zobrist::castling(pos.castling);
    pos.castling &= CASTLE_MASK[sq_index(from)] & CASTLE_MASK[sq_index(to)];
    key ^= zobrist::castling(pos.castling);

    // Clocks.
    if (moving == PieceType::Pawn || is_capture(m)) pos.halfmove_clock = 0;
    else                                            ++pos.halfmove_clock;
    if (us == Color::Black) ++pos.fullmove_number;

    // Flip side to move.
    pos.side_to_move = them;
    key ^= zobrist::side();

    pos.zobrist = key;
}

void unmake_move(Position& pos, Move m, const StateInfo& st) {
    const Color    us   = ~pos.side_to_move;  // the side that made the move
    const Color    them = pos.side_to_move;
    const Square   from = move_from(m);
    const Square   to   = move_to(m);
    const MoveFlag flag = move_flag(m);

    pos.side_to_move = us;
    if (us == Color::Black) --pos.fullmove_number;

    // Pure bitboard mutators — the key is restored wholesale from st.key at the
    // end, so unmake need not touch it incrementally.
    auto put = [&](Color c, PieceType pt, Square s) {
        const Bitboard b = square_bb(s);
        pos.by_type[type_index(pt)]  |= b;
        pos.by_color[color_index(c)] |= b;
    };
    auto take = [&](Color c, PieceType pt, Square s) {
        const Bitboard b = square_bb(s);
        pos.by_type[type_index(pt)]  &= ~b;
        pos.by_color[color_index(c)] &= ~b;
    };
    auto move_pc = [&](Color c, PieceType pt, Square a, Square b) { take(c, pt, a); put(c, pt, b); };

    // Reverse the piece movement.
    if (is_promotion(m)) {
        take(us, promotion_type(m), to);
        put(us, PieceType::Pawn, from);
    } else if (flag == MoveFlag::KingCastle || flag == MoveFlag::QueenCastle) {
        move_pc(us, PieceType::King, to, from);
        Square rfrom, rto;
        if (flag == MoveFlag::KingCastle) {
            rfrom = static_cast<Square>(sq_index(to) + 1);
            rto   = static_cast<Square>(sq_index(to) - 1);
        } else {
            rfrom = static_cast<Square>(sq_index(to) - 2);
            rto   = static_cast<Square>(sq_index(to) + 1);
        }
        move_pc(us, PieceType::Rook, rto, rfrom);  // rook back to its home
    } else {
        const PieceType moved = pos.piece_type_on(to);  // whatever landed on `to`
        move_pc(us, moved, to, from);
    }

    // Restore the captured piece.
    if (flag == MoveFlag::EnPassant) {
        const Square capsq = make_square(file_of(to), rank_of(from));
        put(them, PieceType::Pawn, capsq);
    } else if (is_capture(m)) {
        put(them, st.captured, to);
    }

    // Restore irreversible scalars and the key.
    pos.castling       = st.castling;
    pos.ep_square      = st.ep_square;
    pos.halfmove_clock = st.halfmove_clock;
    pos.zobrist        = st.key;
}
