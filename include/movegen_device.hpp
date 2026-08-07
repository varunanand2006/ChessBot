// Device-side move generation + make/unmake — the GPU port of movegen.cpp and
// position.cpp's make/unmake, as CH_HD functions over a Position (POD) and an
// uploaded slider::DeviceSliders.
//
// TWO deliberate differences from the host code, both safe for the retrograde
// sweep and validated against it:
//   1. NO Zobrist. The host make/unmake maintain the hash incrementally; the
//      sweep only ever reads a position's bitboards (encode works off them), so
//      the device make/unmake skip the key entirely — no zobrist tables on the
//      GPU. StateInfo::key is simply left untouched.
//   2. Free helper functions instead of capturing lambdas, so the code is
//      unambiguously device-compilable.
//
// Everything else mirrors the host line-for-line. Because these are CH_HD, the
// same functions run on the host, where test_movegen_device diffs their legal
// move sets against movegen::generate_legal over full perft trees — a complete
// local Phase 2 gate with no GPU. cuda/movegen_check.cu then confirms the device
// reproduces the host result.

#pragma once

#include <array>
#include <cstdint>

#include "attacks.hpp"
#include "bitboard_device.hpp"
#include "cuda_compat.hpp"
#include "move.hpp"
#include "position.hpp"
#include "slider_device.hpp"
#include "types.hpp"

namespace movegen_dev {

// --- board helpers ---------------------------------------------------------
CH_HD inline Bitboard pieces(const Position& p, Color c, PieceType pt) {
    return p.by_type[type_index(pt)] & p.by_color[color_index(c)];
}

CH_HD inline PieceType piece_type_on(const Position& p, Square s) {
    const Bitboard b = square_bb(s);
    for (int t = 0; t < NUM_PIECE_TYPES; ++t)
        if (p.by_type[t] & b) return static_cast<PieceType>(t);
    return PieceType::None;
}

CH_HD inline bool is_attacked(const Position& p, Square s, Color by,
                              const slider::DeviceSliders& S) {
    const Bitboard occ = p.by_color[0] | p.by_color[1];
    if (attacks::pawn(~by, s) & pieces(p, by, PieceType::Pawn))   return true;
    if (attacks::knight(s)    & pieces(p, by, PieceType::Knight)) return true;
    if (attacks::king(s)      & pieces(p, by, PieceType::King))   return true;
    const Bitboard diag = pieces(p, by, PieceType::Bishop) | pieces(p, by, PieceType::Queen);
    if (slider::bishop_attacks_dev(S, sq_index(s), occ) & diag)   return true;
    const Bitboard orth = pieces(p, by, PieceType::Rook) | pieces(p, by, PieceType::Queen);
    if (slider::rook_attacks_dev(S, sq_index(s), occ) & orth)     return true;
    return false;
}

CH_HD inline Square king_square(const Position& p, Color c) {
    return lsb_dev(pieces(p, c, PieceType::King));
}

// --- make / unmake (no Zobrist) --------------------------------------------
namespace detail {
CH_HD inline void put(Position& p, Color c, PieceType pt, Square s) {
    const Bitboard b = square_bb(s);
    p.by_type[type_index(pt)]  |= b;
    p.by_color[color_index(c)] |= b;
}
CH_HD inline void take(Position& p, Color c, PieceType pt, Square s) {
    const Bitboard b = square_bb(s);
    p.by_type[type_index(pt)]  &= ~b;
    p.by_color[color_index(c)] &= ~b;
}
CH_HD inline void move_pc(Position& p, Color c, PieceType pt, Square a, Square b) {
    take(p, c, pt, a); put(p, c, pt, b);
}

// Castling-rights update mask for ONE square — a private copy of position.cpp's
// CASTLE_MASK logic. A per-square function, not a namespace-scope 64-byte table,
// because make_move is CH_HD: indexing a host array (CASTLE_MASK[sq]) at runtime
// is an ODR-use nvcc rejects in device code ("identifier ... undefined in device
// code"). Only the six king/rook home squares clear any rights; every other
// square keeps all of them, so this short branch chain replaces the table at no
// real cost — two calls per move instead of two loads, and no upload/bind.
CH_HD inline uint8_t castle_mask(int sq) {
    uint8_t m = ANY_CASTLING;
    if (sq == sq_index(Square::E1))      m &= static_cast<uint8_t>(~(WHITE_OO | WHITE_OOO));
    else if (sq == sq_index(Square::A1)) m &= static_cast<uint8_t>(~WHITE_OOO);
    else if (sq == sq_index(Square::H1)) m &= static_cast<uint8_t>(~WHITE_OO);
    else if (sq == sq_index(Square::E8)) m &= static_cast<uint8_t>(~(BLACK_OO | BLACK_OOO));
    else if (sq == sq_index(Square::A8)) m &= static_cast<uint8_t>(~BLACK_OOO);
    else if (sq == sq_index(Square::H8)) m &= static_cast<uint8_t>(~BLACK_OO);
    return m;
}
}  // namespace detail

CH_HD inline void make_move(Position& pos, Move m, StateInfo& st) {
    const Color     us     = pos.side_to_move;
    const Color     them   = ~us;
    const Square    from   = move_from(m);
    const Square    to     = move_to(m);
    const MoveFlag  flag   = move_flag(m);
    const PieceType moving = piece_type_on(pos, from);

    st.castling       = pos.castling;
    st.ep_square      = pos.ep_square;
    st.halfmove_clock = pos.halfmove_clock;
    st.captured       = PieceType::None;

    pos.ep_square = Square::None;

    if (flag == MoveFlag::EnPassant) {
        const Square capsq = make_square(file_of(to), rank_of(from));
        st.captured = PieceType::Pawn;
        detail::take(pos, them, PieceType::Pawn, capsq);
    } else if (is_capture(m)) {
        st.captured = piece_type_on(pos, to);
        detail::take(pos, them, st.captured, to);
    }

    if (is_promotion(m)) {
        detail::take(pos, us, PieceType::Pawn, from);
        detail::put(pos, us, promotion_type(m), to);
    } else if (flag == MoveFlag::KingCastle || flag == MoveFlag::QueenCastle) {
        detail::move_pc(pos, us, PieceType::King, from, to);
        Square rfrom, rto;
        if (flag == MoveFlag::KingCastle) {
            rfrom = static_cast<Square>(sq_index(to) + 1);
            rto   = static_cast<Square>(sq_index(to) - 1);
        } else {
            rfrom = static_cast<Square>(sq_index(to) - 2);
            rto   = static_cast<Square>(sq_index(to) + 1);
        }
        detail::move_pc(pos, us, PieceType::Rook, rfrom, rto);
    } else {
        detail::move_pc(pos, us, moving, from, to);
        if (flag == MoveFlag::DoublePush)
            pos.ep_square = make_square(file_of(from), (rank_of(from) + rank_of(to)) / 2);
    }

    pos.castling &= detail::castle_mask(sq_index(from)) & detail::castle_mask(sq_index(to));

    if (moving == PieceType::Pawn || is_capture(m)) pos.halfmove_clock = 0;
    else                                            ++pos.halfmove_clock;
    if (us == Color::Black) ++pos.fullmove_number;

    pos.side_to_move = them;
}

CH_HD inline void unmake_move(Position& pos, Move m, const StateInfo& st) {
    const Color    us   = ~pos.side_to_move;
    const Color    them = pos.side_to_move;
    const Square   from = move_from(m);
    const Square   to   = move_to(m);
    const MoveFlag flag = move_flag(m);

    pos.side_to_move = us;
    if (us == Color::Black) --pos.fullmove_number;

    if (is_promotion(m)) {
        detail::take(pos, us, promotion_type(m), to);
        detail::put(pos, us, PieceType::Pawn, from);
    } else if (flag == MoveFlag::KingCastle || flag == MoveFlag::QueenCastle) {
        detail::move_pc(pos, us, PieceType::King, to, from);
        Square rfrom, rto;
        if (flag == MoveFlag::KingCastle) {
            rfrom = static_cast<Square>(sq_index(to) + 1);
            rto   = static_cast<Square>(sq_index(to) - 1);
        } else {
            rfrom = static_cast<Square>(sq_index(to) - 2);
            rto   = static_cast<Square>(sq_index(to) + 1);
        }
        detail::move_pc(pos, us, PieceType::Rook, rto, rfrom);
    } else {
        const PieceType moved = piece_type_on(pos, to);
        detail::move_pc(pos, us, moved, to, from);
    }

    if (flag == MoveFlag::EnPassant) {
        const Square capsq = make_square(file_of(to), rank_of(from));
        detail::put(pos, them, PieceType::Pawn, capsq);
    } else if (is_capture(m)) {
        detail::put(pos, them, st.captured, to);
    }

    pos.castling       = st.castling;
    pos.ep_square      = st.ep_square;
    pos.halfmove_clock = st.halfmove_clock;
}

// --- pseudo-legal generation -----------------------------------------------
namespace detail {
CH_HD inline void add_targets(MoveList& list, Square from, Bitboard targets, Bitboard occ) {
    while (targets) {
        const Square to = pop_lsb_dev(targets);
        list.add(make_move(from, to,
                           (occ & square_bb(to)) ? MoveFlag::Capture : MoveFlag::Quiet));
    }
}
CH_HD inline void add_promotions(MoveList& list, Square from, Square to, bool capture) {
    if (capture) {
        list.add(make_move(from, to, MoveFlag::PromoKnightCapture));
        list.add(make_move(from, to, MoveFlag::PromoBishopCapture));
        list.add(make_move(from, to, MoveFlag::PromoRookCapture));
        list.add(make_move(from, to, MoveFlag::PromoQueenCapture));
    } else {
        list.add(make_move(from, to, MoveFlag::PromoKnight));
        list.add(make_move(from, to, MoveFlag::PromoBishop));
        list.add(make_move(from, to, MoveFlag::PromoRook));
        list.add(make_move(from, to, MoveFlag::PromoQueen));
    }
}
}  // namespace detail

CH_HD inline void gen_pawns(const Position& pos, Color us, MoveList& list) {
    const Bitboard occ   = pos.by_color[0] | pos.by_color[1];
    const Bitboard enemy = pos.by_color[color_index(~us)];
    Bitboard pawns       = pieces(pos, us, PieceType::Pawn);

    const int dir        = (us == Color::White) ? 8 : -8;
    const int start_rank = (us == Color::White) ? 1 : 6;
    const int promo_rank = (us == Color::White) ? 7 : 0;

    while (pawns) {
        const Square from = pop_lsb_dev(pawns);

        const Square one = static_cast<Square>(sq_index(from) + dir);
        if (!(occ & square_bb(one))) {
            if (rank_of(one) == promo_rank) {
                detail::add_promotions(list, from, one, false);
            } else {
                list.add(make_move(from, one, MoveFlag::Quiet));
                if (rank_of(from) == start_rank) {
                    const Square two = static_cast<Square>(sq_index(one) + dir);
                    if (!(occ & square_bb(two)))
                        list.add(make_move(from, two, MoveFlag::DoublePush));
                }
            }
        }

        Bitboard caps = attacks::pawn(us, from) & enemy;
        while (caps) {
            const Square to = pop_lsb_dev(caps);
            if (rank_of(to) == promo_rank) detail::add_promotions(list, from, to, true);
            else                           list.add(make_move(from, to, MoveFlag::Capture));
        }

        if (pos.ep_square != Square::None &&
            (attacks::pawn(us, from) & square_bb(pos.ep_square))) {
            list.add(make_move(from, pos.ep_square, MoveFlag::EnPassant));
        }
    }
}

CH_HD inline void gen_pieces(const Position& pos, Color us, MoveList& list,
                             const slider::DeviceSliders& S) {
    const Bitboard own = pos.by_color[color_index(us)];
    const Bitboard occ = pos.by_color[0] | pos.by_color[1];
    const Bitboard mob = ~own;

    Bitboard knights = pieces(pos, us, PieceType::Knight);
    while (knights) {
        const Square from = pop_lsb_dev(knights);
        detail::add_targets(list, from, attacks::knight(from) & mob, occ);
    }

    const Square ksq = lsb_dev(pieces(pos, us, PieceType::King));
    detail::add_targets(list, ksq, attacks::king(ksq) & mob, occ);

    Bitboard bishops = pieces(pos, us, PieceType::Bishop);
    while (bishops) {
        const Square from = pop_lsb_dev(bishops);
        detail::add_targets(list, from,
                            slider::bishop_attacks_dev(S, sq_index(from), occ) & mob, occ);
    }

    Bitboard rooks = pieces(pos, us, PieceType::Rook);
    while (rooks) {
        const Square from = pop_lsb_dev(rooks);
        detail::add_targets(list, from,
                            slider::rook_attacks_dev(S, sq_index(from), occ) & mob, occ);
    }

    Bitboard queens = pieces(pos, us, PieceType::Queen);
    while (queens) {
        const Square from = pop_lsb_dev(queens);
        detail::add_targets(list, from,
                            slider::queen_attacks_dev(S, sq_index(from), occ) & mob, occ);
    }
}

CH_HD inline void gen_castling(const Position& pos, Color us, MoveList& list,
                               const slider::DeviceSliders& S) {
    const Bitboard occ  = pos.by_color[0] | pos.by_color[1];
    const Color    them = ~us;

    if (us == Color::White) {
        if ((pos.castling & WHITE_OO) &&
            !(occ & (square_bb(Square::F1) | square_bb(Square::G1))) &&
            !is_attacked(pos, Square::E1, them, S) &&
            !is_attacked(pos, Square::F1, them, S) &&
            !is_attacked(pos, Square::G1, them, S)) {
            list.add(make_move(Square::E1, Square::G1, MoveFlag::KingCastle));
        }
        if ((pos.castling & WHITE_OOO) &&
            !(occ & (square_bb(Square::B1) | square_bb(Square::C1) | square_bb(Square::D1))) &&
            !is_attacked(pos, Square::E1, them, S) &&
            !is_attacked(pos, Square::D1, them, S) &&
            !is_attacked(pos, Square::C1, them, S)) {
            list.add(make_move(Square::E1, Square::C1, MoveFlag::QueenCastle));
        }
    } else {
        if ((pos.castling & BLACK_OO) &&
            !(occ & (square_bb(Square::F8) | square_bb(Square::G8))) &&
            !is_attacked(pos, Square::E8, them, S) &&
            !is_attacked(pos, Square::F8, them, S) &&
            !is_attacked(pos, Square::G8, them, S)) {
            list.add(make_move(Square::E8, Square::G8, MoveFlag::KingCastle));
        }
        if ((pos.castling & BLACK_OOO) &&
            !(occ & (square_bb(Square::B8) | square_bb(Square::C8) | square_bb(Square::D8))) &&
            !is_attacked(pos, Square::E8, them, S) &&
            !is_attacked(pos, Square::D8, them, S) &&
            !is_attacked(pos, Square::C8, them, S)) {
            list.add(make_move(Square::E8, Square::C8, MoveFlag::QueenCastle));
        }
    }
}

CH_HD inline void generate_pseudo_legal(const Position& pos, MoveList& list,
                                        const slider::DeviceSliders& S) {
    list.clear();
    const Color us = pos.side_to_move;
    gen_pawns(pos, us, list);
    gen_pieces(pos, us, list, S);
    gen_castling(pos, us, list, S);
}

// --- fully legal generation (make/unmake filter) ---------------------------
CH_HD inline void generate_legal(Position& pos, MoveList& list,
                                 const slider::DeviceSliders& S) {
    MoveList pseudo;
    generate_pseudo_legal(pos, pseudo, S);

    list.clear();
    const Color us   = pos.side_to_move;
    const Color them = ~us;

    for (int i = 0; i < pseudo.size(); ++i) {
        const Move m = pseudo[i];
        StateInfo st;
        // Qualified: the 3-arg (Position,Move,StateInfo) make/unmake here would
        // otherwise be ambiguous with the global ::make_move of the same shape
        // (pulled in by ADL on the Position argument).
        movegen_dev::make_move(pos, m, st);
        if (!is_attacked(pos, king_square(pos, us), them, S)) list.add(m);
        movegen_dev::unmake_move(pos, m, st);
    }
}

}  // namespace movegen_dev
