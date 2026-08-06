#include "movegen.hpp"

#include "attacks.hpp"
#include "bitboard.hpp"
#include "slider.hpp"

namespace movegen {

namespace {

using enum MoveFlag;  // Quiet, Capture, DoublePush, ... in scope (no name clashes)

Bitboard pieces(const Position& p, Color c, PieceType pt) {
    return p.by_type[type_index(pt)] & p.by_color[color_index(c)];
}

// Emit non-pawn moves toward a target set, tagging captures by occupancy.
void add_targets(MoveList& list, Square from, Bitboard targets, Bitboard occ) {
    while (targets) {
        const Square to = pop_lsb(targets);
        list.add(make_move(from, to, (occ & square_bb(to)) ? Capture : Quiet));
    }
}

void add_promotions(MoveList& list, Square from, Square to, bool capture) {
    if (capture) {
        list.add(make_move(from, to, PromoKnightCapture));
        list.add(make_move(from, to, PromoBishopCapture));
        list.add(make_move(from, to, PromoRookCapture));
        list.add(make_move(from, to, PromoQueenCapture));
    } else {
        list.add(make_move(from, to, PromoKnight));
        list.add(make_move(from, to, PromoBishop));
        list.add(make_move(from, to, PromoRook));
        list.add(make_move(from, to, PromoQueen));
    }
}

void gen_pawns(const Position& pos, Color us, MoveList& list) {
    const Bitboard occ   = pos.occupied();
    const Bitboard enemy = pos.by_color[color_index(~us)];
    Bitboard pawns       = pieces(pos, us, PieceType::Pawn);

    const int dir        = (us == Color::White) ? 8 : -8;
    const int start_rank = (us == Color::White) ? 1 : 6;
    const int promo_rank = (us == Color::White) ? 7 : 0;

    while (pawns) {
        const Square from = pop_lsb(pawns);

        // Pushes.
        const Square one = static_cast<Square>(sq_index(from) + dir);
        if (!(occ & square_bb(one))) {
            if (rank_of(one) == promo_rank) {
                add_promotions(list, from, one, /*capture=*/false);
            } else {
                list.add(make_move(from, one, Quiet));
                if (rank_of(from) == start_rank) {
                    const Square two = static_cast<Square>(sq_index(one) + dir);
                    if (!(occ & square_bb(two))) list.add(make_move(from, two, DoublePush));
                }
            }
        }

        // Captures (including capture-promotions).
        Bitboard caps = attacks::pawn(us, from) & enemy;
        while (caps) {
            const Square to = pop_lsb(caps);
            if (rank_of(to) == promo_rank) add_promotions(list, from, to, /*capture=*/true);
            else                           list.add(make_move(from, to, Capture));
        }

        // En passant: the pawn attacks the ep target square.
        if (pos.ep_square != Square::None &&
            (attacks::pawn(us, from) & square_bb(pos.ep_square))) {
            list.add(make_move(from, pos.ep_square, EnPassant));
        }
    }
}

void gen_pieces(const Position& pos, Color us, MoveList& list) {
    const Bitboard own = pos.by_color[color_index(us)];
    const Bitboard occ = pos.occupied();
    const Bitboard mob = ~own;  // can move to any square not holding our own piece

    Bitboard knights = pieces(pos, us, PieceType::Knight);
    while (knights) {
        const Square from = pop_lsb(knights);
        add_targets(list, from, attacks::knight(from) & mob, occ);
    }

    const Square ksq = lsb(pieces(pos, us, PieceType::King));
    add_targets(list, ksq, attacks::king(ksq) & mob, occ);

    Bitboard bishops = pieces(pos, us, PieceType::Bishop);
    while (bishops) {
        const Square from = pop_lsb(bishops);
        add_targets(list, from, slider::bishop_attacks(from, occ) & mob, occ);
    }

    Bitboard rooks = pieces(pos, us, PieceType::Rook);
    while (rooks) {
        const Square from = pop_lsb(rooks);
        add_targets(list, from, slider::rook_attacks(from, occ) & mob, occ);
    }

    Bitboard queens = pieces(pos, us, PieceType::Queen);
    while (queens) {
        const Square from = pop_lsb(queens);
        add_targets(list, from, slider::queen_attacks(from, occ) & mob, occ);
    }
}

// Castling. Emitted only when rights are held, the squares between king and
// rook are empty, and the king is not in check on its start, transit, or
// destination square (castling out of / through / into check is illegal). The
// transit-square check is done HERE because the make/unmake landing-square
// filter alone cannot see the square the king passes over.
void gen_castling(const Position& pos, Color us, MoveList& list) {
    const Bitboard occ  = pos.occupied();
    const Color    them = ~us;

    auto safe  = [&](Square s) { return !is_attacked(pos, s, them); };
    auto empty = [&](Bitboard b) { return (occ & b) == 0; };

    if (us == Color::White) {
        if ((pos.castling & WHITE_OO) &&
            empty(square_bb(Square::F1) | square_bb(Square::G1)) &&
            safe(Square::E1) && safe(Square::F1) && safe(Square::G1)) {
            list.add(make_move(Square::E1, Square::G1, KingCastle));
        }
        if ((pos.castling & WHITE_OOO) &&
            empty(square_bb(Square::B1) | square_bb(Square::C1) | square_bb(Square::D1)) &&
            safe(Square::E1) && safe(Square::D1) && safe(Square::C1)) {
            list.add(make_move(Square::E1, Square::C1, QueenCastle));
        }
    } else {
        if ((pos.castling & BLACK_OO) &&
            empty(square_bb(Square::F8) | square_bb(Square::G8)) &&
            safe(Square::E8) && safe(Square::F8) && safe(Square::G8)) {
            list.add(make_move(Square::E8, Square::G8, KingCastle));
        }
        if ((pos.castling & BLACK_OOO) &&
            empty(square_bb(Square::B8) | square_bb(Square::C8) | square_bb(Square::D8)) &&
            safe(Square::E8) && safe(Square::D8) && safe(Square::C8)) {
            list.add(make_move(Square::E8, Square::C8, QueenCastle));
        }
    }
}

}  // namespace

bool is_attacked(const Position& pos, Square s, Color by) {
    const Bitboard occ = pos.occupied();

    // A pawn of color `by` attacks s iff a pawn of the opposite color placed on
    // s would attack one of `by`'s pawns — pawn attacks run "backwards" here.
    if (attacks::pawn(~by, s) & pieces(pos, by, PieceType::Pawn)) return true;
    if (attacks::knight(s)    & pieces(pos, by, PieceType::Knight)) return true;
    if (attacks::king(s)      & pieces(pos, by, PieceType::King)) return true;

    const Bitboard diag = pieces(pos, by, PieceType::Bishop) | pieces(pos, by, PieceType::Queen);
    if (slider::bishop_attacks(s, occ) & diag) return true;

    const Bitboard orth = pieces(pos, by, PieceType::Rook) | pieces(pos, by, PieceType::Queen);
    if (slider::rook_attacks(s, occ) & orth) return true;

    return false;
}

Square king_square(const Position& pos, Color c) {
    return lsb(pieces(pos, c, PieceType::King));
}

bool in_check(const Position& pos) {
    return is_attacked(pos, king_square(pos, pos.side_to_move), ~pos.side_to_move);
}

void generate_pseudo_legal(const Position& pos, MoveList& list) {
    list.clear();
    const Color us = pos.side_to_move;
    gen_pawns(pos, us, list);
    gen_pieces(pos, us, list);
    gen_castling(pos, us, list);
}

void generate_legal(Position& pos, MoveList& list) {
    MoveList pseudo;
    generate_pseudo_legal(pos, pseudo);

    list.clear();
    const Color us   = pos.side_to_move;
    const Color them = ~us;

    for (const Move m : pseudo) {
        StateInfo st;
        make_move(pos, m, st);
        // After make, `us`'s king must not be attacked by `them` (now to move).
        if (!is_attacked(pos, king_square(pos, us), them)) list.add(m);
        unmake_move(pos, m, st);
    }
}

}  // namespace movegen
