from constants import *
from movegen import decode_move


def _zobrist(piece, r, c):
    return ZOBRIST_TABLE[piece + 6][r][c]


def _compute_hash(squares, white_to_move):
    h = 0
    for r in range(8):
        for c in range(8):
            piece = squares[r][c]
            if piece != EMPTY:
                h ^= _zobrist(piece, r, c)
    if not white_to_move:
        h ^= ZOBRIST_SIDE
    return h


class Board:
    def __init__(self):
        self.squares       = [[EMPTY] * 8 for _ in range(8)]
        self.white_to_move = True

        self.white_king_pos = None
        self.black_king_pos = None

        self.en_passant_sq = None

        self.castle_wk = True
        self.castle_wq = True
        self.castle_bk = True
        self.castle_bq = True

        self.hash = 0

        # Tracks how many times each position hash has occurred this game.
        # Used for threefold repetition detection.
        self.position_history = {}

        # Move stack — flat tuple per entry:
        # (move, captured, wkr, wkc, bkr, bkc,
        #  white_to_move, ep_r, ep_c,
        #  castle_wk, castle_wq, castle_bk, castle_bq, hash)
        self.move_stack = []

    # ==========================================
    # Setup
    # ==========================================

    def setup_starting_position(self):
        self.squares = [[EMPTY] * 8 for _ in range(8)]

        for c in range(8):
            self.squares[6][c] =  PAWN
            self.squares[1][c] = -PAWN

        self.squares[7][0] =  ROOK;  self.squares[7][7] =  ROOK
        self.squares[0][0] = -ROOK;  self.squares[0][7] = -ROOK
        self.squares[7][1] =  KNIGHT; self.squares[7][6] =  KNIGHT
        self.squares[0][1] = -KNIGHT; self.squares[0][6] = -KNIGHT
        self.squares[7][2] =  BISHOP; self.squares[7][5] =  BISHOP
        self.squares[0][2] = -BISHOP; self.squares[0][5] = -BISHOP
        self.squares[7][3] =  QUEEN;  self.squares[0][3] = -QUEEN
        self.squares[7][4] =  KING;   self.squares[0][4] = -KING

        self.white_king_pos = (7, 4)
        self.black_king_pos = (0, 4)
        self.white_to_move  = True
        self.en_passant_sq  = None
        self.castle_wk      = True
        self.castle_wq      = True
        self.castle_bk      = True
        self.castle_bq      = True
        self.move_stack        = []
        self.position_history  = {}
        self.hash              = _compute_hash(self.squares, self.white_to_move)
        self.position_history[self.hash] = 1

    def initialize_king_cache(self):
        for r in range(8):
            for c in range(8):
                piece = self.squares[r][c]
                if piece == KING:
                    self.white_king_pos = (r, c)
                elif piece == -KING:
                    self.black_king_pos = (r, c)
        self.hash = _compute_hash(self.squares, self.white_to_move)
        self.position_history = {self.hash: 1}

    # ==========================================
    # Make / Undo
    # ==========================================

    def make_move(self, move):
        r1, c1, r2, c2, flag = decode_move(move)

        moving   = self.squares[r1][c1]
        captured = self.squares[r2][c2]
        white    = moving > 0

        wkr, wkc = self.white_king_pos
        bkr, bkc = self.black_king_pos
        ep_r, ep_c = self.en_passant_sq if self.en_passant_sq else (-1, -1)

        self.move_stack.append((
            move, captured,
            wkr, wkc, bkr, bkc,
            self.white_to_move,
            ep_r, ep_c,
            self.castle_wk, self.castle_wq,
            self.castle_bk, self.castle_bq,
            self.hash
        ))

        self.en_passant_sq = None

        # --- Incremental Zobrist update ---
        self.hash ^= _zobrist(moving, r1, c1)
        if captured != EMPTY:
            self.hash ^= _zobrist(captured, r2, c2)

        # --- Apply move ---
        if flag == FLAG_EN_PASSANT:
            self.squares[r1][c1] = EMPTY
            self.squares[r2][c2] = moving
            cap_r = r2 + (1 if white else -1)
            cap_piece = self.squares[cap_r][c2]
            self.hash ^= _zobrist(cap_piece, cap_r, c2)
            self.squares[cap_r][c2] = EMPTY
            self.hash ^= _zobrist(moving, r2, c2)

        elif flag in PROMOTION_FLAGS:
            promo_piece = PROMOTION_PIECES[flag]
            placed = promo_piece if white else -promo_piece
            self.squares[r1][c1] = EMPTY
            self.squares[r2][c2] = placed
            self.hash ^= _zobrist(placed, r2, c2)

        elif flag == FLAG_CASTLE_KINGSIDE:
            self.squares[r1][c1] = EMPTY
            self.squares[r2][c2] = moving
            self.hash ^= _zobrist(moving, r2, c2)
            rook = self.squares[r1][7]
            self.hash ^= _zobrist(rook, r1, 7)
            self.squares[r1][7] = EMPTY
            self.squares[r1][5] = rook
            self.hash ^= _zobrist(rook, r1, 5)

        elif flag == FLAG_CASTLE_QUEENSIDE:
            self.squares[r1][c1] = EMPTY
            self.squares[r2][c2] = moving
            self.hash ^= _zobrist(moving, r2, c2)
            rook = self.squares[r1][0]
            self.hash ^= _zobrist(rook, r1, 0)
            self.squares[r1][0] = EMPTY
            self.squares[r1][3] = rook
            self.hash ^= _zobrist(rook, r1, 3)

        else:
            self.squares[r1][c1] = EMPTY
            self.squares[r2][c2] = moving
            self.hash ^= _zobrist(moving, r2, c2)

            if abs(moving) == PAWN and abs(r2 - r1) == 2:
                self.en_passant_sq = ((r1 + r2) // 2, c1)

        # Update king cache
        if moving == KING:
            self.white_king_pos = (r2, c2)
        elif moving == -KING:
            self.black_king_pos = (r2, c2)

        # Update castling rights
        if moving == KING:
            self.castle_wk = False;  self.castle_wq = False
        elif moving == -KING:
            self.castle_bk = False;  self.castle_bq = False

        if r1 == 7 and c1 == 7: self.castle_wk = False
        if r1 == 7 and c1 == 0: self.castle_wq = False
        if r1 == 0 and c1 == 7: self.castle_bk = False
        if r1 == 0 and c1 == 0: self.castle_bq = False
        if r2 == 7 and c2 == 7: self.castle_wk = False
        if r2 == 7 and c2 == 0: self.castle_wq = False
        if r2 == 0 and c2 == 7: self.castle_bk = False
        if r2 == 0 and c2 == 0: self.castle_bq = False

        # Flip side to move
        self.hash ^= ZOBRIST_SIDE
        self.white_to_move = not self.white_to_move

        # Record this position for repetition detection
        self.position_history[self.hash] = self.position_history.get(self.hash, 0) + 1

    def undo_move(self):
        (move, captured,
         wkr, wkc, bkr, bkc,
         white_to_move,
         ep_r, ep_c,
         castle_wk, castle_wq,
         castle_bk, castle_bq,
         saved_hash) = self.move_stack.pop()

        r1, c1, r2, c2, flag = decode_move(move)
        white = white_to_move

        if flag == FLAG_EN_PASSANT:
            self.squares[r1][c1] = PAWN if white else -PAWN
            self.squares[r2][c2] = EMPTY
            cap_r = r2 + (1 if white else -1)
            self.squares[cap_r][c2] = -PAWN if white else PAWN

        elif flag in PROMOTION_FLAGS:
            self.squares[r1][c1] = PAWN if white else -PAWN
            self.squares[r2][c2] = captured

        elif flag == FLAG_CASTLE_KINGSIDE:
            self.squares[r1][c1] = KING if white else -KING
            self.squares[r2][c2] = EMPTY
            rook = ROOK if white else -ROOK
            self.squares[r1][5] = EMPTY
            self.squares[r1][7] = rook

        elif flag == FLAG_CASTLE_QUEENSIDE:
            self.squares[r1][c1] = KING if white else -KING
            self.squares[r2][c2] = EMPTY
            rook = ROOK if white else -ROOK
            self.squares[r1][3] = EMPTY
            self.squares[r1][0] = rook

        else:
            self.squares[r1][c1] = self.squares[r2][c2]
            self.squares[r2][c2] = captured

        self.white_king_pos = (wkr, wkc)
        self.black_king_pos = (bkr, bkc)
        self.white_to_move  = white_to_move
        self.en_passant_sq  = (ep_r, ep_c) if ep_r != -1 else None
        self.castle_wk      = castle_wk
        self.castle_wq      = castle_wq
        self.castle_bk      = castle_bk
        self.castle_bq      = castle_bq
        # Decrement the position we're leaving BEFORE restoring the hash
        self.position_history[self.hash] = self.position_history.get(self.hash, 1) - 1
        self.hash = saved_hash

    # ==========================================
    # King Helpers
    # ==========================================

    def get_king_position(self, white):
        return self.white_king_pos if white else self.black_king_pos

    def is_in_check(self, white):
        return self.is_square_attacked(self.get_king_position(white), not white)

    # ==========================================
    # Attack Detection
    # ==========================================

    def is_square_attacked(self, square, by_white):
        r, c = square

        # Pawns
        direction = 1 if by_white else -1
        for dc in (-1, 1):
            rr, cc = r + direction, c + dc
            if 0 <= rr < 8 and 0 <= cc < 8:
                if self.squares[rr][cc] == (PAWN if by_white else -PAWN):
                    return True

        # Knights
        for dr, dc in KNIGHT_OFFSETS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < 8 and 0 <= cc < 8:
                if self.squares[rr][cc] == (KNIGHT if by_white else -KNIGHT):
                    return True

        # King
        for dr, dc in KING_OFFSETS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < 8 and 0 <= cc < 8:
                if self.squares[rr][cc] == (KING if by_white else -KING):
                    return True

        # Rooks / Queens
        for dr, dc in ROOK_DIRS:
            rr, cc = r + dr, c + dc
            while 0 <= rr < 8 and 0 <= cc < 8:
                piece = self.squares[rr][cc]
                if piece != EMPTY:
                    if by_white and piece in (ROOK, QUEEN):         return True
                    if not by_white and piece in (-ROOK, -QUEEN):   return True
                    break
                rr += dr; cc += dc

        # Bishops / Queens
        for dr, dc in BISHOP_DIRS:
            rr, cc = r + dr, c + dc
            while 0 <= rr < 8 and 0 <= cc < 8:
                piece = self.squares[rr][cc]
                if piece != EMPTY:
                    if by_white and piece in (BISHOP, QUEEN):       return True
                    if not by_white and piece in (-BISHOP, -QUEEN): return True
                    break
                rr += dr; cc += dc

        return False

    # ==========================================
    # Display
    # ==========================================

    def __str__(self):
        result = "  ╔═════════════════════════════╗\n"

        for r in range(8):
            result += str(8 - r) + " ║"

            for c in range(8):
                piece = self.squares[r][c]
                symbol = PIECE_SYMBOLS[piece]

                # Simple square coloring
                if (r + c) % 2 == 0:
                    result += f"[{symbol}]"
                else:
                    result += f" {symbol} "

            result += "║\n"

        result += "  ╚═════════════════════════════╝\n"
        result += "    a  b  c  d  e  f  g  h\n"

        if self.white_to_move:
            result += "White to move\n"
        else:
            result += "Black to move\n"

        return result