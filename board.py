from constants import *
from movegen import decode_move

class Board:
    def __init__(self):
        self.squares       = [[EMPTY] * 8 for _ in range(8)]
        self.white_to_move = True

        self.white_king_pos = None
        self.black_king_pos = None

        # En passant target square (landing square), or None
        self.en_passant_sq = None

        # Castling rights
        self.castle_wk = True   # White kingside
        self.castle_wq = True   # White queenside
        self.castle_bk = True   # Black kingside
        self.castle_bq = True   # Black queenside

        # Move stack entries are flat tuples:
        # (move, captured, wkr, wkc, bkr, bkc,
        #  white_to_move, ep_r, ep_c,
        #  castle_wk, castle_wq, castle_bk, castle_bq)
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
        self.move_stack     = []

    def initialize_king_cache(self):
        for r in range(8):
            for c in range(8):
                piece = self.squares[r][c]
                if piece == KING:
                    self.white_king_pos = (r, c)
                elif piece == -KING:
                    self.black_king_pos = (r, c)

    # ==========================================
    # Make / Undo
    # ==========================================

    def make_move(self, move):
        r1, c1, r2, c2, flag = decode_move(move)

        moving   = self.squares[r1][c1]
        captured = self.squares[r2][c2]
        white    = moving > 0

        # Save state
        wkr, wkc = self.white_king_pos
        bkr, bkc = self.black_king_pos
        ep_r, ep_c = self.en_passant_sq if self.en_passant_sq else (-1, -1)

        self.move_stack.append((
            move, captured,
            wkr, wkc, bkr, bkc,
            self.white_to_move,
            ep_r, ep_c,
            self.castle_wk, self.castle_wq,
            self.castle_bk, self.castle_bq
        ))

        # Clear en passant — re-set below if double push
        self.en_passant_sq = None

        # --- Handle each flag ---

        if flag == FLAG_EN_PASSANT:
            self.squares[r1][c1] = EMPTY
            self.squares[r2][c2] = moving
            cap_r = r2 + (1 if white else -1)
            self.squares[cap_r][c2] = EMPTY

        elif flag in PROMOTION_FLAGS:
            promo_piece = PROMOTION_PIECES[flag]
            self.squares[r1][c1] = EMPTY
            self.squares[r2][c2] = promo_piece if white else -promo_piece

        elif flag == FLAG_CASTLE_KINGSIDE:
            # Move king
            self.squares[r1][c1] = EMPTY
            self.squares[r2][c2] = moving
            # Move rook from h-file to f-file
            rook_col_from = 7
            rook_col_to   = 5
            rook = self.squares[r1][rook_col_from]
            self.squares[r1][rook_col_from] = EMPTY
            self.squares[r1][rook_col_to]   = rook

        elif flag == FLAG_CASTLE_QUEENSIDE:
            # Move king
            self.squares[r1][c1] = EMPTY
            self.squares[r2][c2] = moving
            # Move rook from a-file to d-file
            rook_col_from = 0
            rook_col_to   = 3
            rook = self.squares[r1][rook_col_from]
            self.squares[r1][rook_col_from] = EMPTY
            self.squares[r1][rook_col_to]   = rook

        else:
            # Normal move
            self.squares[r1][c1] = EMPTY
            self.squares[r2][c2] = moving

            # Double pawn push — set en passant target
            if abs(moving) == PAWN and abs(r2 - r1) == 2:
                self.en_passant_sq = ((r1 + r2) // 2, c1)

        # Update king cache
        if moving == KING:
            self.white_king_pos = (r2, c2)
        elif moving == -KING:
            self.black_king_pos = (r2, c2)

        # Update castling rights
        # King moves — lose both rights for that side
        if moving == KING:
            self.castle_wk = False
            self.castle_wq = False
        elif moving == -KING:
            self.castle_bk = False
            self.castle_bq = False

        # Rook moves or is captured — lose the relevant right
        if r1 == 7 and c1 == 7: self.castle_wk = False
        if r1 == 7 and c1 == 0: self.castle_wq = False
        if r1 == 0 and c1 == 7: self.castle_bk = False
        if r1 == 0 and c1 == 0: self.castle_bq = False

        # Rook captured on its starting square
        if r2 == 7 and c2 == 7: self.castle_wk = False
        if r2 == 7 and c2 == 0: self.castle_wq = False
        if r2 == 0 and c2 == 7: self.castle_bk = False
        if r2 == 0 and c2 == 0: self.castle_bq = False

        self.white_to_move = not self.white_to_move

    def undo_move(self):
        (move, captured,
         wkr, wkc, bkr, bkc,
         white_to_move,
         ep_r, ep_c,
         castle_wk, castle_wq,
         castle_bk, castle_bq) = self.move_stack.pop()

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
            # Restore king
            self.squares[r1][c1] = KING if white else -KING
            self.squares[r2][c2] = EMPTY
            # Restore rook
            rook = ROOK if white else -ROOK
            self.squares[r1][5] = EMPTY
            self.squares[r1][7] = rook

        elif flag == FLAG_CASTLE_QUEENSIDE:
            # Restore king
            self.squares[r1][c1] = KING if white else -KING
            self.squares[r2][c2] = EMPTY
            # Restore rook
            rook = ROOK if white else -ROOK
            self.squares[r1][3] = EMPTY
            self.squares[r1][0] = rook

        else:
            self.squares[r1][c1] = self.squares[r2][c2]
            self.squares[r2][c2] = captured

        # Restore all saved state
        self.white_king_pos = (wkr, wkc)
        self.black_king_pos = (bkr, bkc)
        self.white_to_move  = white_to_move
        self.en_passant_sq  = (ep_r, ep_c) if ep_r != -1 else None
        self.castle_wk      = castle_wk
        self.castle_wq      = castle_wq
        self.castle_bk      = castle_bk
        self.castle_bq      = castle_bq

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

        # Rooks / Queens (straight lines)
        for dr, dc in ROOK_DIRS:
            rr, cc = r + dr, c + dc
            while 0 <= rr < 8 and 0 <= cc < 8:
                piece = self.squares[rr][cc]
                if piece != EMPTY:
                    if by_white and piece in (ROOK, QUEEN):   return True
                    if not by_white and piece in (-ROOK, -QUEEN): return True
                    break
                rr += dr; cc += dc

        # Bishops / Queens (diagonals)
        for dr, dc in BISHOP_DIRS:
            rr, cc = r + dr, c + dc
            while 0 <= rr < 8 and 0 <= cc < 8:
                piece = self.squares[rr][cc]
                if piece != EMPTY:
                    if by_white and piece in (BISHOP, QUEEN):   return True
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