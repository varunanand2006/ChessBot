from constants import *
from movegen import decode_move, move_from_to

class Board:
    def __init__(self):
        self.squares = [[EMPTY] * 8 for _ in range(8)]
        self.white_to_move = True

        self.white_king_pos = None
        self.black_king_pos = None

        # (r, c) of the en passant target square, or None
        self.en_passant_sq = None

        # Each entry is a tuple:
        # (move, captured, wkr, wkc, bkr, bkc, white_to_move, ep_r, ep_c)
        # ep_r == -1 means no en passant square
        self.move_stack = []

    # ==========================================
    # Setup
    # ==========================================

    def setup_starting_position(self):
        self.squares = [[EMPTY] * 8 for _ in range(8)]

        # Pawns
        for c in range(8):
            self.squares[6][c] =  PAWN
            self.squares[1][c] = -PAWN

        # Rooks
        self.squares[7][0] =  ROOK;  self.squares[7][7] =  ROOK
        self.squares[0][0] = -ROOK;  self.squares[0][7] = -ROOK

        # Knights
        self.squares[7][1] =  KNIGHT;  self.squares[7][6] =  KNIGHT
        self.squares[0][1] = -KNIGHT;  self.squares[0][6] = -KNIGHT

        # Bishops
        self.squares[7][2] =  BISHOP;  self.squares[7][5] =  BISHOP
        self.squares[0][2] = -BISHOP;  self.squares[0][5] = -BISHOP

        # Queens
        self.squares[7][3] =  QUEEN
        self.squares[0][3] = -QUEEN

        # Kings
        self.squares[7][4] =  KING
        self.squares[0][4] = -KING

        self.white_king_pos = (7, 4)
        self.black_king_pos = (0, 4)
        self.white_to_move  = True
        self.en_passant_sq  = None
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

        # Save current state onto the stack
        wkr, wkc = self.white_king_pos
        bkr, bkc = self.black_king_pos
        ep_r, ep_c = self.en_passant_sq if self.en_passant_sq else (-1, -1)

        self.move_stack.append((
            move, captured,
            wkr, wkc, bkr, bkc,
            self.white_to_move,
            ep_r, ep_c
        ))

        # Clear en passant — will be re-set below if double push
        self.en_passant_sq = None

        # Move the piece
        self.squares[r1][c1] = EMPTY

        if flag == FLAG_EN_PASSANT:
            # The captured pawn is behind the destination square
            self.squares[r2][c2] = moving
            cap_r = r2 + (1 if white else -1)
            self.squares[cap_r][c2] = EMPTY

        elif flag in PROMOTION_FLAGS:
            promo_piece = PROMOTION_PIECES[flag]
            self.squares[r2][c2] = promo_piece if white else -promo_piece

        else:
            self.squares[r2][c2] = moving

            # Double pawn push — set en passant target square
            if abs(moving) == PAWN and abs(r2 - r1) == 2:
                self.en_passant_sq = ((r1 + r2) // 2, c1)

        # Update king cache
        if moving == KING:
            self.white_king_pos = (r2, c2)
        elif moving == -KING:
            self.black_king_pos = (r2, c2)

        self.white_to_move = not self.white_to_move

    def undo_move(self):
        (move, captured,
         wkr, wkc, bkr, bkc,
         white_to_move,
         ep_r, ep_c) = self.move_stack.pop()

        r1, c1, r2, c2, flag = decode_move(move)

        moving = self.squares[r2][c2]
        white  = white_to_move  # the side that made the move

        # Restore the moving piece to its origin
        # For promotions, restore original pawn not the promoted piece
        if flag in PROMOTION_FLAGS:
            self.squares[r1][c1] = PAWN if white else -PAWN
        else:
            self.squares[r1][c1] = moving

        # Restore destination square
        self.squares[r2][c2] = captured

        # En passant: restore the captured pawn
        if flag == FLAG_EN_PASSANT:
            cap_r = r2 + (1 if white else -1)
            self.squares[cap_r][c2] = -PAWN if white else PAWN

        # Restore saved state
        self.white_king_pos = (wkr, wkc)
        self.black_king_pos = (bkr, bkc)
        self.white_to_move  = white_to_move
        self.en_passant_sq  = (ep_r, ep_c) if ep_r != -1 else None

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

        # --- Pawns ---
        direction = 1 if by_white else -1
        for dc in (-1, 1):
            rr, cc = r + direction, c + dc
            if 0 <= rr < 8 and 0 <= cc < 8:
                if self.squares[rr][cc] == (PAWN if by_white else -PAWN):
                    return True

        # --- Knights ---
        for dr, dc in KNIGHT_OFFSETS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < 8 and 0 <= cc < 8:
                if self.squares[rr][cc] == (KNIGHT if by_white else -KNIGHT):
                    return True

        # --- King ---
        for dr, dc in KING_OFFSETS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < 8 and 0 <= cc < 8:
                if self.squares[rr][cc] == (KING if by_white else -KING):
                    return True

        # --- Sliding pieces ---
        for dr, dc in ROOK_DIRS:
            rr, cc = r + dr, c + dc
            while 0 <= rr < 8 and 0 <= cc < 8:
                piece = self.squares[rr][cc]
                if piece != EMPTY:
                    if by_white and piece in ( ROOK,  QUEEN): return True
                    if not by_white and piece in (-ROOK, -QUEEN): return True
                    break
                rr += dr
                cc += dc

        for dr, dc in BISHOP_DIRS:
            rr, cc = r + dr, c + dc
            while 0 <= rr < 8 and 0 <= cc < 8:
                piece = self.squares[rr][cc]
                if piece != EMPTY:
                    if by_white and piece in ( BISHOP,  QUEEN): return True
                    if not by_white and piece in (-BISHOP, -QUEEN): return True
                    break
                rr += dr
                cc += dc

        return False

    # ==========================================
    # Display
    # ==========================================

    """def __str__(self):
        piece_symbols = {
             PAWN: "♟",  -PAWN: "♙",
           KNIGHT: "♞",-KNIGHT: "♘",
           BISHOP: "♝",-BISHOP: "♗",
             ROOK: "♜",   -ROOK: "♖",
            QUEEN: "♛",  -QUEEN: "♕",
             KING: "♚",   -KING: "♔",
            EMPTY: " "
        }

        rows = []
        for r in range(8):
            row = str(8 - r) + "  "
            for c in range(8):
                symbol = piece_symbols[self.squares[r][c]]
                row += f"[{symbol}]" if (r + c) % 2 == 0 else f" {symbol} "
            rows.append(row)

        turn = "White to move" if self.white_to_move else "Black to move"
        return "\n" + "\n".join(rows) + "\n    a  b  c  d  e  f  g  h\n" + turn + "\n"
        """

    def __str__(self):
        result = "  ╔═════════════════════════════╗\n"

        for r in range(8):
            result += str(8 - r) + " ║"

            for c in range(8):
                piece = self.squares[r][c]
                symbol = piece_symbols[piece]

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