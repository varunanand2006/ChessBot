from constants import *

class Board:
    def __init__(self):
        self.squares = [[0 for _ in range(8)] for _ in range(8)]
        self.white_to_move = True

        self.white_king_pos = None
        self.black_king_pos = None

        self.move_stack = []

    def setup_starting_position(self):

        # Clear board
        for r in range(8):
            for c in range(8):
                self.squares[r][c] = 0

        # -------------------------
        # Pawns
        # -------------------------
        for c in range(8):
            self.squares[6][c] = PAWN  # White pawns
            self.squares[1][c] = -PAWN  # Black pawns

        # -------------------------
        # Rooks
        # -------------------------
        self.squares[7][0] = ROOK
        self.squares[7][7] = ROOK
        self.squares[0][0] = -ROOK
        self.squares[0][7] = -ROOK

        # -------------------------
        # Knights
        # -------------------------
        self.squares[7][1] = KNIGHT
        self.squares[7][6] = KNIGHT
        self.squares[0][1] = -KNIGHT
        self.squares[0][6] = -KNIGHT

        # -------------------------
        # Bishops
        # -------------------------
        self.squares[7][2] = BISHOP
        self.squares[7][5] = BISHOP
        self.squares[0][2] = -BISHOP
        self.squares[0][5] = -BISHOP

        # -------------------------
        # Queens
        # -------------------------
        self.squares[7][3] = QUEEN
        self.squares[0][3] = -QUEEN

        # -------------------------
        # Kings
        # -------------------------
        self.squares[7][4] = KING
        self.squares[0][4] = -KING

        # -------------------------
        # Initialize king cache
        # -------------------------
        self.white_king_pos = (7, 4)
        self.black_king_pos = (0, 4)

        # White starts
        self.white_to_move = True

        # Clear move history
        self.move_stack = []

    # ------------------------------
    # Setup
    # ------------------------------
    def initialize_king_cache(self):
        for r in range(8):
            for c in range(8):
                piece = self.squares[r][c]
                if piece == KING:
                    self.white_king_pos = (r, c)
                elif piece == -KING:
                    self.black_king_pos = (r, c)

    # ------------------------------
    # Make / Undo
    # ------------------------------
    def make_move(self, move):
        r1, c1, r2, c2 = move
        moving = self.squares[r1][c1]
        captured = self.squares[r2][c2]

        self.move_stack.append({
            "move": move,
            "captured": captured,
            "white_king_pos": self.white_king_pos,
            "black_king_pos": self.black_king_pos,
            "white_to_move": self.white_to_move
        })

        self.squares[r1][c1] = EMPTY
        # Auto-queen promotion
        if moving == PAWN and r2 == 0:
            self.squares[r2][c2] = QUEEN
        elif moving == -PAWN and r2 == 7:
            self.squares[r2][c2] = -QUEEN
        else:
            self.squares[r2][c2] = moving

        if moving == KING:
            self.white_king_pos = (r2, c2)
        elif moving == -KING:
            self.black_king_pos = (r2, c2)

        self.white_to_move = not self.white_to_move

    def undo_move(self):
        info = self.move_stack.pop()
        r1, c1, r2, c2 = info["move"]

        moving = self.squares[r2][c2]

        self.squares[r1][c1] = moving
        self.squares[r2][c2] = info["captured"]

        self.white_king_pos = info["white_king_pos"]
        self.black_king_pos = info["black_king_pos"]
        self.white_to_move = info["white_to_move"]

    # ------------------------------
    # King Helpers
    # ------------------------------
    def get_king_position(self, white):
        return self.white_king_pos if white else self.black_king_pos

    def is_in_check(self, white):
        king_pos = self.get_king_position(white)
        return self.is_square_attacked(king_pos, not white)

    # ------------------------------
    # Attack Detection
    # ------------------------------
    def is_square_attacked(self, square, by_white):
        r, c = square

        # Pawn attacks
        direction = -1 if by_white else 1
        for dc in (-1, 1):
            rr = r + direction
            cc = c + dc
            if 0 <= rr < 8 and 0 <= cc < 8:
                piece = self.squares[rr][cc]
                if piece == (PAWN if by_white else -PAWN):
                    return True

        # Knight attacks
        knight_offsets = [
            (2,1),(2,-1),(-2,1),(-2,-1),
            (1,2),(1,-2),(-1,2),(-1,-2)
        ]
        for dr, dc in knight_offsets:
            rr, cc = r+dr, c+dc
            if 0 <= rr < 8 and 0 <= cc < 8:
                piece = self.squares[rr][cc]
                if piece == (KNIGHT if by_white else -KNIGHT):
                    return True

        # King attacks (adjacent squares)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = r + dr
                cc = c + dc
                if 0 <= rr < 8 and 0 <= cc < 8:
                    piece = self.squares[rr][cc]
                    if piece == (KING if by_white else -KING):
                        return True


        # Sliding pieces
        directions = [
            (1,0),(-1,0),(0,1),(0,-1),
            (1,1),(1,-1),(-1,1),(-1,-1)
        ]

        for dr, dc in directions:
            rr, cc = r+dr, c+dc
            while 0 <= rr < 8 and 0 <= cc < 8:
                piece = self.squares[rr][cc]
                if piece != EMPTY:
                    if by_white and piece > 0:
                        if (dr == 0 or dc == 0) and piece in (ROOK, QUEEN):
                            return True
                        if (dr != 0 and dc != 0) and piece in (BISHOP, QUEEN):
                            return True
                    if not by_white and piece < 0:
                        if (dr == 0 or dc == 0) and piece in (-ROOK, -QUEEN):
                            return True
                        if (dr != 0 and dc != 0) and piece in (-BISHOP, -QUEEN):
                            return True
                    break
                rr += dr
                cc += dc

        return False

    def __str__(self):
        piece_symbols = {
            -1: "♙", 1: "♟",
            -2: "♘", 2: "♞",
            -3: "♗", 3: "♝",
            -4: "♖", 4: "♜",
            -5: "♕", 5: "♛",
            -6: "♔", 6: "♚",
            0: " "
        }

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