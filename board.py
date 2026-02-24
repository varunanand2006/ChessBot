from constants import *


class Board:
    def __init__(self):
        # 8x8 board
        self.squares = [[0 for _ in range(8)] for _ in range(8)]

        # side to move (True = white, False = black)
        self.white_to_move = True

        # Cached king positions
        self.white_king_pos = None
        self.black_king_pos = None

        # Move stack for undo
        self.move_stack = []

    # ------------------------------------------------
    # Setup position (call this after placing pieces)
    # ------------------------------------------------
    def initialize_king_cache(self):
        for r in range(8):
            for c in range(8):
                piece = self.squares[r][c]
                if piece == KING:
                    self.white_king_pos = (r, c)
                elif piece == -KING:
                    self.black_king_pos = (r, c)

    # ------------------------------------------------
    # Make move
    # move = (r1, c1, r2, c2)
    # ------------------------------------------------
    def make_move(self, move):
        r1, c1, r2, c2 = move

        moving_piece = self.squares[r1][c1]
        captured_piece = self.squares[r2][c2]

        # Save state for undo
        move_info = {
            "move": move,
            "captured": captured_piece,
            "white_king_pos": self.white_king_pos,
            "black_king_pos": self.black_king_pos,
            "white_to_move": self.white_to_move
        }

        self.move_stack.append(move_info)

        # Move piece
        self.squares[r2][c2] = moving_piece
        self.squares[r1][c1] = EMPTY

        # Update king cache if needed
        if moving_piece == KING:
            self.white_king_pos = (r2, c2)
        elif moving_piece == -KING:
            self.black_king_pos = (r2, c2)

        # Switch turn
        self.white_to_move = not self.white_to_move

    # ------------------------------------------------
    # Undo move
    # ------------------------------------------------
    def undo_move(self):
        move_info = self.move_stack.pop()

        r1, c1, r2, c2 = move_info["move"]

        # Restore pieces
        moving_piece = self.squares[r2][c2]
        self.squares[r1][c1] = moving_piece
        self.squares[r2][c2] = move_info["captured"]

        # Restore king positions
        self.white_king_pos = move_info["white_king_pos"]
        self.black_king_pos = move_info["black_king_pos"]

        # Restore turn
        self.white_to_move = move_info["white_to_move"]

    # ------------------------------------------------
    # Get king position
    # ------------------------------------------------
    def get_king_position(self, white):
        return self.white_king_pos if white else self.black_king_pos