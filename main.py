import time
from board import Board
from movegen import generate_legal_moves, encode_move, decode_move, move_to_string
from search import find_best_move
from constants import *

import sys
print(sys.version)

DEPTH = 4

def parse_move(move_str):
    from_col = ord(move_str[0]) - ord('a')
    from_row = 8 - int(move_str[1])
    to_col   = ord(move_str[2]) - ord('a')
    to_row   = 8 - int(move_str[3])
    return encode_move(from_row, from_col, to_row, to_col)

board = Board()
board.setup_starting_position()

while True:
    print(board)

    legal_moves = generate_legal_moves(board)

    if not legal_moves:
        if board.is_in_check(board.white_to_move):
            winner = "Black" if board.white_to_move else "White"
            print(f"Checkmate! {winner} wins.")
        else:
            print("Stalemate! It's a draw.")
        break

    if board.white_to_move:
        move_input = input("Your move (e2e4): ")

        if len(move_input) != 4:
            print("Invalid format.")
            continue

        move = parse_move(move_input)
        r1, c1, r2, c2, flag = decode_move(move)

        # Promotion
        if abs(board.squares[r1][c1]) == PAWN and r2 == (0 if board.white_to_move else 7):
            while True:
                promo_input = input("Promote to (q/r/b/n): ").lower()
                if promo_input in ("q", "r", "b", "n"):
                    break
            flag = {"q": FLAG_PROMOTE_QUEEN, "r": FLAG_PROMOTE_ROOK,
                    "b": FLAG_PROMOTE_BISHOP, "n": FLAG_PROMOTE_KNIGHT}[promo_input]
            move = encode_move(r1, c1, r2, c2, flag)

        # En passant
        elif (abs(board.squares[r1][c1]) == PAWN
                and board.en_passant_sq is not None
                and (r2, c2) == board.en_passant_sq):
            move = encode_move(r1, c1, r2, c2, FLAG_EN_PASSANT)

        # Castling
        elif abs(board.squares[r1][c1]) == KING and abs(c2 - c1) == 2:
            if c2 > c1:
                move = encode_move(r1, c1, r2, c2, FLAG_CASTLE_KINGSIDE)
            else:
                move = encode_move(r1, c1, r2, c2, FLAG_CASTLE_QUEENSIDE)

        if move not in legal_moves:
            print("Illegal move.")
            continue

        board.make_move(move)

    else:
        print("Bot thinking...")
        start = time.time()
        best_move = find_best_move(board, DEPTH)
        elapsed = time.time() - start
        print(f"  Time: {elapsed:.2f}s")
        print(f"  {move_to_string(best_move, board)}")
        board.make_move(best_move)