from board import Board
from movegen import generate_legal_moves, encode_move, decode_move, move_to_string
from search import find_best_move
from constants import *

DEPTH = 4

def parse_move(move_str):
    from_col = ord(move_str[0]) - ord('a')
    from_row = 8 - int(move_str[1])
    to_col   = ord(move_str[2]) - ord('a')
    to_row   = 8 - int(move_str[3])
    return encode_move(from_row, from_col, to_row, to_col)

board = Board()
board.setup_starting_position()

"""
for i in range (0, 8):
    board.squares[1][i] = 0
board.squares[0][0] = 0
board.squares[0][1] = 0
board.squares[0][2] = 0
board.squares[0][3] = 0
board.squares[0][5] = 0
board.squares[0][6] = 0
board.squares[0][7] = 0
"""


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

        # Auto-queen if pawn reaches back rank
        # Later: replace with input prompt to choose promotion piece
        r1, c1, r2, c2, flag = decode_move(move)
        if abs(board.squares[r1][c1]) == PAWN and r2 == (0 if board.white_to_move else 7):
            move = encode_move(r1, c1, r2, c2, FLAG_PROMOTE_QUEEN)

        if move not in legal_moves:
            print("Illegal move.")
            continue

        board.make_move(move)

    else:
        print("Bot thinking...")
        best_move = find_best_move(board, DEPTH)
        print(move_to_string(best_move, board))
        board.make_move(best_move)