from board import Board
from search import *

# Takes in a string 'e2e4' and parses it into move coordinates
def parse_move(move_str):
    # expects format like "e2e4"
    from_file = ord(move_str[0]) - ord('a')
    from_rank = 8 - int(move_str[1])
    to_file   = ord(move_str[2]) - ord('a')
    to_rank   = 8 - int(move_str[3])
    return (from_rank, from_file, to_rank, to_file, QUIET)


board = Board()
board.setup_starting_position()
board.initialize_king_cache()
#print(board)

# Setup starting position here
# (You should already have this function)
# board.setup_starting_position()
#

while True:
    print(board)
    #print(evaluate(board))

    if board.white_to_move:
        move_input = input("Your move (e2e4): ")

        if len(move_input) != 4:
            print("Invalid format.")
            continue

        move = parse_move(move_input)
        print(str(move))
        v1, v2, v3, v4, v5 = move

        foundMove = False
        legal_moves = generate_legal_moves(board)
        for m in legal_moves:
            if v1 == m[0] and v2 == m[1] and v3 == m[2] and v4 == m[3]:
                move = m
                foundMove = True
                break


        if not foundMove:
            print("Illegal move.")
            continue
        board.make_move(move)


    else:
        print("Bot thinking...")
        moves = generate_legal_moves(board)

        if not moves:
            print("Game over.")
            break
        best_move = None

        if board.white_to_move:
            best_score = -float("inf")
        else:
            best_score = float("inf")

        for move in moves:
            board.make_move(move)
            score = minimax(board, depth=4, alpha=-float("inf"), beta=float("inf"))
            board.undo_move()

            if board.white_to_move:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

        board.make_move(best_move)