from board import Board
from search import *



def parse_move(move_str):
    # expects format like "e2e4"
    from_file = ord(move_str[0]) - ord('a')
    from_rank = 8 - int(move_str[1])
    to_file   = ord(move_str[2]) - ord('a')
    to_rank   = 8 - int(move_str[3])
    return (from_rank, from_file, to_rank, to_file)

board = Board()
board.setup_starting_position()
print(board)

# Setup starting position here
# (You should already have this function)
# board.setup_starting_position()
# board.initialize_king_cache()

while True:
    print(board)
    #print(evaluate(board))

    if board.white_to_move:
        move_input = input("Your move (e2e4): ")

        if len(move_input) != 4:
            print("Invalid format.")
            continue

        move = parse_move(move_input)

        legal_moves = generate_legal_moves(board)

        if move not in legal_moves:
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
            score = minimax(board, depth=3, alpha=-float("inf"), beta=float("inf"))
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