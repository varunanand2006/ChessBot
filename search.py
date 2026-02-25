from movegen import generate_all_moves
from constants import *
from movegen import generate_legal_moves

def evaluate(board):

    material_score = 0
    positional_score = 0
    total_non_pawn_material = 0

    for r in range(8):
        for c in range(8):
            piece = board.squares[r][c]
            if piece == EMPTY:
                continue

            abs_piece = abs(piece)
            value = PIECE_VALUES[abs_piece]

            if abs_piece != PAWN and abs_piece != KING:
                total_non_pawn_material += value

            # Material
            if piece > 0:
                material_score += value
            else:
                material_score -= value

            # Positional
            if abs_piece == PAWN:
                table_value = PAWN_TABLE[r][c] if piece > 0 else PAWN_TABLE[7-r][c]
            elif abs_piece == KNIGHT:
                table_value = KNIGHT_TABLE[r][c] if piece > 0 else KNIGHT_TABLE[7-r][c]
            elif abs_piece == BISHOP:
                table_value = BISHOP_TABLE[r][c] if piece > 0 else BISHOP_TABLE[7-r][c]
            elif abs_piece == ROOK:
                table_value = ROOK_TABLE[r][c] if piece > 0 else ROOK_TABLE[7-r][c]
            elif abs_piece == QUEEN:
                table_value = QUEEN_TABLE[r][c] if piece > 0 else QUEEN_TABLE[7-r][c]
            elif abs_piece == KING:
                if total_non_pawn_material < 1300:
                    table_value = KING_END_TABLE[r][c] if piece > 0 else KING_END_TABLE[7-r][c]
                else:
                    table_value = KING_TABLE[r][c] if piece > 0 else KING_TABLE[7-r][c]
            else:
                table_value = 0

            if piece > 0:
                positional_score += table_value
            else:
                positional_score -= table_value


    # Check penalty
    check_penalty = 0
    if board.is_in_check(True):
        check_penalty -= 50
    if board.is_in_check(False):
        check_penalty += 50

    return material_score + positional_score + check_penalty


def minimax(board, depth, alpha, beta):
    if depth == 0:
        return evaluate(board)

    legal_moves = generate_legal_moves(board)

    if not legal_moves:
        if board.is_in_check(board.white_to_move):
            return -99999  # checkmate
        return 0  # stalemate

    if board.white_to_move:
        max_eval = -float("inf")
        # Move ordering
        legal_moves.sort(key=lambda m: board.squares[m[2]][m[3]] != 0, reverse=True)
        for move in legal_moves:
            board.make_move(move)
            eval = minimax(board, depth-1, alpha, beta)
            board.undo_move()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float("inf")
        legal_moves.sort(key=lambda m: board.squares[m[2]][m[3]] != 0, reverse=True)
        for move in legal_moves:
            board.make_move(move)
            eval = minimax(board, depth-1, alpha, beta)
            board.undo_move()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def negamax(board, depth, alpha, beta):

    if depth == 0:
        return evaluate(board)

    legal_moves = generate_legal_moves(board)

    if not legal_moves:
        if board.is_in_check(board.white_to_move):
            return -99999  # checkmate
        return 0  # stalemate

    # Simple move ordering (captures first)
    legal_moves.sort(
        key=lambda m: abs(board.squares[m[2]][m[3]]),
        reverse=True
    )

    max_eval = -float("inf")

    for move in legal_moves:
        board.make_move(move)

        # Negamax principle:
        eval = -negamax(board, depth - 1, -beta, -alpha)

        board.undo_move()

        if eval > max_eval:
            max_eval = eval

        alpha = max(alpha, eval)

        if alpha >= beta:
            break  # alpha-beta pruning

    return max_eval