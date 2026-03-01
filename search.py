from constants import *
from movegen import generate_legal_moves
from board import *

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

def score_move(board, move):
    r1, c1, r2, c2 = move
    moving = board.squares[r1][c1]
    target = board.squares[r2][c2]

    score = 0

    # Captures first (MVV-LVA)
    if target != EMPTY:
        victim_value = abs(target)
        attacker_value = abs(moving)

        score += 10000 + (victim_value * 10 - attacker_value)

    return score

def minimax(board, depth, alpha, beta):
    if depth == 0:
        return evaluate(board)

    legal_moves = generate_legal_moves(board)

    if not legal_moves:
        if board.is_in_check(board.white_to_move):
            # Side to move is checkmated
            if board.white_to_move:
                return -99999 # White is mated → bad for White
            else:
                return 99999  # Black is mated → good for White
        return 0  # stalemate

    # ===== MOVE ORDERING =====
    legal_moves.sort(key=lambda move: score_move(board, move), reverse=True)

    if board.white_to_move:
        max_eval = -float("inf")
        # Move ordering
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
        for move in legal_moves:
            board.make_move(move)
            eval = minimax(board, depth-1, alpha, beta)
            board.undo_move()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

board = Board()

board.squares[0][0] = KING
board.squares[1][1] = -KNIGHT

board.white_king_pos = (0, 0)
board.black_king_pos = (7, 7)  # put black king somewhere safe
board.initialize_king_cache()

print(board.is_in_check(True))