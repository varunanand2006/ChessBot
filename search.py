from constants import *
from movegen import generate_legal_moves, decode_move

# ==========================================
# Evaluation
# ==========================================

def evaluate(board):
    material_score   = 0
    positional_score = 0

    for r in range(8):
        for c in range(8):
            piece = board.squares[r][c]
            if piece == EMPTY:
                continue

            abs_piece = abs(piece)
            value     = PIECE_VALUES[abs_piece]

            # Material
            if piece > 0:
                material_score += value
            else:
                material_score -= value

            # Positional
            if abs_piece == PAWN:
                table = PAWN_TABLE
            elif abs_piece == KNIGHT:
                table = KNIGHT_TABLE
            elif abs_piece == BISHOP:
                table = BISHOP_TABLE
            elif abs_piece == ROOK:
                table = ROOK_TABLE
            elif abs_piece == QUEEN:
                table = QUEEN_TABLE
            elif abs_piece == KING:
                table = KING_TABLE
            else:
                continue

            table_value = table[r][c] if piece > 0 else table[7 - r][c]

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


# ==========================================
# Move Ordering
# ==========================================

def score_move(board, move):
    r1, c1, r2, c2, flag = decode_move(move)
    moving = board.squares[r1][c1]
    target = board.squares[r2][c2]

    score = 0

    # MVV-LVA: prioritize capturing high value pieces with low value pieces
    if target != EMPTY:
        victim_value   = PIECE_VALUES[abs(target)]
        attacker_value = PIECE_VALUES[abs(moving)]
        score += 10000 + (victim_value * 10 - attacker_value)

    # Promotions are likely good moves
    if flag in PROMOTION_FLAGS:
        score += PIECE_VALUES[PROMOTION_PIECES[flag]]

    return score


# ==========================================
# Minimax with Alpha-Beta Pruning
# ==========================================

def minimax(board, depth, alpha, beta):
    if depth == 0:
        return evaluate(board)

    legal_moves = generate_legal_moves(board)

    if not legal_moves:
        if board.is_in_check(board.white_to_move):
            # Checkmate — worst outcome for the side to move
            return -99999 if board.white_to_move else 99999
        return 0  # Stalemate

    # Move ordering
    legal_moves.sort(key=lambda move: score_move(board, move), reverse=True)

    if board.white_to_move:
        max_score = -float("inf")
        for move in legal_moves:
            board.make_move(move)
            score = minimax(board, depth - 1, alpha, beta)
            board.undo_move()
            if score > max_score:
                max_score = score
            if score > alpha:
                alpha = score
            if beta <= alpha:
                break
        return max_score

    else:
        min_score = float("inf")
        for move in legal_moves:
            board.make_move(move)
            score = minimax(board, depth - 1, alpha, beta)
            board.undo_move()
            if score < min_score:
                min_score = score
            if score < beta:
                beta = score
            if beta <= alpha:
                break
        return min_score


# ==========================================
# Root Search
# ==========================================

def find_best_move(board, depth):
    legal_moves = generate_legal_moves(board)

    if not legal_moves:
        return None

    # Move ordering at root
    legal_moves.sort(key=lambda move: score_move(board, move), reverse=True)

    best_move  = legal_moves[0]
    alpha      = -float("inf")
    beta       =  float("inf")

    if board.white_to_move:
        best_score = -float("inf")
        for move in legal_moves:
            board.make_move(move)
            score = minimax(board, depth - 1, alpha, beta)
            board.undo_move()
            if score > best_score:
                best_score = score
                best_move  = move
            alpha = max(alpha, score)
    else:
        best_score = float("inf")
        for move in legal_moves:
            board.make_move(move)
            score = minimax(board, depth - 1, alpha, beta)
            board.undo_move()
            if score < best_score:
                best_score = score
                best_move  = move
            beta = min(beta, score)

    return best_move