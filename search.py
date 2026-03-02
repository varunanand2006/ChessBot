from constants import *
from movegen import generate_legal_moves, decode_move

# ==========================================
# Evaluation
# ==========================================

def evaluate(board):
    material_score   = 0
    positional_score = 0
    total_material   = 0

    for r in range(8):
        for c in range(8):
            piece = board.squares[r][c]
            if piece == EMPTY:
                continue

            abs_piece = abs(piece)
            value     = PIECE_VALUES[abs_piece]

            # Track total non-king material for endgame detection
            if abs_piece != KING:
                total_material += value

            if piece > 0:
                material_score += value
            else:
                material_score -= value

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
                table = KING_END_TABLE if total_material < 1500 else KING_TABLE
            else:
                continue

            table_value = table[r][c] if piece > 0 else table[7 - r][c]

            if piece > 0:
                positional_score += table_value
            else:
                positional_score -= table_value

    # King distance penalty — when one side is up material,
    # reward the winning side for bringing kings closer together.
    # This helps convert winning endgames.
    king_distance_score = 0
    if total_material < 2000 and material_score != 0:
        wkr, wkc = board.white_king_pos
        bkr, bkc = board.black_king_pos
        king_distance = abs(wkr - bkr) + abs(wkc - bkc)

        # Winning side wants kings close, losing side wants them far
        if material_score > 0:
            king_distance_score = -king_distance * 5
        else:
            king_distance_score = king_distance * 5

    return material_score + positional_score + king_distance_score


# ==========================================
# Move Ordering
# ==========================================

def score_move(board, move):
    r1, c1, r2, c2, flag = decode_move(move)
    moving = board.squares[r1][c1]
    target = board.squares[r2][c2]

    score = 0

    # MVV-LVA
    if target != EMPTY:
        victim_value   = PIECE_VALUES[abs(target)]
        attacker_value = PIECE_VALUES[abs(moving)]
        score += 10000 + (victim_value * 10 - attacker_value)

    # Promotion bonus
    if flag in PROMOTION_FLAGS:
        score += PIECE_VALUES[PROMOTION_PIECES[flag]]

    return score


# ==========================================
# Quiescence Search
# ==========================================

def quiescence(board, alpha, beta, qdepth=4):
    if qdepth == 0:
        return evaluate(board)

    # Stand-pat: the side to move can always choose not to capture
    stand_pat = evaluate(board)

    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    legal_moves = generate_legal_moves(board)

    # Handle checkmate/stalemate in quiescence
    if not legal_moves:
        if board.is_in_check(board.white_to_move):
            return -99999 if board.white_to_move else 99999
        return 0

    # Captures only (including en passant)
    captures = [
        m for m in legal_moves
        if board.squares[decode_move(m)[2]][decode_move(m)[3]] != EMPTY
        or decode_move(m)[4] == FLAG_EN_PASSANT
    ]

    captures.sort(key=lambda m: score_move(board, m), reverse=True)

    for move in captures:
        board.make_move(move)
        score = -quiescence(board, -beta, -alpha, qdepth - 1)
        board.undo_move()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


# ==========================================
# Minimax with Alpha-Beta Pruning
# ==========================================

def minimax(board, depth, alpha, beta):
    if depth == 0:
        return quiescence(board, alpha, beta)

    legal_moves = generate_legal_moves(board)

    if not legal_moves:
        if board.is_in_check(board.white_to_move):
            # Prefer faster mates by adding depth to the score
            return -99999 - depth if board.white_to_move else 99999 + depth
        return 0  # Stalemate

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