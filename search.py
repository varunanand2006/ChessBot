from constants import *
from movegen import generate_legal_moves, decode_move, move_to_string

# ==========================================
# Transposition Table
# ==========================================
# Persists across the entire game.
# Entry format: (depth, score, flag, best_move)

transposition_table = {}


def tt_lookup(board, depth, alpha, beta):
    """
    Returns (score, best_move) if score is usable,
    (None, best_move) if only the move is useful for ordering,
    (None, None) if no entry exists.
    """
    entry = transposition_table.get(board.hash)
    if entry is None:
        return None, None

    tt_depth, tt_score, tt_flag, tt_move = entry

    if tt_depth >= depth:
        if tt_flag == TT_EXACT:
            return tt_score, tt_move
        elif tt_flag == TT_LOWER_BOUND and tt_score >= beta:
            return tt_score, tt_move
        elif tt_flag == TT_UPPER_BOUND and tt_score <= alpha:
            return tt_score, tt_move

    return None, tt_move


def tt_store(board, depth, score, flag, best_move):
    transposition_table[board.hash] = (depth, score, flag, best_move)


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

    # Check penalty
    check_penalty = 0
    if board.is_in_check(True):
        check_penalty -= 50
    if board.is_in_check(False):
        check_penalty += 50

    # King distance penalty in endgame
    king_distance_score = 0
    if total_material < 2000 and material_score != 0:
        wkr, wkc = board.white_king_pos
        bkr, bkc = board.black_king_pos
        king_distance = abs(wkr - bkr) + abs(wkc - bkc)
        king_distance_score = -king_distance * 5 if material_score > 0 else king_distance * 5

    return material_score + positional_score + check_penalty + king_distance_score


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
# Uses the same convention as minimax: scores are always from
# white's perspective (positive = good for white). Separate
# white/black branches match minimax exactly.

def quiescence(board, alpha, beta, qdepth=3):
    if qdepth == 0:
        return evaluate(board)

    stand_pat = evaluate(board)

    # Stand-pat and delta pruning, side-aware
    if board.white_to_move:
        if stand_pat + 900 < alpha:
            return alpha
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
    else:
        if stand_pat - 900 > beta:
            return beta
        if stand_pat <= alpha:
            return alpha
        if stand_pat < beta:
            beta = stand_pat

    legal_moves = generate_legal_moves(board)

    if not legal_moves:
        if board.is_in_check(board.white_to_move):
            return -99999 if board.white_to_move else 99999
        return 0

    # Captures + en passant only
    captures = [
        m for m in legal_moves
        if board.squares[decode_move(m)[2]][decode_move(m)[3]] != EMPTY
        or decode_move(m)[4] == FLAG_EN_PASSANT
    ]
    captures.sort(key=lambda m: score_move(board, m), reverse=True)

    if board.white_to_move:
        for move in captures:
            board.make_move(move)
            score = quiescence(board, alpha, beta, qdepth - 1)
            board.undo_move()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    else:
        for move in captures:
            board.make_move(move)
            score = quiescence(board, alpha, beta, qdepth - 1)
            board.undo_move()
            if score <= alpha:
                return alpha
            if score < beta:
                beta = score
        return beta


# ==========================================
# Minimax with Alpha-Beta + Transposition Table
# ==========================================

def minimax(board, depth, alpha, beta):
    original_alpha = alpha
    original_beta  = beta

    # Threefold repetition — if this position has been seen twice before,
    # this would be the third occurrence. Score as draw.
    if board.position_history.get(board.hash, 0) >= 2:
        return 0

    # TT lookup
    tt_score, tt_move = tt_lookup(board, depth, alpha, beta)
    if tt_score is not None:
        return tt_score

    if depth == 0:
        return quiescence(board, alpha, beta)

    legal_moves = generate_legal_moves(board)

    if not legal_moves:
        if board.is_in_check(board.white_to_move):
            return -99999 - depth if board.white_to_move else 99999 + depth
        return 0

    # TT move first, then MVV-LVA
    if tt_move is not None and tt_move in legal_moves:
        legal_moves.remove(tt_move)
        legal_moves.insert(0, tt_move)
    legal_moves[1:] = sorted(
        legal_moves[1:],
        key=lambda m: score_move(board, m),
        reverse=True
    )

    best_move = legal_moves[0]

    if board.white_to_move:
        max_score = -float("inf")
        for move in legal_moves:
            board.make_move(move)
            score = minimax(board, depth - 1, alpha, beta)
            board.undo_move()
            if score > max_score:
                max_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if beta <= alpha:
                break

        # Fail low = upper bound, fail high = lower bound, else exact
        if max_score <= original_alpha:
            flag = TT_UPPER_BOUND
        elif max_score >= beta:
            flag = TT_LOWER_BOUND
        else:
            flag = TT_EXACT
        tt_store(board, depth, max_score, flag, best_move)
        return max_score

    else:
        min_score = float("inf")
        for move in legal_moves:
            board.make_move(move)
            score = minimax(board, depth - 1, alpha, beta)
            board.undo_move()
            if score < min_score:
                min_score = score
                best_move = move
            if score < beta:
                beta = score
            if beta <= alpha:
                break

        # For minimizer: compare against original_beta and current alpha
        if min_score >= original_beta:
            flag = TT_LOWER_BOUND
        elif min_score <= alpha:
            flag = TT_UPPER_BOUND
        else:
            flag = TT_EXACT
        tt_store(board, depth, min_score, flag, best_move)
        return min_score


# ==========================================
# Root Search with Iterative Deepening
# ==========================================

def find_best_move(board, max_depth):
    best_move = None

    for depth in range(1, max_depth + 1):
        legal_moves = generate_legal_moves(board)

        if not legal_moves:
            return None

        # Previous best move goes first — TT move is a fallback,
        # but the root is never stored in the TT so best_move from
        # the last completed iteration is the reliable source.
        _, tt_move = tt_lookup(board, depth, -float("inf"), float("inf"))
        priority_move = best_move if best_move is not None else tt_move
        if priority_move is not None and priority_move in legal_moves:
            legal_moves.remove(priority_move)
            legal_moves.insert(0, priority_move)

        legal_moves[1:] = sorted(
            legal_moves[1:],
            key=lambda m: score_move(board, m),
            reverse=True
        )

        alpha     = -float("inf")
        beta      =  float("inf")
        best_move = legal_moves[0]

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

        print(f"  depth {depth} -> score {best_score:+d}  best: {move_to_string(best_move, board)}")

    return best_move