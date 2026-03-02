from constants import *
from movegen import generate_legal_moves, decode_move, move_to_string

MATE_SCORE = 99999

transposition_table = {}


def score_to_tt(score, ply):
    if score > MATE_SCORE - 200:
        return score + ply
    if score < -MATE_SCORE + 200:
        return score - ply
    return score


def score_from_tt(score, ply):
    if score > MATE_SCORE - 200:
        return score - ply
    if score < -MATE_SCORE + 200:
        return score + ply
    return score


def tt_lookup(board, depth, alpha, beta, ply):
    entry = transposition_table.get(board.hash)
    if entry is None:
        return None, None
    tt_depth, tt_score, tt_flag, tt_move = entry
    if tt_depth >= depth:
        score = score_from_tt(tt_score, ply)
        if tt_flag == TT_EXACT:
            return score, tt_move
        elif tt_flag == TT_LOWER_BOUND and score >= beta:
            return score, tt_move
        elif tt_flag == TT_UPPER_BOUND and score <= alpha:
            return score, tt_move
    return None, tt_move


def tt_store(board, depth, score, flag, best_move, ply):
    transposition_table[board.hash] = (depth, score_to_tt(score, ply), flag, best_move)


def evaluate(board):
    total_material = 0
    for r in range(8):
        for c in range(8):
            abs_piece = abs(board.squares[r][c])
            if abs_piece != EMPTY and abs_piece != KING:
                total_material += PIECE_VALUES[abs_piece]

    material_score   = 0
    positional_score = 0

    for r in range(8):
        for c in range(8):
            piece = board.squares[r][c]
            if piece == EMPTY:
                continue
            abs_piece = abs(piece)
            if abs_piece == KING:
                continue
            value = PIECE_VALUES[abs_piece]
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
            else:
                table = QUEEN_TABLE
            if piece > 0:
                positional_score += table[r][c]
            else:
                positional_score -= table[7 - r][c]

    king_table = KING_END_TABLE if total_material < 1500 else KING_TABLE
    wkr, wkc = board.white_king_pos
    bkr, bkc = board.black_king_pos
    positional_score += king_table[wkr][wkc]
    positional_score -= king_table[7 - bkr][bkc]

    king_distance_score = 0
    if total_material < 2000 and material_score != 0:
        king_distance = abs(wkr - bkr) + abs(wkc - bkc)
        king_distance_score = -king_distance * 5 if material_score > 0 else king_distance * 5

    return material_score + positional_score + king_distance_score


def score_move(board, move):
    r1, c1, r2, c2, flag = decode_move(move)
    moving = board.squares[r1][c1]
    target = board.squares[r2][c2]
    score = 0
    if target != EMPTY:
        victim_value   = PIECE_VALUES[abs(target)]
        attacker_value = PIECE_VALUES[abs(moving)]
        score += 10000 + (victim_value * 10 - attacker_value)
    if flag in PROMOTION_FLAGS:
        score += PIECE_VALUES[PROMOTION_PIECES[flag]]
    return score


def quiescence(board, alpha, beta, ply, qdepth=3):
    if qdepth == 0:
        return evaluate(board)

    stand_pat = evaluate(board)

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
            return -MATE_SCORE if board.white_to_move else MATE_SCORE
        return 0

    captures = [
        m for m in legal_moves
        if board.squares[decode_move(m)[2]][decode_move(m)[3]] != EMPTY
        or decode_move(m)[4] == FLAG_EN_PASSANT
    ]
    captures.sort(key=lambda m: score_move(board, m), reverse=True)

    if board.white_to_move:
        for move in captures:
            board.make_move(move)
            score = quiescence(board, alpha, beta, ply + 1, qdepth - 1)
            board.undo_move()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha
    else:
        for move in captures:
            board.make_move(move)
            score = quiescence(board, alpha, beta, ply + 1, qdepth - 1)
            board.undo_move()
            if score <= alpha:
                return alpha
            if score < beta:
                beta = score
        return beta


def minimax(board, depth, alpha, beta, ply):
    original_alpha = alpha
    original_beta  = beta

    tt_score, tt_move = tt_lookup(board, depth, alpha, beta, ply)
    if tt_score is not None:
        return tt_score

    if depth == 0:
        return quiescence(board, alpha, beta, ply)

    legal_moves = generate_legal_moves(board)

    if not legal_moves:
        if board.is_in_check(board.white_to_move):
            return (-MATE_SCORE - depth) if board.white_to_move else (MATE_SCORE + depth)
        return 0

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
            score = minimax(board, depth - 1, alpha, beta, ply + 1)
            board.undo_move()
            if score > max_score:
                max_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if beta <= alpha:
                break
        if max_score <= original_alpha:
            flag = TT_UPPER_BOUND
        elif max_score >= beta:
            flag = TT_LOWER_BOUND
        else:
            flag = TT_EXACT
        tt_store(board, depth, max_score, flag, best_move, ply)
        return max_score

    else:
        min_score = float("inf")
        for move in legal_moves:
            board.make_move(move)
            score = minimax(board, depth - 1, alpha, beta, ply + 1)
            board.undo_move()
            if score < min_score:
                min_score = score
                best_move = move
            if score < beta:
                beta = score
            if beta <= alpha:
                break
        if min_score >= original_beta:
            flag = TT_LOWER_BOUND
        elif min_score <= alpha:
            flag = TT_UPPER_BOUND
        else:
            flag = TT_EXACT
        tt_store(board, depth, min_score, flag, best_move, ply)
        return min_score


def find_best_move(board, max_depth):
    best_move = None

    for depth in range(1, max_depth + 1):
        legal_moves = generate_legal_moves(board)
        if not legal_moves:
            return None

        _, tt_move = tt_lookup(board, depth, -float("inf"), float("inf"), 0)
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
                score = minimax(board, depth - 1, alpha, beta, 1)
                board.undo_move()
                if score > best_score:
                    best_score = score
                    best_move  = move
                alpha = max(alpha, score)
        else:
            best_score = float("inf")
            for move in legal_moves:
                board.make_move(move)
                score = minimax(board, depth - 1, alpha, beta, 1)
                board.undo_move()
                if score < best_score:
                    best_score = score
                    best_move  = move
                beta = min(beta, score)

        print(f"  depth {depth} -> score {best_score:+d}  best: {move_to_string(best_move, board)}")

    return best_move