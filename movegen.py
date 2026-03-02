from constants import *

# ==========================================
# Move Encoding / Decoding
# ==========================================
# Move is a single integer:
#   bits  0-2:  r1   (0-7)
#   bits  3-5:  c1   (0-7)
#   bits  6-8:  r2   (0-7)
#   bits  9-11: c2   (0-7)
#   bits 12-14: flag (0-7)

def encode_move(r1, c1, r2, c2, flag=FLAG_NORMAL):
    return r1 | (c1 << 3) | (r2 << 6) | (c2 << 9) | (flag << 12)

def decode_move(move):
    r1   =  move        & 0x7
    c1   = (move >> 3)  & 0x7
    r2   = (move >> 6)  & 0x7
    c2   = (move >> 9)  & 0x7
    flag = (move >> 12) & 0x7
    return r1, c1, r2, c2, flag

def move_from_to(move):
    """Fast decode when you only need squares, not the flag."""
    return (move & 0x7), (move >> 3 & 0x7), (move >> 6 & 0x7), (move >> 9 & 0x7)

def move_to_string(move, board):
    r1, c1, r2, c2, flag = decode_move(move)
    letters = "abcdefgh"
    return f"Moved {piece_symbols[board.squares[r1][c1]]} from {letters[c1]}{8 - r1} to {letters[c2]}{8 - r2}"

# ==========================================
# Entry Point
# ==========================================

def generate_all_moves(board):
    moves = []
    white = board.white_to_move

    for r in range(8):
        for c in range(8):
            piece = board.squares[r][c]
            if piece == EMPTY:
                continue
            if white and piece < 0:
                continue
            if not white and piece > 0:
                continue

            moves.extend(generate_piece_moves(board, r, c))

    return moves


# ==========================================
# Piece Dispatcher
# ==========================================

def generate_piece_moves(board, r, c):
    piece = abs(board.squares[r][c])

    if piece == PAWN:
        return generate_pawn_moves(board, r, c)
    elif piece == KNIGHT:
        return generate_knight_moves(board, r, c)
    elif piece == BISHOP:
        return generate_sliding_moves(board, r, c, BISHOP_DIRS)
    elif piece == ROOK:
        return generate_sliding_moves(board, r, c, ROOK_DIRS)
    elif piece == QUEEN:
        return generate_sliding_moves(board, r, c, QUEEN_DIRS)
    elif piece == KING:
        return generate_king_moves(board, r, c)

    return []


# ==========================================
# Pawn
# ==========================================

def generate_pawn_moves(board, r, c):
    moves = []
    piece = board.squares[r][c]
    white = piece > 0

    direction  = -1 if white else 1
    start_row  =  6 if white else 1
    promo_row  =  0 if white else 7

    # --- Single push ---
    r1 = r + direction
    if 0 <= r1 < 8 and board.squares[r1][c] == EMPTY:

        if r1 == promo_row:
            # Emit all 4 promotion flags
            for flag in (FLAG_PROMOTE_QUEEN, FLAG_PROMOTE_ROOK,
                         FLAG_PROMOTE_BISHOP, FLAG_PROMOTE_KNIGHT):
                moves.append(encode_move(r, c, r1, c, flag))
        else:
            moves.append(encode_move(r, c, r1, c))

        # --- Double push ---
        r2 = r + 2 * direction
        if r == start_row and board.squares[r2][c] == EMPTY:
            moves.append(encode_move(r, c, r2, c))

    # --- Captures ---
    for dc in (-1, 1):
        rc, cc = r + direction, c + dc
        if not (0 <= rc < 8 and 0 <= cc < 8):
            continue

        target = board.squares[rc][cc]

        # Normal capture
        if target != EMPTY and (target > 0) != white:
            if rc == promo_row:
                for flag in (FLAG_PROMOTE_QUEEN, FLAG_PROMOTE_ROOK,
                             FLAG_PROMOTE_BISHOP, FLAG_PROMOTE_KNIGHT):
                    moves.append(encode_move(r, c, rc, cc, flag))
            else:
                moves.append(encode_move(r, c, rc, cc))

        # En passant
        if board.en_passant_sq is not None and (rc, cc) == board.en_passant_sq:
            moves.append(encode_move(r, c, rc, cc, FLAG_EN_PASSANT))

    return moves


# ==========================================
# Knight
# ==========================================

def generate_knight_moves(board, r, c):
    moves = []
    white = board.squares[r][c] > 0

    for dr, dc in KNIGHT_OFFSETS:
        rr, cc = r + dr, c + dc
        if not (0 <= rr < 8 and 0 <= cc < 8):
            continue
        target = board.squares[rr][cc]
        if target == EMPTY or (target > 0) != white:
            moves.append(encode_move(r, c, rr, cc))

    return moves


# ==========================================
# Sliding Pieces (Bishop, Rook, Queen)
# ==========================================

def generate_sliding_moves(board, r, c, directions):
    moves = []
    white = board.squares[r][c] > 0

    for dr, dc in directions:
        rr, cc = r + dr, c + dc
        while 0 <= rr < 8 and 0 <= cc < 8:
            target = board.squares[rr][cc]
            if target == EMPTY:
                moves.append(encode_move(r, c, rr, cc))
            else:
                if (target > 0) != white:
                    moves.append(encode_move(r, c, rr, cc))
                break
            rr += dr
            cc += dc

    return moves


# ==========================================
# King
# ==========================================

def generate_king_moves(board, r, c):
    moves = []
    white = board.squares[r][c] > 0

    for dr, dc in KING_OFFSETS:
        rr, cc = r + dr, c + dc
        if not (0 <= rr < 8 and 0 <= cc < 8):
            continue
        target = board.squares[rr][cc]
        if target == EMPTY or (target > 0) != white:
            moves.append(encode_move(r, c, rr, cc))

    return moves


# ==========================================
# Legal Move Filter
# ==========================================

def generate_legal_moves(board):
    pseudo_moves = generate_all_moves(board)
    legal = []

    for move in pseudo_moves:
        board.make_move(move)
        if not board.is_in_check(not board.white_to_move):
            legal.append(move)
        board.undo_move()

    return legal