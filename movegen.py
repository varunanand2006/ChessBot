from constants import *

# ==========================================
# Move Encoding / Decoding
# ==========================================
# Bit layout:
#   bits  0-2:  r1
#   bits  3-5:  c1
#   bits  6-8:  r2
#   bits  9-11: c2
#   bits 12-14: flag

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
    """Fast decode — squares only, no flag."""
    return (move & 0x7), (move >> 3 & 0x7), (move >> 6 & 0x7), (move >> 9 & 0x7)

def move_to_string(move, board):
    r1, c1, r2, c2, flag = decode_move(move)
    letters = "abcdefgh"
    return (f"Moved {PIECE_SYMBOLS[board.squares[r1][c1]]} "
            f"from {letters[c1]}{8 - r1} to {letters[c2]}{8 - r2}")


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

    direction = -1 if white else 1
    start_row =  6 if white else 1
    promo_row =  0 if white else 7

    # Single push
    r1 = r + direction
    if 0 <= r1 < 8 and board.squares[r1][c] == EMPTY:
        if r1 == promo_row:
            for flag in PROMOTION_FLAGS:
                moves.append(encode_move(r, c, r1, c, flag))
        else:
            moves.append(encode_move(r, c, r1, c))

        # Double push
        r2 = r + 2 * direction
        if r == start_row and board.squares[r2][c] == EMPTY:
            moves.append(encode_move(r, c, r2, c))

    # Captures
    for dc in (-1, 1):
        rc, cc = r + direction, c + dc
        if not (0 <= rc < 8 and 0 <= cc < 8):
            continue

        target = board.squares[rc][cc]

        # Normal capture
        if target != EMPTY and (target > 0) != white:
            if rc == promo_row:
                for flag in PROMOTION_FLAGS:
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
# Sliding Pieces
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
# King (normal moves + castling)
# ==========================================

def generate_king_moves(board, r, c):
    moves = []
    white = board.squares[r][c] > 0

    # Normal one-square moves
    for dr, dc in KING_OFFSETS:
        rr, cc = r + dr, c + dc
        if not (0 <= rr < 8 and 0 <= cc < 8):
            continue
        target = board.squares[rr][cc]
        if target == EMPTY or (target > 0) != white:
            moves.append(encode_move(r, c, rr, cc))

    # Castling
    # King must not currently be in check
    if board.is_in_check(white):
        return moves

    if white:
        # Kingside: squares f1 and g1 must be empty and not attacked
        if (board.castle_wk
                and board.squares[7][5] == EMPTY
                and board.squares[7][6] == EMPTY
                and not board.is_square_attacked((7, 4), False)
                and not board.is_square_attacked((7, 5), False)
                and not board.is_square_attacked((7, 6), False)):
            moves.append(encode_move(7, 4, 7, 6, FLAG_CASTLE_KINGSIDE))

        # Queenside: squares b1, c1, d1 must be empty; king passes through c1, d1
        if (board.castle_wq
                and board.squares[7][3] == EMPTY
                and board.squares[7][2] == EMPTY
                and board.squares[7][1] == EMPTY
                and not board.is_square_attacked((7, 4), False)
                and not board.is_square_attacked((7, 3), False)
                and not board.is_square_attacked((7, 2), False)):
            moves.append(encode_move(7, 4, 7, 2, FLAG_CASTLE_QUEENSIDE))
    else:
        # Kingside: squares f8 and g8
        if (board.castle_bk
                and board.squares[0][5] == EMPTY
                and board.squares[0][6] == EMPTY
                and not board.is_square_attacked((0, 4), True)
                and not board.is_square_attacked((0, 5), True)
                and not board.is_square_attacked((0, 6), True)):
            moves.append(encode_move(0, 4, 0, 6, FLAG_CASTLE_KINGSIDE))

        # Queenside: squares b8, c8, d8
        if (board.castle_bq
                and board.squares[0][3] == EMPTY
                and board.squares[0][2] == EMPTY
                and board.squares[0][1] == EMPTY
                and not board.is_square_attacked((0, 4), True)
                and not board.is_square_attacked((0, 3), True)
                and not board.is_square_attacked((0, 2), True)):
            moves.append(encode_move(0, 4, 0, 2, FLAG_CASTLE_QUEENSIDE))

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