from constants import *

# ==========================================
# Entry point
# ==========================================

def generate_all_moves(board):
    moves = []

    for r in range(8):
        for c in range(8):
            piece = board.squares[r][c]
            if piece == EMPTY:
                continue

            # Skip opponent pieces
            if board.white_to_move and piece < 0:
                continue
            if not board.white_to_move and piece > 0:
                continue

            moves.extend(generate_piece_moves(board, r, c))

    return moves


# ==========================================
# Piece dispatcher
# ==========================================

def generate_piece_moves(board, r, c):
    piece = board.squares[r][c]

    if abs(piece) == PAWN:
        return generate_pawn_moves(board, r, c)
    elif abs(piece) == KNIGHT:
        return generate_knight_moves(board, r, c)
    elif abs(piece) == BISHOP:
        return generate_sliding_moves(board, r, c,
                                      [(1,1),(1,-1),(-1,1),(-1,-1)])
    elif abs(piece) == ROOK:
        return generate_sliding_moves(board, r, c,
                                      [(1,0),(-1,0),(0,1),(0,-1)])
    elif abs(piece) == QUEEN:
        return generate_sliding_moves(board, r, c,
                                      [(1,0),(-1,0),(0,1),(0,-1),
                                       (1,1),(1,-1),(-1,1),(-1,-1)])
    elif abs(piece) == KING:
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
    start_row = 6 if white else 1

    # Single forward
    if 0 <= r + direction < 8:
        if board.squares[r + direction][c] == EMPTY:
            moves.append((r, c, r + direction, c))

            # Double forward
            if r == start_row and board.squares[r + 2*direction][c] == EMPTY:
                moves.append((r, c, r + 2*direction, c))

    # Captures
    for dc in (-1, 1):
        rr = r + direction
        cc = c + dc
        if 0 <= rr < 8 and 0 <= cc < 8:
            target = board.squares[rr][cc]
            if target != EMPTY and (target > 0) != white:
                moves.append((r, c, rr, cc))

    return moves


# ==========================================
# Knight
# ==========================================

def generate_knight_moves(board, r, c):
    moves = []
    piece = board.squares[r][c]
    white = piece > 0

    offsets = [
        (2,1),(2,-1),(-2,1),(-2,-1),
        (1,2),(1,-2),(-1,2),(-1,-2)
    ]

    for dr, dc in offsets:
        rr, cc = r+dr, c+dc
        if 0 <= rr < 8 and 0 <= cc < 8:
            target = board.squares[rr][cc]
            if target == EMPTY or (target > 0) != white:
                moves.append((r, c, rr, cc))

    return moves


# ==========================================
# Sliding pieces (bishop, rook, queen)
# ==========================================

def generate_sliding_moves(board, r, c, directions):
    moves = []
    piece = board.squares[r][c]
    white = piece > 0

    for dr, dc in directions:
        rr, cc = r+dr, c+dc
        while 0 <= rr < 8 and 0 <= cc < 8:
            target = board.squares[rr][cc]

            if target == EMPTY:
                moves.append((r, c, rr, cc))
            else:
                if (target > 0) != white:
                    moves.append((r, c, rr, cc))
                break

            rr += dr
            cc += dc

    return moves


# ==========================================
# King
# ==========================================

def generate_king_moves(board, r, c):
    moves = []
    piece = board.squares[r][c]
    white = piece > 0

    for dr in (-1,0,1):
        for dc in (-1,0,1):
            if dr == 0 and dc == 0:
                continue

            rr, cc = r+dr, c+dc
            if 0 <= rr < 8 and 0 <= cc < 8:
                target = board.squares[rr][cc]
                if target == EMPTY or (target > 0) != white:
                    moves.append((r, c, rr, cc))

    return moves


# ==========================================
# Legal Move Filter
# ==========================================

def generate_legal_moves(board):
    moves = generate_all_moves(board)
    legal = []

    for move in moves:
        board.make_move(move)

        # After move, side has flipped.
        # We check if the side that just moved is now in check.
        if not board.is_in_check(not board.white_to_move):
            legal.append(move)

        board.undo_move()

    return legal