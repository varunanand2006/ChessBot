from board import *

# Knight jumps
KNIGHT_DIRS = [
    (-2,-1), (-2,1),
    (-1,-2), (-1,2),
    (1,-2),  (1,2),
    (2,-1),  (2,1)
]

# King directions
KING_DIRS = [
    (-1,-1), (-1,0), (-1,1),
    (0,-1),          (0,1),
    (1,-1),  (1,0),  (1,1)
]

# Sliding directions
ROOK_DIRS = [(1,0),(-1,0),(0,1),(0,-1)]
BISHOP_DIRS = [(1,1),(1,-1),(-1,1),(-1,-1)]


def get_legal_moves(board, color):
    moves = []

    for r in range(8):
        for c in range(8):
            piece = board[r][c]

            if piece == 0:
                continue

            if piece * color <= 0:
                continue  # wrong color

            moves.extend(generate_piece_moves(board, r, c))

    legal = []
    for move in moves:
        new_board = make_move(board, move)
        if not is_in_check(new_board, color):
            legal.append(move)

    return legal

def generate_piece_moves(board, r, c):
    piece = board[r][c]
    ptype = abs(piece)

    if ptype == PAWN:
        return generate_pawn_moves(board, r, c)
    elif ptype == KNIGHT:
        return generate_knight_moves(board, r, c)
    elif ptype == BISHOP:
        return generate_sliding_moves(board, r, c, BISHOP_DIRS)
    elif ptype == ROOK:
        return generate_sliding_moves(board, r, c, ROOK_DIRS)
    elif ptype == QUEEN:
        return generate_sliding_moves(board, r, c, ROOK_DIRS + BISHOP_DIRS)
    elif ptype == KING:
        return generate_king_moves(board, r, c)

def generate_pawn_moves(board, r, c):
    moves = []
    piece = board[r][c]

    direction = -1 if piece > 0 else 1   # white moves up, black down
    start_row = 6 if piece > 0 else 1

    # One square forward
    nr = r + direction
    if 0 <= nr < 8:
        if board[nr][c] == 0:
            moves.append((r,c,nr,c))

            # Two squares forward
            if r == start_row:
                nr2 = r + 2*direction
                if board[nr2][c] == 0:
                    moves.append((r,c,nr2,c))

    # Captures
    for dc in [-1, 1]:
        nc = c + dc
        if 0 <= nr < 8 and 0 <= nc < 8:
            target = board[nr][nc]
            if target != 0 and target * piece < 0:
                moves.append((r,c,nr,nc))

    return moves

def generate_knight_moves(board, r, c):
    moves = []
    piece = board[r][c]

    for dr, dc in KNIGHT_DIRS:
        nr, nc = r+dr, c+dc

        if 0 <= nr < 8 and 0 <= nc < 8:
            target = board[nr][nc]

            if target == 0 or (target * piece < 0):
                moves.append((r,c,nr,nc))

    return moves

def generate_bishop_moves(board, r, c):
    return generate_sliding_moves(board, r, c, BISHOP_DIRS)

def generate_rook_moves(board, r, c):
    return generate_sliding_moves(board, r, c, ROOK_DIRS)

def generate_queen_moves(board, r, c):
    return generate_sliding_moves(board, r, c, ROOK_DIRS + BISHOP_DIRS)

def generate_sliding_moves(board, r, c, directions):
    moves = []
    piece = board[r][c]

    for dr, dc in directions:
        nr, nc = r+dr, c+dc

        while 0 <= nr < 8 and 0 <= nc < 8:
            target = board[nr][nc]

            if target == 0:
                moves.append((r,c,nr,nc))
            else:
                if target * piece < 0:
                    moves.append((r,c,nr,nc))
                break  # stop sliding

            nr += dr
            nc += dc

    return moves

def generate_sliding_moves(board, r, c, directions):
    moves = []
    piece = board[r][c]

    for dr, dc in directions:
        nr, nc = r+dr, c+dc

        while 0 <= nr < 8 and 0 <= nc < 8:
            target = board[nr][nc]

            if target == 0:
                moves.append((r,c,nr,nc))
            else:
                if target * piece < 0:
                    moves.append((r,c,nr,nc))
                break  # stop sliding

            nr += dr
            nc += dc

    return moves

def generate_king_moves(board, r, c):
    moves = []
    piece = board[r][c]
    color = 1 if piece > 0 else -1
    opponent = -color

    for dr, dc in KING_DIRS:
        nr, nc = r + dr, c + dc

        if 0 <= nr < 8 and 0 <= nc < 8:
            target = board[nr][nc]

            # must be empty or enemy
            if target == 0 or target * piece < 0:

                # temporarily move king
                original = board[nr][nc]
                board[r][c] = 0
                board[nr][nc] = piece

                safe = not is_square_attacked(board, nr, nc, opponent)

                # undo move
                board[r][c] = piece
                board[nr][nc] = original

                if safe:
                    moves.append((r,c,nr,nc))

    return moves

def generate_piece_moves(board, r, c):
    piece = board[r][c]
    ptype = abs(piece)

    if ptype == 1:
        return generate_pawn_moves(board, r, c)
    elif ptype == 2:
        return generate_knight_moves(board, r, c)
    elif ptype == 3:
        return generate_bishop_moves(board, r, c)
    elif ptype == 4:
        return generate_rook_moves(board, r, c)
    elif ptype == 5:
        return generate_queen_moves(board, r, c)
    elif ptype == 6:
        return generate_king_moves(board, r, c)

    return []

def is_square_attacked(board, r, c, attacker_color):

    knight = attacker_color * 2
    bishop = attacker_color * 3
    rook   = attacker_color * 4
    queen  = attacker_color * 5
    king   = attacker_color * 6
    pawn   = attacker_color * 1

    # Pawn attacks
    pawn_dir = -1 if attacker_color == 1 else 1
    pr = r + pawn_dir

    if 0 <= pr < 8:
        if c > 0 and board[pr][c-1] == pawn:
            return True
        if c < 7 and board[pr][c+1] == pawn:
            return True

    # Knight attacks
    for dr, dc in KNIGHT_DIRS:
        nr, nc = r+dr, c+dc
        if 0 <= nr < 8 and 0 <= nc < 8:
            if board[nr][nc] == knight:
                return True

    # King attacks
    for dr, dc in KING_DIRS:
        nr, nc = r+dr, c+dc
        if 0 <= nr < 8 and 0 <= nc < 8:
            if board[nr][nc] == king:
                return True

    # Bishop/Queen diagonals
    for dr, dc in BISHOP_DIRS:
        nr, nc = r+dr, c+dc
        while 0 <= nr < 8 and 0 <= nc < 8:
            piece = board[nr][nc]
            if piece != 0:
                if piece == bishop or piece == queen:
                    return True
                break
            nr += dr
            nc += dc

    # Rook/Queen straight
    for dr, dc in ROOK_DIRS:
        nr, nc = r+dr, c+dc
        while 0 <= nr < 8 and 0 <= nc < 8:
            piece = board[nr][nc]
            if piece != 0:
                if piece == rook or piece == queen:
                    return True
                break
            nr += dr
            nc += dc

    return False