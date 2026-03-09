import random

# ==========================================
# Piece Types
# ==========================================

EMPTY  = 0
PAWN   = 1
KNIGHT = 2
BISHOP = 3
ROOK   = 4
QUEEN  = 5
KING   = 6

WHITE = True
BLACK = False

# ==========================================
# Move Flags (bits 12-14 of encoded move)
# ==========================================

FLAG_NORMAL           = 0
FLAG_EN_PASSANT       = 1
FLAG_PROMOTE_QUEEN    = 2
FLAG_PROMOTE_ROOK     = 3
FLAG_PROMOTE_BISHOP   = 4
FLAG_PROMOTE_KNIGHT   = 5
FLAG_CASTLE_KINGSIDE  = 6
FLAG_CASTLE_QUEENSIDE = 7

PROMOTION_FLAGS = (FLAG_PROMOTE_QUEEN, FLAG_PROMOTE_ROOK,
                   FLAG_PROMOTE_BISHOP, FLAG_PROMOTE_KNIGHT)

PROMOTION_PIECES = {
    FLAG_PROMOTE_QUEEN:  QUEEN,
    FLAG_PROMOTE_ROOK:   ROOK,
    FLAG_PROMOTE_BISHOP: BISHOP,
    FLAG_PROMOTE_KNIGHT: KNIGHT,
}

# ==========================================
# Directions / Offsets
# ==========================================

BISHOP_DIRS = ((1,1),(1,-1),(-1,1),(-1,-1))
ROOK_DIRS   = ((1,0),(-1,0),(0,1),(0,-1))
QUEEN_DIRS  = ROOK_DIRS + BISHOP_DIRS

KNIGHT_OFFSETS = (
    (2,1),(2,-1),(-2,1),(-2,-1),
    (1,2),(1,-2),(-1,2),(-1,-2)
)

KING_OFFSETS = (
    (-1,-1),(-1,0),(-1,1),
    ( 0,-1),       ( 0,1),
    ( 1,-1),( 1,0),( 1,1)
)

# ==========================================
# Instructions
# ==========================================

instructions = """
================Chess================
Type in moves as startend
Example: e2e4 or d5e3
"""

# ==========================================
# Piece Symbols
# ==========================================

PIECE_SYMBOLS = {
     PAWN: "♟",  -PAWN: "♙",
   KNIGHT: "♞",-KNIGHT: "♘",
   BISHOP: "♝",-BISHOP: "♗",
     ROOK: "♜",   -ROOK: "♖",
    QUEEN: "♛",  -QUEEN: "♕",
     KING: "♚",   -KING: "♔",
    EMPTY: " "
}

PIECE_DICT = {
    PAWN:   "Pawn",
    KNIGHT: "Knight",
    BISHOP: "Bishop",
    ROOK:   "Rook",
    QUEEN:  "Queen",
    KING:   "King"
}

# ==========================================
# Transposition Table Flags
# ==========================================

TT_EXACT       = 0
TT_LOWER_BOUND = 1
TT_UPPER_BOUND = 2

# ==========================================
# Zobrist Hashing
# ==========================================

random.seed(794613)

ZOBRIST_TABLE = [
    [[random.getrandbits(64) for _ in range(8)] for _ in range(8)]
    for _ in range(13)
]

ZOBRIST_SIDE = random.getrandbits(64)


# ============================================================
# Table blending helper
# ============================================================

def _blend(hand, selfplay, lichess, w_hand, w_self, w_lich):
    """
    Return a new 8x8 table as a weighted sum of three source tables.
    Values are rounded to integers. Weights should sum to 1.0.
    """
    return [
        [round(w_hand * hand[r][c]
             + w_self * selfplay[r][c]
             + w_lich * lichess[r][c])
         for c in range(8)]
        for r in range(8)
    ]


# ============================================================
# Source Piece Values
# ============================================================

HANDCRAFT_PIECE_VALUES = {PAWN: 100, KNIGHT: 320, BISHOP: 330, ROOK: 500, QUEEN: 900, KING: 0}
SELFPLAY_PIECE_VALUES  = {PAWN:  87, KNIGHT: 254, BISHOP: 267, ROOK: 398, QUEEN: 718, KING: 0}
LICHESS_PIECE_VALUES   = {PAWN:  96, KNIGHT: 271, BISHOP: 295, ROOK: 440, QUEEN: 799, KING: 0}

# Self-play dropped ~20% uniformly (a known sigmoid scaling artifact), so excluded.
_PV_W_HAND = 0.50
_PV_W_SELF = 0.00
_PV_W_LICH = 0.50

PIECE_VALUES = {
    piece: round(_PV_W_HAND * HANDCRAFT_PIECE_VALUES[piece]
               + _PV_W_SELF * SELFPLAY_PIECE_VALUES[piece]
               + _PV_W_LICH * LICHESS_PIECE_VALUES[piece])
    for piece in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)
}


# ============================================================
# Source Pawn Tables
# ============================================================

HANDCRAFT_PAWN_TABLE = [
[  0,  0,  0,  0,  0,  0,  0,  0],
[ 50, 50, 50, 50, 50, 50, 50, 50],
[ 10, 10, 20, 30, 30, 20, 10, 10],
[  5,  5, 10, 25, 25, 10,  5,  5],
[  0,  0,  0, 20, 20,  0,  0,  0],
[  5, -5,-10,  0,  0,-10, -5,  5],
[  5, 10, 10,-20,-20, 10, 10,  5],
[  0,  0,  0,  0,  0,  0,  0,  0],
]

SELFPLAY_PAWN_TABLE = [
[   0,    0,    0,    0,    0,    0,    0,    0],
[ 140,  145,  118,  148,  115,  126,  155,  121],
[  -1,   45,   38,   37,   18,   37,   36,   23],
[  -3,    5,   -2,    6,   19,    7,    8,  -12],
[ -10,   -6,   -5,    1,    1,    1,   -1,  -20],
[ -21,   -5,   -5,  -14,   -4,   -8,    3,  -18],
[ -31,    0,  -16,  -18,  -25,    3,   11,  -25],
[   0,    0,    0,    0,    0,    0,    0,    0],
]

LICHESS_PAWN_TABLE = [
[   0,    0,    0,    0,    0,    0,    0,    0],
[ 160,  130,  124,  123,  123,   77,  113,   50],
[  52,   45,   50,   36,   38,   36,   44,   33],
[   9,   11,   -1,   -7,   14,   10,   23,   14],
[ -14,    0,   -8,    7,   13,   -4,    7,  -17],
[  -8,  -12,  -17,   -4,   10,   -6,   18,  -12],
[  -9,   -3,  -24,  -19,   -1,   10,   21,   -8],
[   0,    0,    0,    0,    0,    0,    0,    0],
]

# Both tuned tables agree well on pawn structure
_PAWN_W_HAND, _PAWN_W_SELF, _PAWN_W_LICH = 0.25, 0.375, 0.375

PAWN_TABLE = _blend(HANDCRAFT_PAWN_TABLE, SELFPLAY_PAWN_TABLE, LICHESS_PAWN_TABLE,
                    _PAWN_W_HAND, _PAWN_W_SELF, _PAWN_W_LICH)


# ============================================================
# Source Knight Tables
# ============================================================

HANDCRAFT_KNIGHT_TABLE = [
[-50,-40,-30,-30,-30,-30,-40,-50],
[-40,-20,  0,  5,  5,  0,-20,-40],
[-30,  5, 10, 15, 15, 10,  5,-30],
[-30,  0, 15, 20, 20, 15,  0,-30],
[-30,  5, 15, 20, 20, 15,  5,-30],
[-30,  0, 10, 15, 15, 10,  0,-30],
[-40,-20,  0,  0,  0,  0,-20,-40],
[-50,-40,-30,-30,-30,-30,-40,-50],
]

SELFPLAY_KNIGHT_TABLE = [
[-108, -197,  -99,  -48,  -57, -158,  -78, -130],
[-112,  -49,    2,  -56,   15,  -16,  -46,  -14],
[  -6,  -11,    8,   13,   18,  -17,    9,  -26],
[ -31,    3,    5,    6,    3,   21,   12,   -3],
[ -31,    5,    4,    3,    1,    7,   -5,   -7],
[ -19,  -27,    1,   -6,    5,   -1,    7,  -19],
[ -48,  -43,    1,  -19,  -16,  -11,  -49,  -27],
[ -66,  -43,  -17,  -38,  -30,  -12,  -43,  -22],
]

LICHESS_KNIGHT_TABLE = [
[-143,  -63, -104,  -56,  -27, -125, -119, -186],
[ -43,  -15,   32,  -43,   16,    2,  -45,    0],
[  -6,    4,  -28,   47,   39,   40,   -7,   26],
[   0,    7,   30,   31,   19,   48,    7,   22],
[ -16,  -10,   12,    1,    7,    7,   -5,  -18],
[ -56,  -17,   -5,   -7,   -5,   -2,  -11,  -53],
[ -92,  -65,  -22,  -18,  -23,  -20,  -29,  -56],
[ -74,  -44,  -56,  -49,  -49,  -41,  -38,  -73],
]

# Hand-crafted is cleanest; both tuned versions have large corner noise
_KNIGHT_W_HAND, _KNIGHT_W_SELF, _KNIGHT_W_LICH = 0.55, 0.15, 0.30

KNIGHT_TABLE = _blend(HANDCRAFT_KNIGHT_TABLE, SELFPLAY_KNIGHT_TABLE, LICHESS_KNIGHT_TABLE,
                      _KNIGHT_W_HAND, _KNIGHT_W_SELF, _KNIGHT_W_LICH)


# ============================================================
# Source Bishop Tables
# ============================================================

HANDCRAFT_BISHOP_TABLE = [
[-20,-10,-10,-10,-10,-10,-10,-20],
[-10,  5,  0,  0,  0,  0,  5,-10],
[-10, 10, 10, 10, 10, 10, 10,-10],
[-10,  0, 10, 10, 10, 10,  0,-10],
[-10,  5,  5, 10, 10,  5,  5,-10],
[-10,  0,  5, 10, 10,  5,  0,-10],
[-10,  0,  0,  0,  0,  0,  0,-10],
[-20,-10,-10,-10,-10,-10,-10,-20],
]

SELFPLAY_BISHOP_TABLE = [
[ -11,  -27,  -31,    2,  -13,  -76,  -89,  -22],
[ -10,   -3,  -20,  -38,  -48,  -22,  -26,  -14],
[ -81,    2,  -33,   -7,    3,  -19,   20,  -14],
[ -16,   -5,  -22,   10,   21,    2,   21,    4],
[ -24,  -37,    4,    3,    0,    7,  -21,   15],
[   2,  -39,    4,   15,    7,    1,   11,    9],
[  12,   17,  -14,    0,    8,   -6,    9,  -22],
[ -14,  -22,   -9,  -26,  -19,  -23,  -12,   -3],
]

LICHESS_BISHOP_TABLE = [
[ -88,  -41,  -56,  -29,  -22, -134,  -24,  -35],
[ -32,   -3,   18,  -19,  -85,   -4,  -80,  -12],
[  -4,   12,  -36,   -1,    5, -103,  -41,   18],
[   3,   12,    5,   18,   18,    8,    5,  -12],
[  -8,    3,    8,   15,    7,   10,    2,   -9],
[   8,   -3,    7,   14,   11,    7,    2,    6],
[  -4,    9,    2,  -14,    4,   -3,   18,   -3],
[ -19,  -19,  -11,  -42,  -37,  -16,  -30,  -14],
]

# Lichess has -134/-103 outliers; self-play is cleaner here
_BISHOP_W_HAND, _BISHOP_W_SELF, _BISHOP_W_LICH = 0.50, 0.35, 0.15

BISHOP_TABLE = _blend(HANDCRAFT_BISHOP_TABLE, SELFPLAY_BISHOP_TABLE, LICHESS_BISHOP_TABLE,
                      _BISHOP_W_HAND, _BISHOP_W_SELF, _BISHOP_W_LICH)


# ============================================================
# Source Rook Tables
# ============================================================

HANDCRAFT_ROOK_TABLE = [
[  0,  0,  0,  5,  5,  0,  0,  0],
[ -5,  0,  0,  0,  0,  0,  0, -5],
[ -5,  0,  0,  0,  0,  0,  0, -5],
[ -5,  0,  0,  0,  0,  0,  0, -5],
[ -5,  0,  0,  0,  0,  0,  0, -5],
[ -5,  0,  0,  0,  0,  0,  0, -5],
[  5, 10, 10, 10, 10, 10, 10,  5],
[  0,  0,  0,  0,  0,  0,  0,  0],
]

SELFPLAY_ROOK_TABLE = [
[ -20,  -19,  -31,   -4,  -27,  -37,   -6,   -7],
[   7,    3,   12,    8,  -16,  -21,  -16,   13],
[ -24,  -14,   -9,  -14,  -13,  -25,  -28,  -19],
[ -10,  -16,  -21,   -5,  -35,  -19,   -3,  -20],
[ -26,  -39,  -22,  -19,  -22,   -7,  -41,  -39],
[ -30,  -22,  -44,  -13,  -31,  -39,   -6,  -32],
[ -19,  -29,  -23,  -23,  -27,  -19,  -31,  -60],
[ -34,  -25,  -15,   -3,    1,  -15,  -34,  -43],
]

LICHESS_ROOK_TABLE = [
[  15,    6,    9,  -24,    5,   26,   44,   -9],
[  28,   40,   56,   44,   38,   41,   39,   48],
[  26,   25,   36,   21,   13,   15,   47,   28],
[   9,   20,   25,   13,    7,   13,   15,  -15],
[ -15,   -6,   -8,    2,  -16,  -24,  -11,  -16],
[ -41,  -14,  -16,  -14,  -25,  -21,   -6,  -16],
[ -37,  -29,  -23,  -24,  -16,  -15,  -12,  -23],
[ -23,  -19,   -4,   -3,   -4,   -7,    6,  -25],
]

# Self-play produced almost entirely negative values (overfitting artifact).
# Lichess correctly shows 7th-rank and open-file bonuses.
_ROOK_W_HAND, _ROOK_W_SELF, _ROOK_W_LICH = 0.40, 0.10, 0.50

ROOK_TABLE = _blend(HANDCRAFT_ROOK_TABLE, SELFPLAY_ROOK_TABLE, LICHESS_ROOK_TABLE,
                    _ROOK_W_HAND, _ROOK_W_SELF, _ROOK_W_LICH)


# ============================================================
# Source Queen Tables
# ============================================================

HANDCRAFT_QUEEN_TABLE = [
[-20,-10,-10, -5, -5,-10,-10,-20],
[-10,  0,  0,  0,  0,  0,  0,-10],
[-10,  5,  5,  5,  5,  5,  0,-10],
[ -5,  0,  5,  5,  5,  5,  0, -5],
[  0,  0,  5,  5,  5,  5,  0, -5],
[-10,  5,  5,  5,  5,  5,  0,-10],
[-10,  0,  0,  0,  0,  0,  0,-10],
[-20,-10,-10, -5, -5,-10,-10,-20],
]

SELFPLAY_QUEEN_TABLE = [
[ -45,   -2,   -1,   17,  -12,  -12,  -44,  -55],
[ -23,    0,   20,   -4,   15,   46,   -8,  -25],
[ -29,    5,   -4,    3,   29,   36,   -2,  -51],
[ -35,    3,  -17,   -7,   12,    3,  -20,  -12],
[ -16,  -47,  -19,   -5,  -11,  -16,  -17,   -7],
[ -53,  -15,  -28,  -17,  -20,   -7,  -29,  -43],
[  -8,  -31,   -9,  -22,  -11,  -35,  -35,   16],
[ -31,  -47,  -17,  -28,  -34,  -60,  -77,  -45],
]

LICHESS_QUEEN_TABLE = [
[ -24,  -24,   40,   -4,   52,   11,   -2,    7],
[ -23,  -14,   18,   23,   23,   58,   42,   57],
[ -22,  -20,   16,   21,   18,   67,   40,   35],
[ -13,   -4,    2,   11,   22,   18,    7,   24],
[  -4,   -9,   -8,   10,    2,   -6,   -6,    1],
[ -24,   -7,  -12,  -14,  -19,   -1,    1,   -9],
[ -36,   -8,  -15,  -14,  -13,  -22,  -25,  -35],
[ -36,  -35,  -40,  -15,  -28,  -43,  -23,  -48],
]

# Both tuned versions have large outliers; hand-crafted anchors the blend
_QUEEN_W_HAND, _QUEEN_W_SELF, _QUEEN_W_LICH = 0.60, 0.15, 0.25

QUEEN_TABLE = _blend(HANDCRAFT_QUEEN_TABLE, SELFPLAY_QUEEN_TABLE, LICHESS_QUEEN_TABLE,
                     _QUEEN_W_HAND, _QUEEN_W_SELF, _QUEEN_W_LICH)


# ============================================================
# Source King Tables (middlegame)
# ============================================================

HANDCRAFT_KING_TABLE = [
[-30,-40,-40,-50,-50,-40,-40,-30],
[-30,-40,-40,-50,-50,-40,-40,-30],
[-30,-40,-40,-50,-50,-40,-40,-30],
[-30,-40,-40,-50,-50,-40,-40,-30],
[-20,-30,-30,-40,-40,-30,-30,-20],
[-10,-20,-20,-20,-20,-20,-20,-10],
[ 20, 20,  0,  0,  0,  0, 20, 20],
[ 20, 30, 10,  0,  0, 10, 30, 20],
]

SELFPLAY_KING_TABLE = [
[ -18, -125, -115,    3,   80,   34,  -18, -150],
[ -25,  -78,  -28,   12,    8,   22,   -1,  -26],
[ -47,  -36,   27,   25,   24,  -18,  -28,  -47],
[  -3,    0,   16,   55,   24,   10,  -12,  -44],
[ -40,   15,   16,  -15,    7,  -19,   -6,  -51],
[ -24,   -4,    0,  -10,  -11,   11,   -1,  -24],
[  18,   20,    0,  -17,  -20,  -11,    9,   -3],
[ -23,    4,    4,  -34,  -10,  -19,   13,   24],
]

LICHESS_KING_TABLE = [
[ 311,   41, -112,  164,   15,  169,   19,  -56],
[ -10,   41,   50,   45,   27,  -35,   63,  -72],
[  29,   39,   77,   72,   99,   97,   55,   17],
[   4,   19,   79,   35,   34,   24,   37,  -34],
[ -15,    6,    4,   32,   15,    1,  -14,  -32],
[ -41,   -1,  -13,   -1,   -5,  -14,   -4,  -49],
[ -20,  -11,  -11,  -17,  -23,  -13,   -2,  -20],
[ -30,   15,    3,  -62,  -14,  -42,    5,  -11],
]

# Both tuned versions are broken (e.g. +311, -150).
# King safety requires pawn shelter and open-file context a PST cannot encode.
_KING_W_HAND, _KING_W_SELF, _KING_W_LICH = 1.00, 0.00, 0.00

KING_TABLE = _blend(HANDCRAFT_KING_TABLE, SELFPLAY_KING_TABLE, LICHESS_KING_TABLE,
                    _KING_W_HAND, _KING_W_SELF, _KING_W_LICH)


# ============================================================
# Source King Endgame Tables
# ============================================================

HANDCRAFT_KING_END_TABLE = [
[-50,-40,-30,-20,-20,-30,-40,-50],
[-30,-20,-10,  0,  0,-10,-20,-30],
[-30,-10, 20, 30, 30, 20,-10,-30],
[-30,-10, 30, 40, 40, 30,-10,-30],
[-30,-10, 30, 40, 40, 30,-10,-30],
[-30,-10, 20, 30, 30, 20,-10,-30],
[-30,-30,  0,  0,  0,  0,-30,-30],
[-50,-30,-30,-30,-30,-30,-30,-50],
]

SELFPLAY_KING_END_TABLE = [
[  60,   35,   44,  -34,   13,   12,   19,  -28],
[  54,   20,   45,   26,    2,   36,   -4,  -19],
[  -3,   46,   20,   10,    4,   20,    8,   34],
[  14,    8,   23,    2,    4,    2,   12,  -13],
[ -92,   -5,  -23,   13,   -4,    4,   -1,  -44],
[  -7,  -35,  -19,    5,    3,  -17,   -4,   -1],
[ -69,  -31,  -18,  -27,  -29,  -31,  -34,   -6],
[  12,  -48,  -24,  -49,  -39,  -22,  -26,  -10],
]

LICHESS_KING_END_TABLE = [
[ -57,  -50,  -79,  -18,  -92,   14,  -63,  -72],
[ -72,   -3,   26,   72,  109,   15,   87,   39],
[ -19,   48,   35,   33,   63,   40,   89,   -2],
[   9,   30,   20,   26,   27,   -1,   40,   42],
[  16,  -11,  -11,   -1,    7,   -4,  -26,  -30],
[ -37,  -50,  -29,  -14,  -35,  -26,  -23,  -46],
[ -23,  -36,  -50,  -40,  -30,  -16,  -22,  -13],
[-107,  -29,  -39,  -30,  -47,  -55,   -4,  -25],
]

# Both tuned versions are noisy. Use hand-crafted only.
_KING_END_W_HAND, _KING_END_W_SELF, _KING_END_W_LICH = 1.00, 0.00, 0.00

KING_END_TABLE = _blend(HANDCRAFT_KING_END_TABLE, SELFPLAY_KING_END_TABLE, LICHESS_KING_END_TABLE,
                        _KING_END_W_HAND, _KING_END_W_SELF, _KING_END_W_LICH)
