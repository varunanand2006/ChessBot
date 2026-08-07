"""
engine_main.py — run with PyPy for maximum search speed.

Protocol (stdin → stdout, one line each):
  newgame           → ok
  move e2e4         → ok | illegal
  go <depth>        → bestmove e7e5  (engine also makes the move internally)
  legal             → e2e4 d2d4 ...  | none
  board             → 64 comma-separated piece chars (row-major, '.' = empty)
  status            → ongoing <side> | check <side> | checkmate <winner>
                      | stalemate | draw repetition
  quit              → (exit)

All debug/search output is redirected to stderr so it never
pollutes the protocol stream.
"""

import sys
import builtins

# ── Redirect all prints to stderr BEFORE importing search ──
_proto_print = builtins.print

def _stderr(*args, **kwargs):
    kwargs['file'] = sys.stderr
    _proto_print(*args, **kwargs)

builtins.print = _stderr

# ── Now safe to import modules whose prints go to stderr ──
from board import Board
from movegen import generate_legal_moves, encode_move, decode_move
from search import find_best_move
from constants import *

# ── Protocol helpers ──

def send(msg: str):
    """Write one line to stdout (the GUI reads this)."""
    _proto_print(msg, flush=True)


def encode_uci(move: int) -> str:
    r1, c1, r2, c2, flag = decode_move(move)
    letters = "abcdefgh"
    s = f"{letters[c1]}{8 - r1}{letters[c2]}{8 - r2}"
    promo = {
        FLAG_PROMOTE_QUEEN:  'q',
        FLAG_PROMOTE_ROOK:   'r',
        FLAG_PROMOTE_BISHOP: 'b',
        FLAG_PROMOTE_KNIGHT: 'n',
    }
    if flag in promo:
        s += promo[flag]
    return s


def parse_uci(s: str, board: Board) -> int:
    c1 = ord(s[0]) - ord('a')
    r1 = 8 - int(s[1])
    c2 = ord(s[2]) - ord('a')
    r2 = 8 - int(s[3])
    flag = FLAG_NORMAL

    if len(s) == 5:
        flag = {
            'q': FLAG_PROMOTE_QUEEN,
            'r': FLAG_PROMOTE_ROOK,
            'b': FLAG_PROMOTE_BISHOP,
            'n': FLAG_PROMOTE_KNIGHT,
        }[s[4]]
    elif abs(board.squares[r1][c1]) == PAWN and board.en_passant_sq == (r2, c2):
        flag = FLAG_EN_PASSANT
    elif abs(board.squares[r1][c1]) == KING and abs(c2 - c1) == 2:
        flag = FLAG_CASTLE_KINGSIDE if c2 > c1 else FLAG_CASTLE_QUEENSIDE

    return encode_move(r1, c1, r2, c2, flag)


def board_to_str(board: Board) -> str:
    chars = {
        PAWN:   'P', KNIGHT: 'N', BISHOP: 'B', ROOK: 'R', QUEEN: 'Q', KING: 'K',
        -PAWN:  'p', -KNIGHT:'n', -BISHOP:'b', -ROOK: 'r', -QUEEN:'q', -KING: 'k',
        EMPTY:  '.',
    }
    return ','.join(chars[board.squares[r][c]] for r in range(8) for c in range(8))


# ── Main loop ──

board = Board()
board.setup_starting_position()

for raw in sys.stdin:
    line = raw.strip()
    if not line:
        continue

    # ── newgame ──────────────────────────────────────────────
    if line == 'newgame':
        board = Board()
        board.setup_starting_position()
        send('ok')

    # ── move <uci> ───────────────────────────────────────────
    elif line.startswith('move '):
        uci = line[5:]
        try:
            move  = parse_uci(uci, board)
            legal = generate_legal_moves(board)
            if move in legal:
                board.make_move(move)
                send('ok')
            else:
                send('illegal')
        except Exception as e:
            send(f'error {e}')

    # ── go <depth> ───────────────────────────────────────────
    elif line.startswith('go '):
        depth = int(line.split()[1])
        best  = find_best_move(board, depth)
        if best is None:
            send('none')
        else:
            board.make_move(best)
            send(f'bestmove {encode_uci(best)}')

    # ── legal ────────────────────────────────────────────────
    elif line == 'legal':
        legal = generate_legal_moves(board)
        send(' '.join(encode_uci(m) for m in legal) if legal else 'none')

    # ── board ────────────────────────────────────────────────
    elif line == 'board':
        send(board_to_str(board))

    # ── status ───────────────────────────────────────────────
    elif line == 'status':
        rep   = board.position_history.get(board.hash, 0)
        legal = generate_legal_moves(board)
        side  = 'white' if board.white_to_move else 'black'

        if rep >= 3:
            send('draw repetition')
        elif not legal:
            if board.is_in_check(board.white_to_move):
                winner = 'black' if board.white_to_move else 'white'
                send(f'checkmate {winner}')
            else:
                send('stalemate')
        elif board.is_in_check(board.white_to_move):
            send(f'check {side}')
        else:
            send(f'ongoing {side}')

    # ── quit ─────────────────────────────────────────────────
    elif line == 'quit':
        break