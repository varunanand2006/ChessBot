"""
gui.py — run with CPython + pygame.

Launches engine_main.py as a PyPy subprocess and communicates
over stdin/stdout.  Human plays White, engine plays Black.

Configuration:
  PYPY_EXE      — path/name of your PyPy executable
  ENGINE_SCRIPT — path to engine_main.py
  DEPTH         — engine search depth
"""

import pygame
import subprocess
import threading
import sys
import os

# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════

PYPY_EXE = r"C:\pypy3.11-v7.3.20-win64\pypy3.11.exe"
ENGINE_SCRIPT = os.path.join(os.path.dirname(__file__), "engine_main.py")
DEPTH         = 5

# ══════════════════════════════════════════════════════════════
# Layout
# ══════════════════════════════════════════════════════════════

SQ        = 72
BOARD_X   = 48
BOARD_Y   = 48
BOARD_PX  = SQ * 8          # 576
PANEL_X   = BOARD_X + BOARD_PX + 20
PANEL_W   = 264
WIN_W     = PANEL_X + PANEL_W + 16
WIN_H     = BOARD_Y + BOARD_PX + BOARD_Y   # symmetric vertical padding

# ══════════════════════════════════════════════════════════════
# Colour palette  (dark-luxury theme)
# ══════════════════════════════════════════════════════════════

C_BG         = ( 14,  14,  22)   # window background
C_LIGHT      = (237, 214, 176)   # light squares
C_DARK       = (179, 133,  94)   # dark squares
C_PANEL      = ( 22,  22,  36)   # right panel background
C_BORDER     = ( 48,  48,  68)   # panel border line
C_TEXT       = (210, 200, 180)   # primary text
C_ACCENT     = (196, 154, 100)   # headings / highlights
C_DIM        = (120, 112,  98)   # secondary text
C_SEL_L      = (220, 210,  55)   # selection tint on light sq
C_SEL_D      = (190, 172,  20)   # selection tint on dark sq
C_DOT        = ( 75, 105,  50)   # legal-move dot fill
C_RING       = (175,  48,  48)   # capture highlight ring
C_CHECK      = (210,  38,  38)   # king-in-check square
C_WPC        = (244, 239, 222)   # white piece fill
C_BPC        = ( 38,  30,  22)   # black piece fill
C_WPC_BDR    = ( 88,  68,  48)   # white piece border
C_BPC_BDR    = (200, 184, 160)   # black piece border

# ══════════════════════════════════════════════════════════════
# Unicode piece symbols
# ══════════════════════════════════════════════════════════════

PIECE_UNICODE = {
    'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟',
}

# ══════════════════════════════════════════════════════════════
# Engine I/O
# ══════════════════════════════════════════════════════════════

_engine_lock = threading.Lock()
_engine_proc = None

ENGINE_DONE = pygame.USEREVENT + 1   # posted when engine finishes its move


def start_engine():
    global _engine_proc
    _engine_proc = subprocess.Popen(
        [PYPY_EXE, ENGINE_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,   # suppress engine's debug output
        text=True,
    )


# ══════════════════════════════════════════════════════════════
# Piece image loading
# ══════════════════════════════════════════════════════════════

# Maps piece char → pygame.Surface (loaded in main after pygame.init)
PIECE_IMAGES: dict = {}

PIECE_PNG_NAMES = {
    'K': 'wK', 'Q': 'wQ', 'R': 'wR', 'B': 'wB', 'N': 'wN', 'P': 'wP',
    'k': 'bK', 'q': 'bQ', 'r': 'bR', 'b': 'bB', 'n': 'bN', 'p': 'bP',
}

def load_piece_images():
    pieces_dir = os.path.join(os.path.dirname(__file__), "pieces")
    if not os.path.isdir(pieces_dir):
        raise FileNotFoundError(
            f"'pieces/' folder not found at {pieces_dir}\n"
            "Run download_pieces.py first to fetch the piece images."
        )
    for char, name in PIECE_PNG_NAMES.items():
        path = os.path.join(pieces_dir, f"{name}.png")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing piece image: {path}\nRun download_pieces.py first.")
        img = pygame.image.load(path).convert_alpha()
        PIECE_IMAGES[char] = pygame.transform.smoothscale(img, (SQ - 8, SQ - 8))


def _send(cmd: str) -> str:
    """Send one command and return the single-line response (blocking)."""
    with _engine_lock:
        _engine_proc.stdin.write(cmd + "\n")
        _engine_proc.stdin.flush()
        return _engine_proc.stdout.readline().strip()


def engine_newgame():
    _send("newgame")


def engine_move(uci: str) -> bool:
    return _send(f"move {uci}") == "ok"


def engine_board() -> list:
    """Return the board as a list[64] of piece chars ('.' = empty)."""
    resp = _send("board")
    return resp.split(",")


def engine_legal() -> list:
    """Return list of legal UCI strings for the side to move."""
    resp = _send("legal")
    if resp == "none" or not resp:
        return []
    return resp.split()


def engine_status() -> str:
    return _send("status")


def engine_go_async(depth: int):
    """Ask engine to find+make its move in a background thread.
    Posts ENGINE_DONE with the bestmove string when finished."""
    def worker():
        resp = _send(f"go {depth}")
        # resp = "bestmove e7e5"  or  "none"
        pygame.event.post(pygame.event.Event(ENGINE_DONE, {"resp": resp}))

    t = threading.Thread(target=worker, daemon=True)
    t.start()


def engine_quit():
    try:
        _engine_proc.stdin.write("quit\n")
        _engine_proc.stdin.flush()
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════
# Coordinate helpers
# ══════════════════════════════════════════════════════════════

def sq_to_px(row: int, col: int):
    """Top-left pixel of a board square."""
    return BOARD_X + col * SQ, BOARD_Y + row * SQ


def px_to_sq(px: int, py: int):
    """Board (row, col) from pixel, or None if outside board."""
    col = (px - BOARD_X) // SQ
    row = (py - BOARD_Y) // SQ
    if 0 <= row < 8 and 0 <= col < 8:
        return row, col
    return None


def idx(row: int, col: int) -> int:
    return row * 8 + col


# ══════════════════════════════════════════════════════════════
# Drawing helpers
# ══════════════════════════════════════════════════════════════

def draw_board(surf, selected, legal_sqs, check_sq):
    """Draw the 8×8 chessboard with highlights."""
    for r in range(8):
        for c in range(8):
            light = (r + c) % 2 == 0
            base  = C_LIGHT if light else C_DARK
            x, y  = sq_to_px(r, c)

            # Check highlight
            if (r, c) == check_sq:
                color = C_CHECK
            # Selection highlight
            elif (r, c) == selected:
                color = C_SEL_L if light else C_SEL_D
            else:
                color = base

            pygame.draw.rect(surf, color, (x, y, SQ, SQ))

            # Capture ring
            if (r, c) in legal_sqs and (r, c) != selected:
                # check if there's a piece on this square (capture target)
                pass   # handled after piece draw

    # Coordinates along edges
    coord_font = pygame.font.SysFont("Georgia", 13, italic=True)
    files = "abcdefgh"
    for i in range(8):
        # file letters (bottom edge)
        txt = coord_font.render(files[i], True, C_DIM)
        x, _ = sq_to_px(7, i)
        surf.blit(txt, (x + SQ - txt.get_width() - 4, BOARD_Y + BOARD_PX - txt.get_height() - 2))
        # rank numbers (left edge)
        txt = coord_font.render(str(8 - i), True, C_DIM)
        _, y = sq_to_px(i, 0)
        surf.blit(txt, (BOARD_X + 3, y + 3))


def draw_legal_overlays(surf, pieces, selected, legal_sqs):
    """Draw dots on empty legal-move squares and rings on capture squares."""
    for (r, c) in legal_sqs:
        x, y = sq_to_px(r, c)
        cx, cy = x + SQ // 2, y + SQ // 2
        piece = pieces[idx(r, c)]
        if piece == '.':
            # Dot for empty square
            pygame.draw.circle(surf, C_DOT, (cx, cy), SQ // 9)
        else:
            # Ring for capture square
            pygame.draw.circle(surf, C_RING, (cx, cy), SQ // 2 - 3, 4)


def draw_piece(surf, piece_char: str, x: int, y: int, piece_font=None):
    """Draw a single piece centred in the square starting at (x, y)."""
    if piece_char == '.':
        return

    img = PIECE_IMAGES.get(piece_char)
    if img is None:
        return

    cx = x + SQ // 2
    cy = y + SQ // 2

    # Soft drop shadow — blit image slightly offset at reduced alpha
    shadow = img.copy()
    shadow.set_alpha(60)
    surf.blit(shadow, img.get_rect(center=(cx + 2, cy + 3)))

    # Draw actual piece centred on the square
    surf.blit(img, img.get_rect(center=(cx, cy)))


def draw_pieces(surf, pieces, piece_font):
    for r in range(8):
        for c in range(8):
            p = pieces[idx(r, c)]
            if p != '.':
                x, y = sq_to_px(r, c)
                draw_piece(surf, p, x, y, piece_font)


def draw_panel(surf, status: str, move_log: list,
               panel_font, header_font, small_font, thinking: bool):
    """Draw the right-hand info panel."""
    px, pw, ph = PANEL_X, PANEL_W, WIN_H

    # Panel background
    pygame.draw.rect(surf, C_PANEL, (px, 0, pw + 20, ph))
    pygame.draw.line(surf, C_BORDER, (px, 0), (px, ph), 1)

    y = 28

    # Title
    title = header_font.render("CHESS", True, C_ACCENT)
    surf.blit(title, (px + (pw - title.get_width()) // 2, y))
    y += title.get_height() + 6

    # Divider
    pygame.draw.line(surf, C_BORDER,
                     (px + 16, y), (px + pw - 16, y), 1)
    y += 14

    # Status
    parts = status.split()
    status_text = {
        "ongoing":    lambda p: ("Your turn" if p[1] == "white" else "Engine thinking…"),
        "check":      lambda p: ("⚠ You are in check" if p[1] == "white" else "⚠ Engine in check"),
        "checkmate":  lambda p: (f"★ {'You win!' if p[1] == 'white' else 'Engine wins.'}" ),
        "stalemate":  lambda p: "½ Stalemate — draw",
        "draw":       lambda p: "½ Draw by repetition",
    }.get(parts[0], lambda p: status)(parts)

    if thinking:
        status_text = "Engine thinking…"

    color = C_ACCENT if "win" in status_text or "★" in status_text else C_TEXT
    st = panel_font.render(status_text, True, color)
    surf.blit(st, (px + (pw - st.get_width()) // 2, y))
    y += st.get_height() + 18

    # Divider
    pygame.draw.line(surf, C_BORDER,
                     (px + 16, y), (px + pw - 16, y), 1)
    y += 12

    # Move log header
    lh = small_font.render("Move history", True, C_DIM)
    surf.blit(lh, (px + 16, y))
    y += lh.get_height() + 8

    # Moves (most recent at top, show last ~20)
    visible = move_log[-20:][::-1]
    for i, entry in enumerate(visible):
        col = C_TEXT if i == 0 else C_DIM
        mt  = small_font.render(entry, True, col)
        surf.blit(mt, (px + 20, y))
        y   += mt.get_height() + 3
        if y > WIN_H - 80:
            break

    # New-game button
    btn_rect = pygame.Rect(px + 20, WIN_H - 56, pw - 40, 36)
    hovered  = btn_rect.collidepoint(pygame.mouse.get_pos())
    btn_col  = C_ACCENT if hovered else (60, 55, 45)
    pygame.draw.rect(surf, btn_col, btn_rect, border_radius=6)
    pygame.draw.rect(surf, C_BORDER, btn_rect, 1, border_radius=6)
    bn = panel_font.render("New game", True, C_BG if hovered else C_TEXT)
    surf.blit(bn, bn.get_rect(center=btn_rect.center))

    return btn_rect


def draw_promotion_dialog(surf, dialog_font):
    """Overlay asking the player to pick a promotion piece."""
    options = [('Q', 'Queen'), ('R', 'Rook'), ('B', 'Bishop'), ('N', 'Knight')]
    box_w, box_h = 300, 180
    bx = (WIN_W - box_w) // 2
    by = (WIN_H - box_h) // 2

    # Dim background
    overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 140))
    surf.blit(overlay, (0, 0))

    pygame.draw.rect(surf, C_PANEL, (bx, by, box_w, box_h), border_radius=10)
    pygame.draw.rect(surf, C_BORDER, (bx, by, box_w, box_h), 1, border_radius=10)

    heading = dialog_font.render("Promote to:", True, C_ACCENT)
    surf.blit(heading, (bx + (box_w - heading.get_width()) // 2, by + 14))

    rects = {}
    btn_w, btn_h = 56, 44
    gap = (box_w - 4 * btn_w) // 5
    bby = by + 60
    for i, (key, label) in enumerate(options):
        bx2 = bx + gap + i * (btn_w + gap)
        r   = pygame.Rect(bx2, bby, btn_w, btn_h)
        hov = r.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(surf, C_ACCENT if hov else (55, 50, 42), r, border_radius=6)
        pygame.draw.rect(surf, C_BORDER, r, 1, border_radius=6)
        lt = dialog_font.render(key, True, C_BG if hov else C_TEXT)
        surf.blit(lt, lt.get_rect(center=r.center))
        # label below
        ll = pygame.font.SysFont("Georgia", 11).render(label, True, C_DIM)
        surf.blit(ll, (r.centerx - ll.get_width() // 2, bby + btn_h + 4))
        rects[key.lower()] = r

    return rects


# ══════════════════════════════════════════════════════════════
# Game state
# ══════════════════════════════════════════════════════════════

class GameState:
    def __init__(self):
        self.reset()

    def reset(self):
        engine_newgame()
        self.pieces     = engine_board()    # list[64]
        self.legal      = engine_legal()    # ['e2e4', ...]
        self.status     = engine_status()   # 'ongoing white'
        self.selected   = None             # (row, col) or None
        self.legal_sqs  = set()            # destination (row,col) for selected piece
        self.thinking   = False
        self.move_log   = []
        self.promoting  = None             # (r1,c1,r2,c2) pending promotion choice
        self.pending_uci = None            # uci string waiting for promotion suffix

    def is_game_over(self):
        return self.status.startswith(("checkmate", "stalemate", "draw"))

    def find_king_in_check(self) -> tuple | None:
        if not self.status.startswith("check"):
            return None
        side = self.status.split()[1]
        king_char = 'K' if side == 'white' else 'k'
        for r in range(8):
            for c in range(8):
                if self.pieces[idx(r, c)] == king_char:
                    return (r, c)
        return None

    def legal_dests_for(self, row: int, col: int) -> set:
        """Return destination squares reachable from (row, col)."""
        prefix = f"{chr(ord('a') + col)}{8 - row}"
        dests  = set()
        for uci in self.legal:
            if uci[:2] == prefix:
                tc = ord(uci[2]) - ord('a')
                tr = 8 - int(uci[3])
                dests.add((tr, tc))
        return dests

    def uci_for_move(self, r1, c1, r2, c2):
        """Return the matching UCI string(s) for a board move (may be multiple for promo)."""
        fr = chr(ord('a') + c1) + str(8 - r1)
        to = chr(ord('a') + c2) + str(8 - r2)
        matches = [m for m in self.legal if m[:4] == fr + to]
        return matches

    def human_move(self, uci: str) -> bool:
        if not engine_move(uci):
            return False
        self.pieces   = engine_board()
        self.status   = engine_status()
        self.selected = None
        self.legal_sqs = set()
        letters = "abcdefgh"
        r1 = 8 - int(uci[1]); c1 = ord(uci[0]) - ord('a')
        r2 = 8 - int(uci[3]); c2 = ord(uci[2]) - ord('a')
        self.move_log.append(f"You:  {uci[0]}{uci[1]}→{uci[2]}{uci[3]}")
        return True

    def after_engine_move(self, resp: str):
        self.thinking = False
        if resp.startswith("bestmove"):
            uci = resp.split()[1]
            self.pieces  = engine_board()
            self.status  = engine_status()
            self.legal   = engine_legal()
            self.move_log.append(f"Eng:  {uci[0]}{uci[1]}→{uci[2]}{uci[3]}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    pygame.init()
    start_engine()

    screen = pygame.display.set_mode((WIN_W, WIN_H))
    load_piece_images()
    pygame.display.set_caption("Chess")
    clock = pygame.time.Clock()

    # Fonts
    piece_font  = None  # PNGs are used; font kept for future use
    panel_font  = pygame.font.SysFont("Georgia",        16)
    header_font = pygame.font.SysFont("Georgia",        22, bold=True)
    small_font  = pygame.font.SysFont("Consolas",       13)
    dialog_font = pygame.font.SysFont("Georgia",        18, bold=True)

    state = GameState()

    running = True
    while running:
        clock.tick(60)

        # ── Board surface ─────────────────────────────────────
        screen.fill(C_BG)

        check_sq = state.find_king_in_check()
        draw_board(screen, state.selected, state.legal_sqs, check_sq)
        draw_legal_overlays(screen, state.pieces, state.selected, state.legal_sqs)
        draw_pieces(screen, state.pieces, piece_font)

        # Board border
        pygame.draw.rect(screen, C_BORDER,
                         (BOARD_X - 2, BOARD_Y - 2, BOARD_PX + 4, BOARD_PX + 4), 1)

        # ── Panel ─────────────────────────────────────────────
        btn_rect = draw_panel(screen, state.status, state.move_log,
                              panel_font, header_font, small_font, state.thinking)

        # ── Promotion dialog ──────────────────────────────────
        promo_rects = {}
        if state.promoting:
            promo_rects = draw_promotion_dialog(screen, dialog_font)

        pygame.display.flip()

        # ── Events ────────────────────────────────────────────
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            # ── Engine finished thinking ──────────────────────
            elif event.type == ENGINE_DONE:
                state.after_engine_move(event.resp)

            # ── Mouse click ───────────────────────────────────
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos

                # New-game button
                if btn_rect.collidepoint(mx, my):
                    state.reset()
                    continue

                # Promotion dialog click
                if state.promoting:
                    for promo_key, rect in promo_rects.items():
                        if rect.collidepoint(mx, my):
                            uci = state.pending_uci + promo_key
                            if state.human_move(uci):
                                state.promoting   = None
                                state.pending_uci = None
                                # Trigger engine if game not over
                                if not state.is_game_over():
                                    state.thinking = True
                                    engine_go_async(DEPTH)
                    continue

                # Board click (human turn only, not thinking)
                if state.thinking or state.is_game_over():
                    continue
                if not state.status.startswith(("ongoing white", "check white")):
                    continue

                sq = px_to_sq(mx, my)
                if sq is None:
                    state.selected  = None
                    state.legal_sqs = set()
                    continue

                r, c = sq

                # If a piece is already selected and this is a legal destination
                if state.selected and sq in state.legal_sqs:
                    r1, c1 = state.selected
                    matches = state.uci_for_move(r1, c1, r, c)

                    if len(matches) > 1:
                        # Multiple matches → promotion
                        state.promoting   = (r1, c1, r, c)
                        state.pending_uci = matches[0][:4]
                    elif matches:
                        uci = matches[0]
                        if state.human_move(uci):
                            if not state.is_game_over():
                                state.thinking = True
                                engine_go_async(DEPTH)
                    else:
                        state.selected  = None
                        state.legal_sqs = set()

                else:
                    # Select a piece (only own pieces — white pieces are uppercase)
                    piece = state.pieces[idx(r, c)]
                    if piece != '.' and piece.isupper():
                        state.selected  = (r, c)
                        state.legal_sqs = state.legal_dests_for(r, c)
                    else:
                        state.selected  = None
                        state.legal_sqs = set()

    engine_quit()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()